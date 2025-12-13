#!/usr/bin/env python3
"""
vibish - Find GitHub issues by vibe using semantic search.

Usage:
    vibish help                     Show this help
    vibish fetch                    Download issues from GitHub
    vibish index build              Build the embedding index
    vibish index summarize          Show index stats
    vibish index nuke               Delete the index (with one recovery)
    vibish search -q "memory leak"  Find issues by query
    vibish search --concept race_conditions
    vibish vibe                     Do it all: fetch, index, search
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from fetch_issues import fetch_and_save_issues, load_issues
from embeddings import IssueEmbeddings, create_index_from_issues
from search import (
    CONCEPTS,
    load_concepts_with_descriptions,
    search_concept,
    search_all_concepts,
    search_with_prompts,
    refine_search,
    format_results,
    interactive_search,
)
from discover import discover_vibes, find_optimal_k, format_vibes


HELP_TEXT = """
vibish - Find GitHub issues by vibe using semantic search

Commands:
  help                  Show this help message
  fetch                 Download issues from GitHub repos
  issues <subcommand>   Manage fetched issues
  index <subcommand>    Manage the embedding index
  search                Find issues by concept or free-form query
  discover              Find natural clusters (vibes) using GMM
  concepts              List available concepts and their prompts
  vibe                  Fetch, index, and search all in one go
  nuke                  Delete everything (issues + index)

Issues subcommands:
  issues summarize      Show issue statistics
  issues nuke           Delete fetched issues (renames to .nuked for recovery)
  issues help           Show issues-specific help

Index subcommands:
  index build           Build searchable embeddings from fetched issues
  index summarize       Show index statistics
  index nuke            Delete the index (renames to .nuked for recovery)
  index help            Show index-specific help

Examples:
  vibish fetch                           Download issues from GitHub
  vibish index build                     Build the embedding index
  vibish search -q "memory leak"         Find issues about memory leaks
  vibish search --concept race_conditions
  vibish search -q "deadlock" -s open    Search open issues only
  vibish search -p "race condition" -p "deadlock"   Custom prompts
  vibish search -p "race condition" -r 3  Refine search over 3 iterations
  vibish search -i                       Interactive mode (use 'open: query')
  vibish discover -k 20                  Discover 20 natural clusters
  vibish discover --find-k               Find optimal number of clusters
  vibish concepts                        Show all concepts and prompts
  vibish vibe                            Do fetch + index + search
  vibish nuke                            Start fresh

Search options:
  -q, --query QUERY     Free-form search query
  --concept CONCEPT     Search by predefined concept (from concepts.json)
  -p, --prompt PROMPT   Custom prompt (can be repeated for multi-prompt search)
  -s, --state STATE     Filter by state: open or closed
  -k N                  Number of results (default: 10)
  -r, --refine N        Refine search over N iterations using centroid of results
  --anchor-weight W     Weight for original prompts during refinement (0-1, default: 0.3)
  -i, --interactive     Interactive search mode

Refinement uses pseudo-relevance feedback: it computes the centroid of top
results and searches again from that point. The anchor weight controls drift
from your original prompts (higher = more stable, lower = more exploration).

Concepts are defined in concepts.json. Use 'vibish concepts' to see them.
"""

INDEX_HELP_TEXT = """
vibish index - Manage the embedding index

Subcommands:
  build       Build searchable embeddings from fetched issues
  summarize   Show index statistics (size, issue count, model, etc.)
  nuke        Delete the index (renames to .nuked for one recovery)
  help        Show this help message

Examples:
  vibish index build                     Build the index
  vibish index build --model all-mpnet-base-v2   Use a different model
  vibish index summarize                 Show index stats
  vibish index nuke                      Delete the index

Options for 'build':
  --issues-dir DIR    Source directory for issues (default: data/issues)
  --index-dir DIR     Output directory for index (default: data/index)
  --model MODEL       Sentence transformer model (default: all-MiniLM-L6-v2)
"""

ISSUES_HELP_TEXT = """
vibish issues - Manage fetched issues

Subcommands:
  summarize   Show issue statistics (count by repo, date range, etc.)
  nuke        Delete fetched issues (renames to .nuked for one recovery)
  help        Show this help message

Examples:
  vibish issues summarize                Show issue stats
  vibish issues nuke                     Delete fetched issues
"""


def cmd_help(args):
    """Show overall help."""
    print(HELP_TEXT)


def cmd_concepts(args):
    """List available concepts and their prompts."""
    concepts = load_concepts_with_descriptions()

    if not concepts:
        print("No concepts defined. Create concepts.json to add some.")
        return

    print("Available Concepts")
    print("=" * 60)

    for name, config in concepts.items():
        desc = config.get("description", "")
        prompts = config.get("prompts", [])

        print(f"\n{name}")
        if desc:
            print(f"  {desc}")
        print(f"  Prompts ({len(prompts)}):")
        for prompt in prompts:
            print(f"    - {prompt}")

    print(f"\nConcepts are defined in: concepts.json")
    print("Edit this file to add or modify concepts.")


def cmd_fetch(args):
    """Fetch issues from GitHub."""
    counts = fetch_and_save_issues(
        output_dir=args.output_dir,
        max_per_repo=args.max_per_repo,
    )
    total = sum(counts.values())
    print(f"\nFetched {total} total issues")


def cmd_issues(args):
    """Handle issues subcommands."""
    if args.issues_command == "help" or args.issues_command is None:
        print(ISSUES_HELP_TEXT)
    elif args.issues_command == "summarize":
        cmd_issues_summarize(args)
    elif args.issues_command == "nuke":
        cmd_issues_nuke(args)


def cmd_issues_summarize(args):
    """Show issue statistics."""
    issues_dir = Path(args.issues_dir)

    if not issues_dir.exists():
        print(f"No issues found at {issues_dir}")
        print("Run 'vibish fetch' to download issues.")
        return

    issues = load_issues(str(issues_dir))

    if not issues:
        print("No issues loaded.")
        return

    # Count by repo
    repo_counts = {}
    open_counts = {}
    for issue in issues:
        repo = issue.get("repo", "unknown")
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if issue.get("state") == "open":
            open_counts[repo] = open_counts.get(repo, 0) + 1

    # Get total size
    total_size = sum(f.stat().st_size for f in issues_dir.glob("*.json"))

    print("Issues Summary")
    print("=" * 50)
    print(f"Location:      {issues_dir}")
    print(f"Total issues:  {len(issues)}")
    print(f"Total size:    {total_size / 1024 / 1024:.2f} MB")
    print()
    print("Issues by repo:")
    for repo in sorted(repo_counts.keys()):
        total = repo_counts[repo]
        open_count = open_counts.get(repo, 0)
        print(f"  {repo}: {total} ({open_count} open)")


def cmd_issues_nuke(args):
    """Delete fetched issues (with one recovery option)."""
    issues_dir = Path(args.issues_dir)
    nuked_dir = Path(str(issues_dir) + ".nuked")

    if not issues_dir.exists():
        print(f"No issues found at {issues_dir}")
        return

    if nuked_dir.exists():
        print(f"WARNING: {nuked_dir} already exists!")
        print("You already have nuked issues waiting for recovery.")
        print("Either restore them or delete manually before nuking again.")
        print(f"\n  rm -rf {nuked_dir}  # to permanently delete")
        print(f"  mv {nuked_dir} {issues_dir}  # to restore")
        return

    # Rename to .nuked
    issues_dir.rename(nuked_dir)
    print(f"Issues moved to {nuked_dir}")
    print(f"To recover: mv {nuked_dir} {issues_dir}")
    print(f"To permanently delete: rm -rf {nuked_dir}")


def cmd_index(args):
    """Handle index subcommands."""
    if args.index_command == "help" or args.index_command is None:
        print(INDEX_HELP_TEXT)
    elif args.index_command == "build":
        cmd_index_build(args)
    elif args.index_command == "summarize":
        cmd_index_summarize(args)
    elif args.index_command == "nuke":
        cmd_index_nuke(args)


def cmd_index_build(args):
    """Build the embedding index."""
    create_index_from_issues(
        issues_dir=args.issues_dir,
        index_dir=args.index_dir,
        model_name=args.model,
    )


def cmd_index_summarize(args):
    """Show index statistics."""
    index_dir = Path(args.index_dir)

    if not index_dir.exists():
        print(f"No index found at {index_dir}")
        print("Run 'vibish index build' to create one.")
        return

    # Load config
    config_file = index_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        model_name = config.get("model_name", "unknown")
    else:
        model_name = "unknown"

    # Load issues metadata
    issues_file = index_dir / "issues.json"
    if issues_file.exists():
        with open(issues_file) as f:
            issues = json.load(f)

        # Count by repo
        repo_counts = {}
        for issue in issues:
            repo = issue.get("repo", "unknown")
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
    else:
        issues = []
        repo_counts = {}

    # Get file sizes
    index_file = index_dir / "issues.index"
    index_size = index_file.stat().st_size if index_file.exists() else 0
    issues_size = issues_file.stat().st_size if issues_file.exists() else 0

    # Get modification time
    if index_file.exists():
        mtime = datetime.fromtimestamp(index_file.stat().st_mtime)
        built_at = mtime.strftime("%Y-%m-%d %H:%M:%S")
    else:
        built_at = "never"

    print("Index Summary")
    print("=" * 50)
    print(f"Location:      {index_dir}")
    print(f"Model:         {model_name}")
    print(f"Total issues:  {len(issues)}")
    print(f"Index size:    {index_size / 1024 / 1024:.2f} MB")
    print(f"Metadata size: {issues_size / 1024 / 1024:.2f} MB")
    print(f"Built at:      {built_at}")
    print()
    print("Issues by repo:")
    for repo, count in sorted(repo_counts.items()):
        print(f"  {repo}: {count}")


def cmd_index_nuke(args):
    """Delete the index (with one recovery option)."""
    index_dir = Path(args.index_dir)
    nuked_dir = Path(str(index_dir) + ".nuked")

    if not index_dir.exists():
        print(f"No index found at {index_dir}")
        return

    if nuked_dir.exists():
        print(f"WARNING: {nuked_dir} already exists!")
        print("You already have a nuked index waiting for recovery.")
        print("Either restore it or delete it manually before nuking again.")
        print(f"\n  rm -rf {nuked_dir}  # to permanently delete")
        print(f"  mv {nuked_dir} {index_dir}  # to restore")
        return

    # Rename to .nuked
    index_dir.rename(nuked_dir)
    print(f"Index moved to {nuked_dir}")
    print(f"To recover: mv {nuked_dir} {index_dir}")
    print(f"To permanently delete: rm -rf {nuked_dir}")


def filter_by_state(results: list[tuple[dict, float]], state: str | None, k: int) -> list[tuple[dict, float]]:
    """Filter results by issue state (open/closed)."""
    if state is None:
        return results[:k]
    filtered = [(issue, score) for issue, score in results if issue.get("state") == state]
    return filtered[:k]


def cmd_search(args):
    """Search for issues."""
    # Check for incompatible options
    if args.interactive:
        if args.state:
            print("Note: --state is ignored in interactive mode.")
            print("Use 'open:' or 'closed:' prefix instead (e.g., 'open: memory leak')")
            print()
        if args.concept:
            print("Note: --concept is ignored in interactive mode.")
            print(f"Just type '{args.concept}' at the prompt.")
            print()
        if args.query:
            print("Note: --query is ignored in interactive mode.")
            print(f"Just type your query at the prompt.")
            print()
        if args.prompts:
            print("Note: --prompt is ignored in interactive mode.")
            print("Just type your prompts at the prompt.")
            print()

    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    # Fetch more results if filtering, to ensure we get enough after filter
    fetch_k = args.k * 5 if args.state else args.k

    if args.interactive:
        interactive_search(embedder)
    elif args.prompts:
        if args.refine:
            # Iterative refinement mode
            print(f"\nRefining search over {args.refine} iteration(s)...")
            print("Prompts used:")
            for p in args.prompts:
                print(f"  - {p}")
            print()

            refinement = refine_search(
                embedder,
                args.prompts,
                iterations=args.refine,
                top_k_for_centroid=10,
                results_k=fetch_k,
                anchor_weight=args.anchor_weight,
            )

            # Show drift history
            print("Refinement History")
            print("=" * 60)
            for h in refinement["history"]:
                stats = h["cluster_stats"]
                overlap = h["result_overlap"]
                print(f"\nIteration {h['iteration']}:")
                print(f"  Drift: {h['drift_from_original']['angular_distance_degrees']:.1f} deg from original, "
                      f"{h['drift_from_previous']['angular_distance_degrees']:.1f} deg from prev")
                print(f"  Cluster: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                      f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Results: {overlap['overlap_count']}/10 same, "
                      f"+{overlap['new_count']} new, -{overlap['dropped_count']} dropped")
                print(f"  Top: ", end="")
                top = [f"{r[1]}#{r[0]}" for r in h["top_results"][:3]]
                print(", ".join(top))

            # Summary comparison
            initial = refinement["initial_stats"]
            final = refinement["final_stats"]
            final_drift = refinement["final_drift_from_original"]

            print(f"\nSummary")
            print("=" * 60)
            print(f"Cluster quality improvement:")
            print(f"  Mean similarity: {initial['mean']:.3f} -> {final['mean']:.3f} "
                  f"({'+' if final['mean'] > initial['mean'] else ''}{final['mean'] - initial['mean']:.3f})")
            print(f"  Std deviation:   {initial['std']:.3f} -> {final['std']:.3f} "
                  f"({'tighter' if final['std'] < initial['std'] else 'looser'})")
            print(f"  Score range:     [{initial['min']:.3f}, {initial['max']:.3f}] -> "
                  f"[{final['min']:.3f}, {final['max']:.3f}]")
            print(f"\nCentroid drift: {final_drift['angular_distance_degrees']:.1f} degrees "
                  f"(cosine sim: {final_drift['cosine_similarity']:.3f})")

            results = refinement["results"]
            results = filter_by_state(results, args.state, args.k)
            state_msg = f" ({args.state} only)" if args.state else ""
            print(f"\nRefined results{state_msg}:\n")
            print(format_results(results, max_display=args.k))
        else:
            results = search_with_prompts(embedder, args.prompts, k=fetch_k)
            results = filter_by_state(results, args.state, args.k)
            state_msg = f" ({args.state} only)" if args.state else ""
            prompt_summary = f"{len(args.prompts)} prompt(s)"
            print(f"\nResults for {prompt_summary}{state_msg}:\n")
            print("Prompts used:")
            for p in args.prompts:
                print(f"  - {p}")
            print()
            print(format_results(results, max_display=args.k))
    elif args.concept:
        results = search_concept(embedder, args.concept, k=fetch_k)
        results = filter_by_state(results, args.state, args.k)
        state_msg = f" ({args.state} only)" if args.state else ""
        print(f"\nResults for concept '{args.concept}'{state_msg}:\n")
        print(format_results(results, max_display=args.k))
    elif args.query:
        results = embedder.search(args.query, k=fetch_k)
        results = filter_by_state(results, args.state, args.k)
        state_msg = f" ({args.state} only)" if args.state else ""
        print(f"\nResults for query '{args.query}'{state_msg}:\n")
        print(format_results(results, max_display=args.k))
    else:
        print("Searching all concepts...\n")
        all_results = search_all_concepts(embedder, k=fetch_k)
        for concept, results in all_results.items():
            filtered = filter_by_state(results, args.state, 5)
            print(f"\n{'=' * 60}")
            print(f"CONCEPT: {concept}")
            print("=" * 60)
            print(format_results(filtered, max_display=5))


def cmd_vibe(args):
    """Run full pipeline: fetch, index, and search."""
    print("Step 1: Fetching issues...")
    print("=" * 60)
    fetch_and_save_issues(
        output_dir=args.output_dir,
        max_per_repo=args.max_per_repo,
    )

    print("\nStep 2: Building index...")
    print("=" * 60)
    embedder = create_index_from_issues(
        issues_dir=args.output_dir,
        index_dir=args.index_dir,
    )

    print("\nStep 3: Searching concepts...")
    print("=" * 60)
    all_results = search_all_concepts(embedder, k=10)
    for concept, results in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"CONCEPT: {concept}")
        print("=" * 60)
        print(format_results(results, max_display=10))


def cmd_discover(args):
    """Discover natural clusters using GMM."""
    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    if args.find_k:
        print("Finding optimal number of clusters...")
        results = find_optimal_k(
            embedder,
            k_range=range(args.k_min, args.k_max + 1, 5),
            use_pca=True,
            pca_components=args.pca_dim,
        )
        print("\nResults:")
        print(f"{'k':>5} {'BIC':>12} {'AIC':>12}")
        print("-" * 31)
        for r in results:
            print(f"{r['k']:>5} {r['bic']:>12.0f} {r['aic']:>12.0f}")
        best = min(results, key=lambda r: r["bic"])
        print(f"\nBest k by BIC: {best['k']}")
        print(f"Run 'vibish discover -k {best['k']}' to see those clusters")
    else:
        result = discover_vibes(
            embedder,
            n_vibes=args.k,
            top_k_per_vibe=args.top_k,
            use_pca=True,
            pca_components=args.pca_dim,
        )
        print("\n" + format_vibes(result, max_issues=args.top_k))


def cmd_nuke(args):
    """Delete everything: issues and index."""
    import shutil

    issues_dir = Path(args.issues_dir)
    index_dir = Path(args.index_dir)
    nuked_issues = Path(str(issues_dir) + ".nuked")
    nuked_index = Path(str(index_dir) + ".nuked")

    nuked_something = False

    # Handle issues
    if issues_dir.exists():
        if nuked_issues.exists():
            print(f"WARNING: {nuked_issues} already exists!")
            print("Delete it manually before nuking again.")
        else:
            issues_dir.rename(nuked_issues)
            print(f"Issues moved to {nuked_issues}")
            nuked_something = True
    else:
        print(f"No issues found at {issues_dir}")

    # Handle index
    if index_dir.exists():
        if nuked_index.exists():
            print(f"WARNING: {nuked_index} already exists!")
            print("Delete it manually before nuking again.")
        else:
            index_dir.rename(nuked_index)
            print(f"Index moved to {nuked_index}")
            nuked_something = True
    else:
        print(f"No index found at {index_dir}")

    if nuked_something:
        print("\nTo recover:")
        if nuked_issues.exists():
            print(f"  mv {nuked_issues} {issues_dir}")
        if nuked_index.exists():
            print(f"  mv {nuked_index} {index_dir}")
        print("\nTo permanently delete:")
        print(f"  rm -rf {nuked_issues} {nuked_index}")


def main():
    parser = argparse.ArgumentParser(
        description="vibish - Find GitHub issues by vibe using semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    # help command
    help_parser = subparsers.add_parser("help", help="Show help message", add_help=False)

    # fetch command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download issues from GitHub repos",
        description="Fetches issues from configured GitHub repos and saves them locally.",
    )
    fetch_parser.add_argument(
        "--max-per-repo", type=int, help="Max issues per repository"
    )
    fetch_parser.add_argument(
        "--output-dir", default="data/issues", help="Output directory for issues"
    )

    # issues command with subcommands
    issues_parser = subparsers.add_parser(
        "issues",
        help="Manage fetched issues",
        description="Summarize or delete fetched issues.",
    )
    issues_subparsers = issues_parser.add_subparsers(dest="issues_command")

    # issues summarize
    issues_summarize = issues_subparsers.add_parser("summarize", help="Show issue statistics")
    issues_summarize.add_argument(
        "--issues-dir", default="data/issues", help="Issues directory"
    )

    # issues nuke
    issues_nuke = issues_subparsers.add_parser("nuke", help="Delete fetched issues")
    issues_nuke.add_argument(
        "--issues-dir", default="data/issues", help="Issues directory"
    )

    # issues help
    issues_subparsers.add_parser("help", help="Show issues help")

    # index command with subcommands
    index_parser = subparsers.add_parser(
        "index",
        help="Manage the embedding index",
        description="Build, summarize, or delete the embedding index.",
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command")

    # index build
    index_build = index_subparsers.add_parser("build", help="Build the embedding index")
    index_build.add_argument(
        "--issues-dir", default="data/issues", help="Issues directory"
    )
    index_build.add_argument(
        "--index-dir", default="data/index", help="Output index directory"
    )
    index_build.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Sentence transformer model"
    )

    # index summarize
    index_summarize = index_subparsers.add_parser("summarize", help="Show index statistics")
    index_summarize.add_argument(
        "--index-dir", default="data/index", help="Index directory"
    )

    # index nuke
    index_nuke = index_subparsers.add_parser("nuke", help="Delete the index")
    index_nuke.add_argument(
        "--index-dir", default="data/index", help="Index directory"
    )

    # index help
    index_subparsers.add_parser("help", help="Show index help")

    # concepts command
    concepts_parser = subparsers.add_parser(
        "concepts",
        help="List available concepts and their prompts",
        description="Show all concepts defined in concepts.json.",
    )

    # search command
    search_parser = subparsers.add_parser(
        "search",
        help="Find issues by concept or free-form query",
        description="Search the index using predefined concepts or your own queries.",
    )
    search_parser.add_argument("--index-dir", default="data/index", help="Index directory")
    search_parser.add_argument(
        "--concept",
        choices=list(CONCEPTS.keys()),
        help="Predefined concept to search",
    )
    search_parser.add_argument("--query", "-q", help="Free-form search query")
    search_parser.add_argument(
        "--prompt", "-p", dest="prompts", action="append",
        help="Custom prompt (can be repeated for multi-prompt search)"
    )
    search_parser.add_argument(
        "-k", type=int, default=10, help="Number of results (default: 10)"
    )
    search_parser.add_argument(
        "--state", "-s", choices=["open", "closed"], help="Filter by issue state"
    )
    search_parser.add_argument(
        "--refine", "-r", type=int, metavar="N",
        help="Refine search over N iterations using centroid of top results"
    )
    search_parser.add_argument(
        "--anchor-weight", type=float, default=0.3,
        help="Weight for original prompt centroid during refinement (0-1, default: 0.3)"
    )
    search_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )

    # discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Find natural clusters (vibes) using GMM",
        description="Use Gaussian Mixture Models to discover latent clusters in the issue embedding space.",
    )
    discover_parser.add_argument("--index-dir", default="data/index", help="Index directory")
    discover_parser.add_argument(
        "-k", type=int, default=10, help="Number of clusters to discover (default: 10)"
    )
    discover_parser.add_argument(
        "--top-k", type=int, default=3, help="Representative issues per cluster (default: 3)"
    )
    discover_parser.add_argument(
        "--pca-dim", type=int, default=50, help="PCA dimensions (default: 50)"
    )
    discover_parser.add_argument(
        "--find-k", action="store_true", help="Find optimal k using BIC/AIC"
    )
    discover_parser.add_argument(
        "--k-min", type=int, default=5, help="Min k for --find-k (default: 5)"
    )
    discover_parser.add_argument(
        "--k-max", type=int, default=30, help="Max k for --find-k (default: 30)"
    )

    # vibe command (full pipeline)
    vibe_parser = subparsers.add_parser(
        "vibe",
        help="Fetch, index, and search all in one go",
        description="Runs the full pipeline: fetches issues, builds index, then searches all concepts.",
    )
    vibe_parser.add_argument(
        "--max-per-repo", type=int, help="Max issues per repo (default: all)"
    )
    vibe_parser.add_argument(
        "--output-dir", default="data/issues", help="Issues directory"
    )
    vibe_parser.add_argument(
        "--index-dir", default="data/index", help="Index directory"
    )

    # nuke command (delete everything)
    nuke_parser = subparsers.add_parser(
        "nuke",
        help="Delete everything (issues + index)",
        description="Moves issues and index to .nuked directories for recovery.",
    )
    nuke_parser.add_argument(
        "--issues-dir", default="data/issues", help="Issues directory"
    )
    nuke_parser.add_argument(
        "--index-dir", default="data/index", help="Index directory"
    )

    args = parser.parse_args()

    # Handle no command or help
    if args.command is None or args.command == "help":
        print(HELP_TEXT)
        return

    commands = {
        "fetch": cmd_fetch,
        "issues": cmd_issues,
        "index": cmd_index,
        "concepts": cmd_concepts,
        "search": cmd_search,
        "discover": cmd_discover,
        "vibe": cmd_vibe,
        "nuke": cmd_nuke,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
