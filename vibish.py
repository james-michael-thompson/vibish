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
    search_concept,
    search_all_concepts,
    format_results,
    interactive_search,
)


HELP_TEXT = """
vibish - Find GitHub issues by vibe using semantic search

Commands:
  help                  Show this help message
  fetch                 Download issues from GitHub repos
  issues <subcommand>   Manage fetched issues
  index <subcommand>    Manage the embedding index
  search                Find issues by concept or free-form query
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
  vibish index summarize                 Show index stats
  vibish search -q "memory leak"         Find issues about memory leaks
  vibish search --concept race_conditions
  vibish search -q "deadlock" -s open    Search open issues only
  vibish search -i                       Interactive mode (use 'open: query')
  vibish vibe                            Do fetch + index + search
  vibish nuke                            Start fresh

Search options:
  -q, --query QUERY     Free-form search query
  --concept CONCEPT     Search by predefined concept
  -s, --state STATE     Filter by state: open or closed
  -k N                  Number of results (default: 10)
  -i, --interactive     Interactive search mode

Concepts:
  race_conditions          Data races, deadlocks, timing issues
  node_selection           Affinity, taints, topology, scheduling
  constraints              Resource limits, quotas, budgets
  provisioning_performance Scaling speed, consolidation, node lifecycle
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

    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    # Fetch more results if filtering, to ensure we get enough after filter
    fetch_k = args.k * 5 if args.state else args.k

    if args.interactive:
        interactive_search(embedder)
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
        "-k", type=int, default=10, help="Number of results (default: 10)"
    )
    search_parser.add_argument(
        "--state", "-s", choices=["open", "closed"], help="Filter by issue state"
    )
    search_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
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
        "search": cmd_search,
        "vibe": cmd_vibe,
        "nuke": cmd_nuke,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
