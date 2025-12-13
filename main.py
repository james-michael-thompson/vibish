#!/usr/bin/env python3
"""
Issue Embeddings - Find GitHub issues by concept using semantic search.

Usage:
    # Fetch issues from GitHub
    uv run main.py fetch --max-per-repo 1000

    # Build embedding index
    uv run main.py index

    # Search by concept
    uv run main.py search --concept race_conditions

    # Free-form search
    uv run main.py search --query "pod scheduling fails with taints"

    # Interactive mode
    uv run main.py search --interactive

    # Full pipeline (fetch + index + search all concepts)
    uv run main.py run --max-per-repo 500
"""

import argparse
import sys

from fetch_issues import fetch_and_save_issues, load_issues
from embeddings import IssueEmbeddings, create_index_from_issues
from search import (
    CONCEPTS,
    search_concept,
    search_all_concepts,
    format_results,
    interactive_search,
)


def cmd_fetch(args):
    """Fetch issues from GitHub."""
    counts = fetch_and_save_issues(
        output_dir=args.output_dir,
        max_per_repo=args.max_per_repo,
    )
    total = sum(counts.values())
    print(f"\nFetched {total} total issues")


def cmd_index(args):
    """Build embedding index."""
    create_index_from_issues(
        issues_dir=args.issues_dir,
        index_dir=args.index_dir,
        model_name=args.model,
    )


def cmd_search(args):
    """Search for issues."""
    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    if args.interactive:
        interactive_search(embedder)
    elif args.concept:
        results = search_concept(embedder, args.concept, k=args.k)
        print(f"\nResults for concept '{args.concept}':\n")
        print(format_results(results, max_display=args.k))
    elif args.query:
        results = embedder.search(args.query, k=args.k)
        print(f"\nResults for query '{args.query}':\n")
        print(format_results(results, max_display=args.k))
    else:
        print("Searching all concepts...\n")
        all_results = search_all_concepts(embedder, k=5)
        for concept, results in all_results.items():
            print(f"\n{'=' * 60}")
            print(f"CONCEPT: {concept}")
            print("=" * 60)
            print(format_results(results, max_display=5))


def cmd_run(args):
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


def main():
    parser = argparse.ArgumentParser(
        description="Find GitHub issues by concept using semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch issues from GitHub")
    fetch_parser.add_argument(
        "--max-per-repo", type=int, help="Max issues per repository"
    )
    fetch_parser.add_argument(
        "--output-dir", default="data/issues", help="Output directory for issues"
    )

    # index command
    index_parser = subparsers.add_parser("index", help="Build embedding index")
    index_parser.add_argument(
        "--issues-dir", default="data/issues", help="Issues directory"
    )
    index_parser.add_argument(
        "--index-dir", default="data/index", help="Output index directory"
    )
    index_parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Sentence transformer model"
    )

    # search command
    search_parser = subparsers.add_parser("search", help="Search for issues")
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
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )

    # run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument(
        "--max-per-repo", type=int, default=500, help="Max issues per repo"
    )
    run_parser.add_argument(
        "--output-dir", default="data/issues", help="Issues directory"
    )
    run_parser.add_argument(
        "--index-dir", default="data/index", help="Index directory"
    )

    args = parser.parse_args()

    commands = {
        "fetch": cmd_fetch,
        "index": cmd_index,
        "search": cmd_search,
        "run": cmd_run,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
