"""
Search for issues related to specific concepts.
"""

import json
from pathlib import Path

from embeddings import IssueEmbeddings


DEFAULT_CONCEPTS_FILE = Path(__file__).parent / "concepts.json"


def load_concepts(concepts_file: Path | str | None = None) -> dict:
    """Load concepts from JSON file."""
    path = Path(concepts_file) if concepts_file else DEFAULT_CONCEPTS_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    # Extract just the prompts for backward compatibility
    return {name: config["prompts"] for name, config in data.items()}


def load_concepts_with_descriptions(concepts_file: Path | str | None = None) -> dict:
    """Load concepts with descriptions from JSON file."""
    path = Path(concepts_file) if concepts_file else DEFAULT_CONCEPTS_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# Load default concepts for backward compatibility
CONCEPTS = load_concepts()


def get_concept_queries(concept_name: str, concepts: dict | None = None) -> list[str]:
    """Get query phrases for a concept."""
    concepts = concepts or CONCEPTS
    if concept_name not in concepts:
        raise ValueError(f"Unknown concept: {concept_name}. Known: {list(concepts.keys())}")
    return concepts[concept_name]


def search_with_prompts(
    embedder: IssueEmbeddings,
    prompts: list[str],
    k: int = 20,
) -> list[tuple[dict, float]]:
    """
    Search for issues using multiple prompts.

    Combines results from all prompts, keeping best score per issue.
    Returns deduplicated results sorted by best score.
    """
    all_results = {}  # issue_id -> (issue, best_score)

    for prompt in prompts:
        results = embedder.search(prompt, k=k)
        for issue, score in results:
            issue_id = issue["id"]
            if issue_id not in all_results or score > all_results[issue_id][1]:
                all_results[issue_id] = (issue, score)

    sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k]


def search_concept(
    embedder: IssueEmbeddings,
    concept_name: str,
    k: int = 20,
    dedupe: bool = True,
) -> list[tuple[dict, float]]:
    """
    Search for issues related to a concept.

    Uses multiple query phrases and combines results.
    Returns deduplicated results sorted by best score.
    """
    queries = get_concept_queries(concept_name)

    all_results = {}  # issue_id -> (issue, best_score)

    for query in queries:
        results = embedder.search(query, k=k)
        for issue, score in results:
            issue_id = issue["id"]
            if issue_id not in all_results or score > all_results[issue_id][1]:
                all_results[issue_id] = (issue, score)

    # Sort by score descending
    sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)

    return sorted_results[:k]


def search_all_concepts(
    embedder: IssueEmbeddings,
    k: int = 20,
) -> dict[str, list[tuple[dict, float]]]:
    """Search for issues related to all predefined concepts."""
    return {
        concept: search_concept(embedder, concept, k=k)
        for concept in CONCEPTS
    }


def format_results(
    results: list[tuple[dict, float]],
    max_display: int = 10,
) -> str:
    """Format search results for display."""
    lines = []
    for i, (issue, score) in enumerate(results[:max_display], 1):
        repo_short = issue["repo"].split("/")[-1]
        state_marker = "[closed]" if issue["state"] == "closed" else "[open]"
        lines.append(
            f"{i:2d}. [{score:.3f}] {repo_short}#{issue['number']} {state_marker}"
        )
        lines.append(f"    {issue['title'][:80]}")
        lines.append(f"    {issue['url']}")
        lines.append("")

    return "\n".join(lines)


def interactive_search(embedder: IssueEmbeddings) -> None:
    """Interactive search mode."""
    print("\nConcept Search")
    print("=" * 50)
    print(f"Available concepts: {', '.join(CONCEPTS.keys())}")
    print("Or enter a free-form query")
    print("Prefix with 'open:' or 'closed:' to filter by state")
    print("Type 'quit' to exit\n")

    while True:
        query = input("Search: ").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        # Parse state filter prefix
        state_filter = None
        if query.lower().startswith("open:"):
            state_filter = "open"
            query = query[5:].strip()
        elif query.lower().startswith("closed:"):
            state_filter = "closed"
            query = query[7:].strip()

        if not query:
            print("Please enter a query after the state filter")
            continue

        # Fetch more if filtering
        fetch_k = 50 if state_filter else 10

        if query in CONCEPTS:
            results = search_concept(embedder, query, k=fetch_k)
            label = f"concept '{query}'"
        else:
            results = embedder.search(query, k=fetch_k)
            label = f"query '{query}'"

        # Apply state filter
        if state_filter:
            results = [(issue, score) for issue, score in results if issue.get("state") == state_filter]
            results = results[:10]
            label += f" ({state_filter} only)"

        print(f"\nResults for {label}:")
        print(format_results(results))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search issues by concept")
    parser.add_argument("--index-dir", default="data/index", help="Index directory")
    parser.add_argument("--concept", help="Concept to search for")
    parser.add_argument("--query", help="Free-form query")
    parser.add_argument("-k", type=int, default=10, help="Number of results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    if args.interactive:
        interactive_search(embedder)
    elif args.concept:
        results = search_concept(embedder, args.concept, k=args.k)
        print(f"Results for concept '{args.concept}':")
        print(format_results(results, max_display=args.k))
    elif args.query:
        results = embedder.search(args.query, k=args.k)
        print(f"Results for query '{args.query}':")
        print(format_results(results, max_display=args.k))
    else:
        # Show results for all concepts
        print("Searching all concepts...\n")
        all_results = search_all_concepts(embedder, k=5)
        for concept, results in all_results.items():
            print(f"\n{'=' * 50}")
            print(f"CONCEPT: {concept}")
            print("=" * 50)
            print(format_results(results, max_display=5))
