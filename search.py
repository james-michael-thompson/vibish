"""
Search for issues related to specific concepts.
"""

from embeddings import IssueEmbeddings


# Predefined concepts for Karpenter/Kubernetes issues
# Each concept has multiple query phrases to capture different aspects
CONCEPTS = {
    "race_conditions": [
        "race condition concurrent access data race",
        "timing issue synchronization deadlock mutex lock contention",
        "concurrent modification thread safety atomicity",
        "parallel execution ordering conflict state corruption",
    ],
    "node_selection": [
        "node selection node filtering node affinity",
        "nodeSelector node scheduling pod placement",
        "taint toleration node matching constraint",
        "topology spread zone selection availability",
        "node requirements node constraints scheduling",
    ],
    "constraints": [
        "resource constraints limits requests quota",
        "scheduling constraints pod constraints",
        "node constraints instance type selection",
        "memory CPU resource allocation budget",
        "capacity constraints provisioner limits",
    ],
    "provisioning_performance": [
        "provisioning slow performance latency",
        "consolidation efficiency node lifecycle",
        "scale up delay provisioner speed",
        "node startup time boot performance",
        "scheduling throughput batch scheduling",
        "deprovisioning consolidation interruption",
    ],
}


def get_concept_queries(concept_name: str) -> list[str]:
    """Get query phrases for a concept."""
    if concept_name not in CONCEPTS:
        raise ValueError(f"Unknown concept: {concept_name}. Known: {list(CONCEPTS.keys())}")
    return CONCEPTS[concept_name]


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
    print("Type 'quit' to exit\n")

    while True:
        query = input("Search: ").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        if query in CONCEPTS:
            results = search_concept(embedder, query, k=10)
            print(f"\nResults for concept '{query}':")
        else:
            results = embedder.search(query, k=10)
            print(f"\nResults for query '{query}':")

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
