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


import numpy as np


def compute_centroid(
    embedder: IssueEmbeddings,
    results: list[tuple[dict, float]],
    top_k: int = 10,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute the centroid of top search results.

    Args:
        embedder: The embeddings manager
        results: List of (issue, score) tuples
        top_k: Number of top results to use
        weighted: If True, weight by score; if False, simple average

    Returns:
        Normalized centroid vector
    """
    top_results = results[:top_k]

    # Find indices of these issues in the embedder
    indices = []
    weights = []
    for issue, score in top_results:
        for i, emb_issue in enumerate(embedder.issues):
            if emb_issue["id"] == issue["id"]:
                indices.append(i)
                weights.append(score)
                break

    if not indices:
        raise ValueError("No matching issues found in index")

    # Get embeddings for these indices
    embeddings = embedder.get_embeddings_by_indices(indices)

    if weighted:
        # Weighted average by score
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        centroid = np.average(embeddings, axis=0, weights=weights)
    else:
        # Simple average
        centroid = np.mean(embeddings, axis=0)

    # Normalize the centroid
    centroid = centroid / np.linalg.norm(centroid)

    return centroid.astype(np.float32)


def compute_cluster_stats(results: list[tuple[dict, float]], top_k: int = 10) -> dict:
    """
    Compute statistics about the cluster quality of search results.

    Returns:
        Dict with mean, std, min, max of similarity scores for top-k results
    """
    scores = [score for _, score in results[:top_k]]
    if not scores:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def compute_result_overlap(
    results1: list[tuple[dict, float]],
    results2: list[tuple[dict, float]],
    top_k: int = 10,
) -> dict:
    """
    Compute overlap between two result sets.

    Returns:
        Dict with overlap count, jaccard similarity, and new/dropped issue IDs
    """
    ids1 = {r[0]["id"] for r in results1[:top_k]}
    ids2 = {r[0]["id"] for r in results2[:top_k]}

    intersection = ids1 & ids2
    union = ids1 | ids2

    return {
        "overlap_count": len(intersection),
        "jaccard": len(intersection) / len(union) if union else 0,
        "new_count": len(ids2 - ids1),
        "dropped_count": len(ids1 - ids2),
    }


def compute_drift(vec1: np.ndarray, vec2: np.ndarray) -> dict:
    """
    Compute drift metrics between two vectors.

    Returns:
        Dict with cosine_similarity, euclidean_distance, and angular_distance
    """
    # Ensure normalized
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    cosine_sim = float(np.dot(vec1, vec2))
    euclidean_dist = float(np.linalg.norm(vec1 - vec2))

    # Angular distance in degrees
    # Clamp to avoid numerical issues with arccos
    cosine_sim_clamped = np.clip(cosine_sim, -1.0, 1.0)
    angular_dist = float(np.degrees(np.arccos(cosine_sim_clamped)))

    return {
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist,
        "angular_distance_degrees": angular_dist,
    }


def refine_search(
    embedder: IssueEmbeddings,
    prompts: list[str],
    iterations: int = 3,
    top_k_for_centroid: int = 10,
    results_k: int = 20,
    anchor_weight: float = 0.3,
    show_drift: bool = True,
) -> dict:
    """
    Iteratively refine search using centroid of top results.

    Args:
        embedder: The embeddings manager
        prompts: Initial search prompts
        iterations: Number of refinement iterations
        top_k_for_centroid: How many top results to use for centroid
        results_k: How many results to return
        anchor_weight: Weight given to original prompt centroid (0-1)
                       Higher = more stable, lower = more drift
        show_drift: Whether to track drift metrics

    Returns:
        Dict with final results, all iterations, and drift history
    """
    # Compute initial centroid from prompts
    prompt_embeddings = embedder.embed_texts(prompts)
    current_centroid = np.mean(prompt_embeddings, axis=0)
    current_centroid = current_centroid / np.linalg.norm(current_centroid)
    original_centroid = current_centroid.copy()

    history = []
    prev_results = None

    # Get initial results for comparison
    initial_results = embedder.search_by_vector(current_centroid, k=results_k * 2)
    initial_stats = compute_cluster_stats(initial_results, top_k=top_k_for_centroid)

    for i in range(iterations):
        # Search with current centroid
        results = embedder.search_by_vector(current_centroid, k=results_k * 2)

        # Compute cluster quality metrics
        cluster_stats = compute_cluster_stats(results, top_k=top_k_for_centroid)

        # Compute result overlap with previous iteration
        if prev_results is not None:
            overlap = compute_result_overlap(prev_results, results, top_k=top_k_for_centroid)
        else:
            overlap = compute_result_overlap(initial_results, results, top_k=top_k_for_centroid)

        # Compute new centroid from results
        new_centroid = compute_centroid(
            embedder, results, top_k=top_k_for_centroid, weighted=True
        )

        # Mix with original centroid to prevent drift
        blended_centroid = (
            anchor_weight * original_centroid +
            (1 - anchor_weight) * new_centroid
        )
        blended_centroid = blended_centroid / np.linalg.norm(blended_centroid)

        # Track drift
        drift_from_prev = compute_drift(current_centroid, blended_centroid)
        drift_from_original = compute_drift(original_centroid, blended_centroid)

        iteration_info = {
            "iteration": i + 1,
            "top_results": [(r[0]["number"], r[0]["repo"].split("/")[-1], r[1]) for r in results[:5]],
            "drift_from_previous": drift_from_prev,
            "drift_from_original": drift_from_original,
            "cluster_stats": cluster_stats,
            "result_overlap": overlap,
        }
        history.append(iteration_info)

        prev_results = results
        current_centroid = blended_centroid

    # Final search with refined centroid
    final_results = embedder.search_by_vector(current_centroid, k=results_k)
    final_stats = compute_cluster_stats(final_results, top_k=top_k_for_centroid)

    return {
        "results": final_results,
        "history": history,
        "initial_stats": initial_stats,
        "final_stats": final_stats,
        "final_drift_from_original": compute_drift(original_centroid, current_centroid),
    }


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
