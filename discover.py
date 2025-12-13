"""
Discover latent issue clusters using Gaussian Mixture Models.

This implements unsupervised "vibe discovery" - finding natural groupings
in the issue embedding space without predefined concepts.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from embeddings import IssueEmbeddings


def get_all_embeddings(embedder: IssueEmbeddings) -> np.ndarray:
    """Extract all embeddings from the FAISS index."""
    import faiss
    n = embedder.index.ntotal
    d = embedder.dimension
    return faiss.rev_swig_ptr(
        embedder.index.get_xb(), n * d
    ).reshape(n, d).copy()


def fit_gmm(
    embeddings: np.ndarray,
    n_components: int = 10,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> GaussianMixture:
    """
    Fit a Gaussian Mixture Model to embeddings.

    Args:
        embeddings: (n_samples, n_features) array of embeddings
        n_components: Number of mixture components (clusters)
        covariance_type: 'diag' is faster for high dimensions, 'full' is more flexible
        random_state: For reproducibility

    Returns:
        Fitted GaussianMixture model
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
        n_init=3,
    )
    gmm.fit(embeddings)
    return gmm


def discover_vibes(
    embedder: IssueEmbeddings,
    n_vibes: int = 10,
    top_k_per_vibe: int = 5,
    use_pca: bool = False,
    pca_components: int = 50,
) -> dict:
    """
    Discover natural clusters (vibes) in the issue embedding space.

    Args:
        embedder: Loaded IssueEmbeddings with index
        n_vibes: Number of clusters to discover
        top_k_per_vibe: Number of representative issues per cluster
        use_pca: Whether to reduce dimensionality before clustering
        pca_components: Number of PCA components if use_pca=True

    Returns:
        Dict with vibes, their representative issues, and model metrics
    """
    print(f"Extracting embeddings for {len(embedder.issues)} issues...")
    embeddings = get_all_embeddings(embedder)

    # Optional dimensionality reduction
    if use_pca:
        print(f"Reducing to {pca_components} dimensions with PCA...")
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explains {explained_var:.1%} of variance")
    else:
        embeddings_reduced = embeddings
        explained_var = 1.0

    print(f"Fitting GMM with {n_vibes} components...")
    gmm = fit_gmm(embeddings_reduced, n_components=n_vibes)

    # Get cluster assignments and probabilities
    probs = gmm.predict_proba(embeddings_reduced)  # (n_samples, n_components)
    assignments = probs.argmax(axis=1)

    # Build vibe info
    vibes = []
    for vibe_id in range(n_vibes):
        # Issues assigned to this cluster
        mask = assignments == vibe_id
        cluster_size = mask.sum()

        # Get probabilities for this cluster
        cluster_probs = probs[:, vibe_id]

        # Top issues by probability of belonging to this cluster
        top_indices = np.argsort(cluster_probs)[-top_k_per_vibe:][::-1]
        top_issues = [
            {
                "issue": embedder.issues[i],
                "probability": float(cluster_probs[i]),
            }
            for i in top_indices
        ]

        # Cluster centroid (in reduced space if PCA)
        centroid = gmm.means_[vibe_id]

        # Cluster "tightness" - average probability of members
        member_probs = cluster_probs[mask]
        avg_prob = float(member_probs.mean()) if len(member_probs) > 0 else 0

        vibes.append({
            "vibe_id": vibe_id,
            "size": int(cluster_size),
            "avg_membership_prob": avg_prob,
            "top_issues": top_issues,
        })

    # Sort by cluster size
    vibes.sort(key=lambda v: v["size"], reverse=True)

    return {
        "vibes": vibes,
        "n_vibes": n_vibes,
        "total_issues": len(embedder.issues),
        "bic": gmm.bic(embeddings_reduced),
        "aic": gmm.aic(embeddings_reduced),
        "converged": gmm.converged_,
        "pca_variance_explained": explained_var if use_pca else None,
    }


def find_optimal_k(
    embedder: IssueEmbeddings,
    k_range: range = range(5, 30, 5),
    use_pca: bool = True,
    pca_components: int = 50,
) -> list[dict]:
    """
    Find optimal number of clusters using BIC/AIC.

    Lower BIC/AIC indicates better model fit with appropriate complexity.
    """
    print(f"Extracting embeddings...")
    embeddings = get_all_embeddings(embedder)

    if use_pca:
        print(f"Reducing to {pca_components} dimensions with PCA...")
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)

    results = []
    for k in k_range:
        print(f"Fitting k={k}...", end=" ")
        gmm = fit_gmm(embeddings, n_components=k)
        bic = gmm.bic(embeddings)
        aic = gmm.aic(embeddings)
        print(f"BIC={bic:.0f}, AIC={aic:.0f}")
        results.append({"k": k, "bic": bic, "aic": aic, "converged": gmm.converged_})

    return results


def format_vibes(result: dict, max_issues: int = 3) -> str:
    """Format discovered vibes for display."""
    lines = []
    lines.append(f"Discovered {result['n_vibes']} vibes in {result['total_issues']} issues")
    lines.append(f"Model: BIC={result['bic']:.0f}, AIC={result['aic']:.0f}, "
                 f"converged={result['converged']}")
    if result.get('pca_variance_explained'):
        lines.append(f"PCA variance explained: {result['pca_variance_explained']:.1%}")
    lines.append("")

    for vibe in result["vibes"]:
        lines.append("=" * 60)
        lines.append(f"VIBE {vibe['vibe_id']} ({vibe['size']} issues, "
                     f"avg prob: {vibe['avg_membership_prob']:.2f})")
        lines.append("=" * 60)

        for item in vibe["top_issues"][:max_issues]:
            issue = item["issue"]
            prob = item["probability"]
            repo = issue["repo"].split("/")[-1]
            state = "[open]" if issue["state"] == "open" else "[closed]"
            lines.append(f"  [{prob:.2f}] {repo}#{issue['number']} {state}")
            lines.append(f"         {issue['title'][:60]}")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover issue vibes with GMM")
    parser.add_argument("--index-dir", default="data/index", help="Index directory")
    parser.add_argument("-k", "--n-vibes", type=int, default=10, help="Number of vibes to discover")
    parser.add_argument("--top-k", type=int, default=5, help="Issues per vibe")
    parser.add_argument("--pca", action="store_true", help="Use PCA reduction")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimensions")
    parser.add_argument("--find-k", action="store_true", help="Find optimal k using BIC/AIC")
    parser.add_argument("--k-min", type=int, default=5, help="Min k for --find-k")
    parser.add_argument("--k-max", type=int, default=30, help="Max k for --find-k")
    args = parser.parse_args()

    embedder = IssueEmbeddings()
    embedder.load(args.index_dir)

    if args.find_k:
        results = find_optimal_k(
            embedder,
            k_range=range(args.k_min, args.k_max + 1, 5),
            use_pca=args.pca,
            pca_components=args.pca_dim,
        )
        print("\nResults:")
        print(f"{'k':>5} {'BIC':>12} {'AIC':>12} {'Converged':>10}")
        for r in results:
            print(f"{r['k']:>5} {r['bic']:>12.0f} {r['aic']:>12.0f} {str(r['converged']):>10}")
        best = min(results, key=lambda r: r["bic"])
        print(f"\nBest k by BIC: {best['k']}")
    else:
        result = discover_vibes(
            embedder,
            n_vibes=args.n_vibes,
            top_k_per_vibe=args.top_k,
            use_pca=args.pca,
            pca_components=args.pca_dim,
        )
        print("\n" + format_vibes(result))
