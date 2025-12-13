"""
Generate and manage embeddings for GitHub issues using sentence-transformers.
Uses FAISS for efficient similarity search.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Small, fast model that works well on CPU
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class IssueEmbeddings:
    """Manages embeddings for GitHub issues."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # FAISS index for similarity search
        self.index: faiss.IndexFlatIP | None = None

        # Metadata for each embedded issue (parallel to index)
        self.issues: list[dict] = []

    def _prepare_text(self, issue: dict) -> str:
        """Prepare issue text for embedding."""
        # Combine title and body, with title weighted more heavily
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        labels = issue.get("labels", [])

        # Truncate body to avoid very long texts
        if len(body) > 2000:
            body = body[:2000] + "..."

        # Include labels as they often contain useful categorization
        label_str = " ".join(labels) if labels else ""

        return f"{title}\n{label_str}\n{body}".strip()

    def embed_issues(
        self,
        issues: list[dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of issues.

        Returns the embedding matrix.
        """
        texts = [self._prepare_text(issue) for issue in issues]

        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )

        return embeddings

    def build_index(
        self,
        issues: list[dict],
        batch_size: int = 64,
    ) -> None:
        """Build FAISS index from issues."""
        print(f"Embedding {len(issues)} issues...")

        embeddings = self.embed_issues(issues, batch_size=batch_size)

        # Use inner product index (equivalent to cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

        self.issues = issues

        print(f"Index built with {self.index.ntotal} vectors")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and return the vector."""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return embedding[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text strings and return the vectors."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return embeddings

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[dict, float]]:
        """
        Search for issues similar to a query string.

        Returns list of (issue, score) tuples.
        """
        query_embedding = self.embed_text(query)
        return self.search_by_vector(query_embedding, k)

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple[dict, float]]:
        """
        Search for issues similar to a query vector.

        Returns list of (issue, score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.issues):
                results.append((self.issues[idx], float(score)))

        return results

    def get_issue_embedding(self, issue_id: int) -> np.ndarray | None:
        """Get the embedding vector for a specific issue by its ID."""
        for i, issue in enumerate(self.issues):
            if issue["id"] == issue_id:
                # Reconstruct from index
                return faiss.rev_swig_ptr(
                    self.index.get_xb(), self.index.ntotal * self.dimension
                ).reshape(self.index.ntotal, self.dimension)[i]
        return None

    def get_embeddings_by_indices(self, indices: list[int]) -> np.ndarray:
        """Get embedding vectors for issues by their index positions."""
        all_vectors = faiss.rev_swig_ptr(
            self.index.get_xb(), self.index.ntotal * self.dimension
        ).reshape(self.index.ntotal, self.dimension)
        return all_vectors[indices]

    def save(self, output_dir: str = "data/index") -> None:
        """Save index and metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(output_path / "issues.index"))

        # Save issues metadata
        with open(output_path / "issues.json", "w") as f:
            json.dump(self.issues, f)

        # Save model info
        with open(output_path / "config.json", "w") as f:
            json.dump({"model_name": self.model_name}, f)

        print(f"Saved index to {output_path}")

    def load(self, input_dir: str = "data/index") -> None:
        """Load index and metadata from disk."""
        input_path = Path(input_dir)

        # Load FAISS index
        index_file = input_path / "issues.index"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))

        # Load issues metadata
        with open(input_path / "issues.json") as f:
            self.issues = json.load(f)

        print(f"Loaded index with {len(self.issues)} issues")


def create_index_from_issues(
    issues_dir: str = "data/issues",
    index_dir: str = "data/index",
    model_name: str = DEFAULT_MODEL,
) -> IssueEmbeddings:
    """
    Load issues from disk and create/save an embedding index.
    """
    from fetch_issues import load_issues

    issues = load_issues(issues_dir)
    print(f"Loaded {len(issues)} issues")

    embedder = IssueEmbeddings(model_name=model_name)
    embedder.build_index(issues)
    embedder.save(index_dir)

    return embedder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding index")
    parser.add_argument("--issues-dir", default="data/issues", help="Issues directory")
    parser.add_argument("--index-dir", default="data/index", help="Output index directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    args = parser.parse_args()

    create_index_from_issues(
        issues_dir=args.issues_dir,
        index_dir=args.index_dir,
        model_name=args.model,
    )
