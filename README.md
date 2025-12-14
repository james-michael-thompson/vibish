# vibish

Semantic search for GitHub issues using embeddings. Find issues by "vibe" rather than exact keyword matches.

## Features

- **Semantic search**: Find issues similar to a query even without exact keyword matches
- **Concept search**: Define multi-prompt concepts in `concepts.json` and search for related issues
- **Iterative refinement**: Refine searches using centroid of top results (pseudo-relevance feedback)
- **Cluster discovery**: Use GMM to discover natural issue clusters automatically
- **State filtering**: Filter by open/closed issues

## Installation

```bash
# Clone the repo
git clone https://github.com/james-michael-thompson/k8s-issue-embedding
cd k8s-issue-embedding

# Install with uv
uv sync
```

## Quick Start

```bash
# Fetch issues from GitHub
uv run vibish fetch

# Build the embedding index
uv run vibish index build

# Search!
uv run vibish search -q "memory leak"
```

Or do it all at once:

```bash
uv run vibish vibe
```

## Usage

### Basic Search

```bash
# Free-form query
uv run vibish search -q "node not ready"

# Search by predefined concept
uv run vibish search --concept race_conditions

# Filter by state
uv run vibish search -q "consolidation" -s open

# Interactive mode
uv run vibish search -i
```

### Custom Prompts

Search using multiple prompts (results are combined):

```bash
uv run vibish search -p "race condition" -p "deadlock" -p "concurrent access"
```

### Iterative Refinement

Refine search by computing centroid of top results and searching again:

```bash
uv run vibish search -p "race condition" -p "deadlock" --refine 3
```

This uses pseudo-relevance feedback to find a tighter cluster of related issues. Output shows drift metrics and cluster quality improvements.

Options:
- `-r N, --refine N`: Number of refinement iterations
- `--anchor-weight W`: Weight for original prompts (0-1, default 0.3). Higher = more stable, lower = more exploration.

### Cluster Discovery

Discover natural issue clusters using Gaussian Mixture Models:

```bash
# Find 20 clusters
uv run vibish discover -k 20

# Find optimal number of clusters using BIC
uv run vibish discover --find-k
```

### Managing Data

```bash
# Show issue/index statistics
uv run vibish issues summarize
uv run vibish index summarize

# Delete everything (with recovery)
uv run vibish nuke

# List available concepts
uv run vibish concepts
```

## Concepts

Concepts are defined in `concepts.json`. Each concept has a description and multiple search prompts:

```json
{
  "race_conditions": {
    "description": "Data races, deadlocks, timing issues",
    "prompts": [
      "race condition concurrent access data race",
      "timing issue synchronization deadlock"
    ]
  }
}
```

## How It Works

1. **Embeddings**: Issues are embedded using `sentence-transformers` (all-MiniLM-L6-v2, 384 dimensions)
2. **Indexing**: FAISS IndexFlatIP for fast cosine similarity search
3. **Search**: Query text is embedded and compared to all issue embeddings
4. **Refinement**: Computes weighted centroid of top results, blends with original query, repeats
5. **Discovery**: GMM clustering on PCA-reduced embeddings to find natural groupings

## Configuration

Edit `fetch_issues.py` to change which repos are indexed:

```python
REPOS = [
    "kubernetes-sigs/karpenter",
    "aws/karpenter-provider-aws",
]
```

## Requirements

- Python 3.12+
- ~500MB disk for model + index
- CPU only (no GPU required)
