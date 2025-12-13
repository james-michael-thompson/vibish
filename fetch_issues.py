"""
Fetch GitHub issues from Karpenter and Kubernetes repositories.
Uses the GitHub REST API with pagination support.
"""

import json
import os
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm


REPOS = [
    # "kubernetes/kubernetes",  # ~50K issues, enable when ready for full scale
    "kubernetes-sigs/karpenter",
    "aws/karpenter-provider-aws",
]

GITHUB_API = "https://api.github.com"


def get_github_token() -> str | None:
    """Get GitHub token from environment or gh CLI config."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    # Try to get from gh CLI config
    gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
    if gh_config.exists():
        import yaml
        try:
            with open(gh_config) as f:
                config = yaml.safe_load(f)
                return config.get("github.com", {}).get("oauth_token")
        except Exception:
            pass

    return None


def fetch_issues_page(
    repo: str,
    page: int,
    per_page: int = 100,
    state: str = "all",
    token: str | None = None,
) -> list[dict]:
    """Fetch a single page of issues from a repository."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API}/repos/{repo}/issues"
    params = {
        "state": state,
        "per_page": per_page,
        "page": page,
        "sort": "created",
        "direction": "desc",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 403:
        # Rate limited - check reset time
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        wait_time = max(0, reset_time - time.time()) + 1
        print(f"Rate limited. Waiting {wait_time:.0f}s...")
        time.sleep(wait_time)
        return fetch_issues_page(repo, page, per_page, state, token)

    response.raise_for_status()
    return response.json()


def fetch_all_issues(
    repo: str,
    max_issues: int | None = None,
    token: str | None = None,
) -> Iterator[dict]:
    """
    Fetch all issues from a repository.

    Note: GitHub's issues API includes pull requests. We filter those out.
    """
    page = 1
    per_page = 100
    total_fetched = 0

    with tqdm(desc=f"Fetching {repo}", unit=" issues") as pbar:
        while True:
            issues = fetch_issues_page(repo, page, per_page, token=token)

            if not issues:
                break

            for issue in issues:
                # Skip pull requests (they appear in issues API)
                if "pull_request" in issue:
                    continue

                yield {
                    "id": issue["id"],
                    "number": issue["number"],
                    "repo": repo,
                    "title": issue["title"],
                    "body": issue.get("body") or "",
                    "state": issue["state"],
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "url": issue["html_url"],
                }

                total_fetched += 1
                pbar.update(1)

                if max_issues and total_fetched >= max_issues:
                    return

            page += 1

            # Small delay to be nice to the API
            time.sleep(0.1)


def fetch_and_save_issues(
    output_dir: str = "data/issues",
    max_per_repo: int | None = None,
) -> dict[str, int]:
    """
    Fetch issues from all configured repos and save to JSON files.

    Returns dict mapping repo name to issue count.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = get_github_token()
    if not token:
        print("Warning: No GitHub token found. Rate limits will be restrictive.")
        print("Set GITHUB_TOKEN env var or authenticate with `gh auth login`")

    counts = {}

    for repo in REPOS:
        safe_name = repo.replace("/", "_")
        output_file = output_path / f"{safe_name}.json"

        issues = list(fetch_all_issues(repo, max_issues=max_per_repo, token=token))

        with open(output_file, "w") as f:
            json.dump(issues, f, indent=2)

        counts[repo] = len(issues)
        print(f"Saved {len(issues)} issues from {repo} to {output_file}")

    return counts


def load_issues(data_dir: str = "data/issues") -> list[dict]:
    """Load all saved issues from JSON files."""
    data_path = Path(data_dir)
    all_issues = []

    for json_file in data_path.glob("*.json"):
        with open(json_file) as f:
            issues = json.load(f)
            all_issues.extend(issues)

    return all_issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch GitHub issues")
    parser.add_argument("--max-per-repo", type=int, help="Max issues per repo")
    parser.add_argument("--output-dir", default="data/issues", help="Output directory")
    args = parser.parse_args()

    counts = fetch_and_save_issues(
        output_dir=args.output_dir,
        max_per_repo=args.max_per_repo,
    )

    total = sum(counts.values())
    print(f"\nTotal: {total} issues fetched")
