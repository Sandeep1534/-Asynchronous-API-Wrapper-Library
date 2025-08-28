# -Asynchronous-API-Wrapper-Library
Develop a production-ready Python library that provides an async interface to the  GitHub API using aiohttp. The library will handle rate limiting, pagination, and errors  while exposing a clean, typed interface for developers. The package will be distributable  via PyPI with comprehensive documentation.
"""
async_github.py
A small async GitHub API wrapper using aiohttp.

Features:
- Async GET/POST/PUT/DELETE helpers
- Rate-limit handling (reads X-RateLimit headers and sleeps if exhausted)
- Pagination helper as an async generator for endpoints that use Link headers
- Retries with exponential backoff on 429/5xx
- Simple typed dataclass example for repo items
- Clear exceptions
"""

from __future__ import annotations
import asyncio
import aiohttp
import time
import math
from typing import Optional, AsyncIterator, Dict, Any, List
from dataclasses import dataclass


# ------------------------
# Exceptions
# ------------------------
class GitHubAPIError(Exception):
    """General API error with status and (optional) payload."""
    def __init__(self, status: int, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.payload = payload or {}


# ------------------------
# Typed models (examples)
# ------------------------
@dataclass
class RepoInfo:
    id: int
    name: str
    full_name: str
    private: bool
    html_url: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RepoInfo":
        return RepoInfo(
            id=d.get("id"),
            name=d.get("name"),
            full_name=d.get("full_name"),
            private=bool(d.get("private", False)),
            html_url=d.get("html_url"),
        )


# ------------------------
# GitHub client
# ------------------------
class AsyncGitHub:
    BASE = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        user_agent: str = "async-github-client/1.0",
        session: Optional[aiohttp.ClientSession] = None,
        max_retries: int = 4,
        backoff_factor: float = 0.5,
    ):
        """
        token: optional personal access token (recommended for higher rate limits)
        session: optional aiohttp.ClientSession (will create one if not provided)
        max_retries: number of retries for 5xx or 429 responses
        backoff_factor: base for exponential backoff (seconds)
        """
        self.token = token
        self.user_agent = user_agent
        self._external_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    # -------- Session management --------
    async def _ensure_session(self):
        if self._session is None:
            headers = {"User-Agent": self.user_agent, "Accept": "application/vnd.github.v3+json"}
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            self._session = aiohttp.ClientSession(headers=headers)

    async def close(self):
        if self._session and not self._external_session:
            await self._session.close()
            self._session = None

    # -------- Internal request with retries & rate-limit handling --------
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        await self._ensure_session()
        url = path if path.startswith("http") else f"{self.BASE}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.request(method, url, **kwargs) as resp:
                    # Read headers for rate-limit info
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    reset = resp.headers.get("X-RateLimit-Reset")
                    status = resp.status
                    text = await resp.text()
                    # If rate-limited by GitHub (429 or remaining == 0), wait until reset
                    if status == 429 or (remaining is not None and remaining.isdigit() and int(remaining) == 0):
                        wait_seconds = 0
                        if reset and reset.isdigit():
                            reset_ts = int(reset)
                            wait_seconds = max(0, reset_ts - int(time.time()))
                        # add a small backoff if the reset header isn't available
                        wait_seconds = wait_seconds or (self.backoff_factor * (2 ** attempt))
                        await asyncio.sleep(wait_seconds)
                        continue

                    # Retry on server errors (5xx)
                    if 500 <= status < 600:
                        # exponential backoff
                        if attempt < self.max_retries:
                            backoff = self.backoff_factor * (2 ** attempt)
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise GitHubAPIError(status, f"Server error after retries: {text}")

                    # For client errors (4xx other than 429) return a clear exception
                    if 400 <= status < 600:
                        # Try to parse json message if possible
                        payload = None
                        try:
                            payload = await resp.json()
                        except Exception:
                            payload = {"raw": text}
                        raise GitHubAPIError(status, payload.get("message", text[:200]), payload)

                    # Success path: parse JSON if possible
                    try:
                        return await resp.json()
                    except Exception:
                        # If there's no JSON, return raw text wrapped
                        return {"_raw_text": text}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                # Exponential backoff before retrying
                if attempt < self.max_retries:
                    backoff = self.backoff_factor * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise GitHubAPIError(0, f"Network error after retries: {e}") from e

        # If we exit loop without return, raise last exception
        raise last_exc or GitHubAPIError(0, "Unknown error")

    # -------- Convenience HTTP methods --------
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("POST", path, json=json)

    async def patch(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("PATCH", path, json=json)

    async def delete(self, path: str) -> Dict[str, Any]:
        return await self._request("DELETE", path)

    # -------- Pagination helper (async generator) --------
    async def paginate(self, path: str, params: Optional[Dict[str, Any]] = None, per_page: int = 100) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields items from paginated endpoints.
        It follows the Link header 'next' relation if present.
        """
        await self._ensure_session()
        url = f"{self.BASE}{path}" if not path.startswith("http") else path
        params = params.copy() if params else {}
        params.setdefault("per_page", per_page)
        next_url = url

        while next_url:
            # perform raw request to get full response object (we need headers)
            async with self._session.get(next_url, params=params) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise GitHubAPIError(resp.status, text)

                items = await resp.json()
                if isinstance(items, list):
                    for it in items:
                        yield it
                else:
                    # non-list result (e.g., single object) - yield once then stop
                    yield items
                    return

                # parse Link header for next page
                link = resp.headers.get("Link", "")
                next_url = None
                if link:
                    # Link: <https://api.github.com/...&page=2>; rel="next", <...>; rel="last"
                    parts = [p.strip() for p in link.split(",")]
                    for p in parts:
                        if 'rel="next"' in p:
                            # extract url between < and >
                            start = p.find("<")
                            end = p.find(">", start)
                            if start != -1 and end != -1:
                                next_url = p[start + 1:end]
                                break
                # after first page we should NOT re-send params (they are in next_url)
                params = None

    # -------- Example high-level helpers --------
    async def list_user_repos(self, username: str) -> List[RepoInfo]:
        """
        Collect all public repositories for a user (uses pagination).
        """
        items: List[RepoInfo] = []
        async for obj in self.paginate(f"/users/{username}/repos", params={"type": "owner", "sort": "full_name"}, per_page=100):
            try:
                items.append(RepoInfo.from_dict(obj))
            except Exception:
                # tolerate unexpected shapes
                continue
        return items

    async def get_repo(self, owner: str, repo: str) -> RepoInfo:
        j = await self.get(f"/repos/{owner}/{repo}")
        return RepoInfo.from_dict(j)


# ------------------------
# Minimal usage example
# ------------------------
async def main_example():
    # Optionally set GITHUB_TOKEN env var or pass token here for higher rate limits
    import os
    token = os.getenv("GITHUB_TOKEN", None)

    client = AsyncGitHub(token=token)
    try:
        # list repos (example)
        repos = await client.list_user_repos("octocat")
        print(f"Found {len(repos)} repos for octocat (sample):")
        for r in repos[:10]:
            print("-", r.full_name, r.html_url)

        # get a single repo
        repo = await client.get_repo("octocat", "Hello-World")
        print("Sample repo:", repo.full_name, repo.html_url)
    except GitHubAPIError as e:
        print("GitHub error:", e)
    finally:
        await client.close()


# When run as a script
if __name__ == "__main__":
    asyncio.run(main_example())
