import asyncio

from ddgs import DDGS


class WebSearchClient:
    def __init__(self, search_engine: str = "duckduckgo"):
        self._engine = search_engine

    async def search(self, query: str, max_results: int = 5) -> str:
        if self._engine == "duckduckgo":
            results = await asyncio.to_thread(self._ddgs_search, query, max_results)
            return "\n".join([f"{r['title']}:{r['body']}" for r in results])
        raise ValueError("Only DuckDuckGo is supported")

    def _ddgs_search(self, query: str, max_results: int):
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
