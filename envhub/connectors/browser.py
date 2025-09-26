class BrowserTool:
    """Stub browser automation connector; integrate with Playwright later."""

    async def open(self, url: str) -> dict:
        return {"ok": True, "url": url, "action": "open"}

    async def click(self, selector: str) -> dict:
        return {"ok": True, "selector": selector, "action": "click"}

    async def extract(self, selector: str) -> dict:
        return {"ok": True, "selector": selector, "action": "extract", "text": ""}
