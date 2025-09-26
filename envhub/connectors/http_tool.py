import httpx


class HTTPTool:
    """Perform HTTP requests with minimal configuration."""

    def __init__(self, timeout: int = 30):
        self.client = httpx.Client(timeout=timeout)

    def request(self, method: str, url: str, body: str | None = None):
        resp = self.client.request(method=method.upper(), url=url, content=body)
        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        }
