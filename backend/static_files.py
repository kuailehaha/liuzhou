from typing import Any

from fastapi import Response
from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """
    Static file handler that sets Cache-Control headers to avoid stale resources
    during active development.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

