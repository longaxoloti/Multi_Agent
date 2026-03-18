import logging
import asyncio
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

class CamoufoxBrowser:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9377",
        *,
        session_key: str | None = None,
        user_id: str | None = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.session_key = (session_key or "agent_research").strip() or "agent_research"
        self.user_id = (user_id or "agent").strip() or "agent"
        self.max_retries = 3
        self.request_timeout = 45.0

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_on_5xx: bool = True,
    ) -> dict:
        url = f"{self.base_url}{path}"
        timeout_value = timeout if timeout is not None else self.request_timeout
        attempts = max(1, retries if retries is not None else self.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_value) as client:
                    resp = await client.request(method, url, json=json_data, params=params)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < attempts:
                    backoff = 0.8 * attempt
                    logger.warning(
                        "Camoufox %s %s transient error (attempt %s/%s): %s. Retrying in %.1fs...",
                        method,
                        path,
                        attempt,
                        attempts,
                        e,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                # Retry on server-side errors only.
                if retry_on_5xx and status and 500 <= status < 600 and attempt < attempts:
                    backoff = 0.8 * attempt
                    logger.warning(
                        "Camoufox %s %s server error %s (attempt %s/%s). Retrying in %.1fs...",
                        method,
                        path,
                        status,
                        attempt,
                        attempts,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"Unexpected Camoufox request failure for {method} {path}")

    async def _post(self, path: str, json_data: dict) -> dict:
        return await self._request_with_retry("POST", path, json_data=json_data)

    async def _get(self, path: str, params: dict = None) -> dict:
        return await self._request_with_retry("GET", path, params=params)

    async def _delete(self, path: str, params: dict = None) -> dict:
        return await self._request_with_retry("DELETE", path, params=params, timeout=25.0)

    async def ping(self) -> bool:
        """Kiem tra server Camoufox co song khong."""
        try:
            res = await self._get("/health")
            # In Camoufox server, health API returns {"ok": true}
            return res.get("ok") is True
        except Exception as e:
            logger.error(f"Ping Camoufox error: {e}")
            return False

    async def create_tab(self, url: str) -> Optional[str]:
        """Tao tab moi, tra ve tabId"""
        try:
            payload = {
                "url": url,
                "userId": self.user_id,
                "sessionKey": self.session_key
            }
            # Important: do not retry create-tab on 5xx/timeouts. The server-side
            # create can still finish late and retries may spawn duplicate tabs.
            res = await self._request_with_retry(
                "POST",
                "/tabs",
                json_data=payload,
                timeout=55.0,
                retries=1,
                retry_on_5xx=False,
            )
            return res.get("tabId")
        except Exception as e:
            logger.error(f"Error creating tab: {e}")
            return None

    async def list_tabs(self) -> list[str]:
        """Lay danh sach tab id hien tai cua user."""
        try:
            res = await self._get("/tabs", params={"userId": self.user_id})
            tabs = res.get("tabs")
            if not isinstance(tabs, list):
                return []
            tab_ids: list[str] = []
            for item in tabs:
                if not isinstance(item, dict):
                    continue
                tab_id = item.get("tabId") or item.get("targetId")
                if tab_id:
                    tab_ids.append(str(tab_id))
            return tab_ids
        except Exception as e:
            logger.warning(f"Error listing tabs: {e}")
            return []

    async def close_all_tabs(self) -> int:
        """Dong tat ca tab hien co cua user de tranh ton dong session cu."""
        closed = 0
        for tab_id in await self.list_tabs():
            if await self.close_tab(tab_id):
                closed += 1
        return closed

    async def search_google(self, tabId: str, query: str) -> bool:
        """Su dung macro @google_search của Camoufox server"""
        try:
            payload = {
                "userId": self.user_id,
                "macro": "@google_search",
                "query": query
            }
            await self._post(f"/tabs/{tabId}/navigate", payload)
            return True
        except Exception as e:
            logger.error(f"Error searching google: {e}")
            return False

    async def navigate(self, tabId: str, url: str) -> bool:
        """Dieu huong tab toi mot URL moi"""
        try:
            payload = {
                "userId": self.user_id,
                "url": url
            }
            await self._post(f"/tabs/{tabId}/navigate", payload)
            return True
        except Exception as e:
            logger.error(f"Error navigating: {e}")
            return False

    async def click(self, tabId: str, ref: str) -> bool:
        """Click vao element ref (e1, e2, ...) tren snapshot."""
        try:
            payload = {
                "userId": self.user_id,
                "ref": ref,
            }
            await self._post(f"/tabs/{tabId}/click", payload)
            return True
        except Exception as e:
            logger.error(f"Error clicking ref {ref}: {e}")
            return False

    async def get_snapshot(self, tabId: str) -> Optional[str]:
        """Lay HTML quy doi thanh text tu snapshot API cua trinh duyet"""
        try:
            params = {
                "userId": self.user_id,
                "includeScreenshot": "false"
            }
            res = await self._get(f"/tabs/{tabId}/snapshot", params=params)
            return res.get("snapshot")
        except Exception as e:
            logger.error(f"Error getting snapshot: {e}")
            return None

    async def get_snapshot_page(self, tabId: str, offset: Optional[int] = None) -> Optional[dict]:
        """Lay snapshot page voi metadata (url, refsCount, hasMore, nextOffset)."""
        try:
            params = {
                "userId": self.user_id,
                "includeScreenshot": "false"
            }
            if offset is not None:
                params["offset"] = str(offset)

            res = await self._get(f"/tabs/{tabId}/snapshot", params=params)
            return {
                "url": res.get("url", ""),
                "snapshot": res.get("snapshot", ""),
                "refsCount": res.get("refsCount", 0),
                "truncated": bool(res.get("truncated", False)),
                "hasMore": bool(res.get("hasMore", False)),
                "nextOffset": res.get("nextOffset")
            }
        except Exception as e:
            logger.error(f"Error getting snapshot page: {e}")
            return None

    async def close_tab(self, tabId: str) -> bool:
        """Dong tab don dep RAM"""
        try:
            params = {"userId": self.user_id}
            await self._delete(f"/tabs/{tabId}", params=params)
            return True
        except Exception as e:
            logger.error(f"Error closing tab: {e}")
            return False
