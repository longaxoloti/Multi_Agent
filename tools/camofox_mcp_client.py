import asyncio
import json
import logging
import os
from typing import Any, Optional
from urllib.parse import quote_plus
from urllib.parse import urlparse

from main.config import (
    CAMOFOX_ALLOWED_HOSTS,
    CAMOFOX_API_KEY,
    CAMOFOX_AUTH_REQUIRED,
    CAMOFOX_BLOCK_REMOTE,
    CAMOFOX_MCP_ARGS,
    CAMOFOX_MCP_COMMAND,
    CAMOFOX_MCP_MAX_RETRIES,
    CAMOFOX_REQUIRE_HTTPS_REMOTE,
    CAMOFOX_MCP_RETRY_BACKOFF_SECONDS,
    CAMOFOX_MCP_TIMEOUT_MS,
    CAMOFOX_MCP_TRANSPORT,
    CAMOFOX_MCP_URL,
    CAMOFOX_URL,
)

logger = logging.getLogger(__name__)


def _is_loopback_host(hostname: str) -> bool:
    lowered = (hostname or "").strip().lower()
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        return True
    if lowered.startswith("127."):
        return True
    return False


def _validate_endpoint_security(url: str, *, name: str) -> None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise RuntimeError(f"Invalid {name} URL: {url}")

    host = parsed.hostname.lower()
    is_allowed_host = host in set(CAMOFOX_ALLOWED_HOSTS) or _is_loopback_host(host)

    if CAMOFOX_BLOCK_REMOTE and not is_allowed_host:
        raise RuntimeError(
            f"Security policy blocks remote {name} host '{host}'. "
            "Set CAMOFOX_ALLOWED_HOSTS / CAMOFOX_BLOCK_REMOTE carefully if intentional."
        )

    if CAMOFOX_REQUIRE_HTTPS_REMOTE and not is_allowed_host and parsed.scheme.lower() != "https":
        raise RuntimeError(
            f"Security policy requires HTTPS for remote {name} URL: {url}"
        )


class CamoFoxMCPClient:
    def __init__(
        self,
        *,
        user_id: str,
        session_key: str,
    ):
        self.user_id = (user_id or "agent").strip() or "agent"
        self.session_key = (session_key or "agent_research").strip() or "agent_research"

        self._session = None
        self._session_ctx = None
        self._transport_ctx = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._tool_names: set[str] = set()
        self._snapshot_cache: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            if CAMOFOX_AUTH_REQUIRED and not CAMOFOX_API_KEY:
                raise RuntimeError(
                    "CAMOFOX_AUTH_REQUIRED=true but CAMOFOX_API_KEY is empty. "
                    "Refusing to initialize insecure MCP/browser session."
                )

            _validate_endpoint_security(CAMOFOX_URL, name="CAMOFOX_URL")
            if CAMOFOX_MCP_TRANSPORT == "http":
                _validate_endpoint_security(CAMOFOX_MCP_URL, name="CAMOFOX_MCP_URL")

            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                from mcp.client.streamable_http import streamable_http_client
            except Exception as error:
                raise RuntimeError(
                    "MCP SDK import failed. Install dependency `mcp` first."
                ) from error

            if CAMOFOX_MCP_TRANSPORT == "http":
                self._transport_ctx = streamable_http_client(CAMOFOX_MCP_URL)
                read_stream, write_stream, _ = await self._transport_ctx.__aenter__()
            else:
                process_env = dict(os.environ)
                process_env["CAMOFOX_URL"] = CAMOFOX_URL
                process_env["CAMOFOX_API_KEY"] = CAMOFOX_API_KEY

                server_params = StdioServerParameters(
                    command=CAMOFOX_MCP_COMMAND,
                    args=CAMOFOX_MCP_ARGS,
                    env=process_env,
                )
                self._transport_ctx = stdio_client(server_params)
                read_stream, write_stream = await self._transport_ctx.__aenter__()

            self._session_ctx = ClientSession(read_stream, write_stream)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            await self._refresh_tools()
            self._initialized = True
            logger.info(
                "CamoFox MCP initialized transport=%s tools=%s",
                CAMOFOX_MCP_TRANSPORT,
                len(self._tool_names),
            )

    async def close(self) -> None:
        async with self._lock:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
            if self._transport_ctx is not None:
                await self._transport_ctx.__aexit__(None, None, None)

            self._session = None
            self._session_ctx = None
            self._transport_ctx = None
            self._tool_names = set()
            self._snapshot_cache = {}
            self._initialized = False

    async def _refresh_tools(self) -> None:
        if self._session is None:
            return
        response = await self._session.list_tools()
        self._tool_names = {tool.name for tool in getattr(response, "tools", [])}

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        lowered = (name or "").lower().strip()
        if lowered.startswith("camofox_"):
            lowered = lowered[len("camofox_"):]
        return "".join(ch for ch in lowered if ch.isalnum())

    def _resolve_tool_name(self, requested_name: str) -> Optional[str]:
        if requested_name in self._tool_names:
            return requested_name

        candidates = [
            requested_name,
            f"camofox_{requested_name}",
            requested_name.replace("camofox_", ""),
        ]
        for candidate in candidates:
            if candidate in self._tool_names:
                return candidate

        normalized_request = self._normalize_tool_name(requested_name)
        for existing_name in self._tool_names:
            if self._normalize_tool_name(existing_name) == normalized_request:
                return existing_name

        return None

    @staticmethod
    def _result_to_dict(result: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        structured_content = getattr(result, "structuredContent", None)
        if isinstance(structured_content, dict):
            payload.update(structured_content)

        raw_text_parts: list[str] = []
        for content_block in getattr(result, "content", []) or []:
            text_value = getattr(content_block, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                raw_text_parts.append(text_value)

        if raw_text_parts:
            joined_text = "\n".join(raw_text_parts)
            parsed_json: Any = None
            try:
                parsed_json = json.loads(joined_text)
            except Exception:
                parsed_json = None

            if isinstance(parsed_json, dict):
                payload.update(parsed_json)
                payload.setdefault("text", joined_text)
            elif parsed_json is not None:
                payload.setdefault("result", parsed_json)
                payload.setdefault("text", joined_text)
            else:
                payload.setdefault("result", joined_text)
                payload.setdefault("text", joined_text)

        if not payload:
            payload = {"ok": not bool(getattr(result, "isError", False))}

        if getattr(result, "isError", False):
            payload["ok"] = False
            payload.setdefault("error", payload.get("text") or "MCP tool call returned an error")

        return payload

    async def _call_tool_once(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()

        if self._session is None:
            raise RuntimeError("MCP session is not initialized")

        resolved_tool = self._resolve_tool_name(tool_name)
        if resolved_tool is None:
            await self._refresh_tools()
            resolved_tool = self._resolve_tool_name(tool_name)

        if resolved_tool is None:
            raise RuntimeError(f"MCP tool not found: {tool_name}")

        response = await asyncio.wait_for(
            self._session.call_tool(resolved_tool, arguments=arguments),
            timeout=max(5, CAMOFOX_MCP_TIMEOUT_MS / 1000),
        )
        return self._result_to_dict(response)

    async def _call_tool_with_retry(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, CAMOFOX_MCP_MAX_RETRIES)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return await self._call_tool_once(tool_name, arguments)
            except Exception as error:
                last_error = error
                if isinstance(error, RuntimeError):
                    msg = str(error).lower()
                    if "auth_required" in msg or "security policy" in msg or "refusing to initialize insecure" in msg:
                        break
                if attempt >= attempts:
                    break
                backoff_seconds = CAMOFOX_MCP_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "MCP tool %s failed (attempt %s/%s): %s. Retrying in %.1fs",
                    tool_name,
                    attempt,
                    attempts,
                    error,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)

        raise RuntimeError(f"MCP tool call failed: {tool_name}: {last_error}") from last_error

    async def _call_tool_variants(
        self,
        tool_names: list[str],
        argument_variants: list[dict[str, Any]],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for tool_name in tool_names:
            for args in argument_variants:
                try:
                    return await self._call_tool_with_retry(tool_name, args)
                except Exception as error:
                    last_error = error
                    continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No MCP tool variant could be called")

    @staticmethod
    def _extract_tab_id(payload: dict[str, Any]) -> Optional[str]:
        for key in ("tabId", "tab_id", "targetId", "target_id", "id"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    async def ping(self) -> bool:
        try:
            payload = await self._call_tool_variants(
                ["server_status", "health", "status"],
                [{}, {"user_id": self.user_id}, {"userId": self.user_id}],
            )
            if payload.get("ok") is False:
                return False
            if payload.get("connected") is False:
                return False
            if payload.get("browserConnected") is False:
                return False
            return True
        except Exception as error:
            logger.error("CamoFox MCP ping failed: %s", error)
            return False

    async def create_tab(self, url: str) -> Optional[str]:
        try:
            payload = await self._call_tool_variants(
                ["create_tab", "open_tab"],
                [
                    {"url": url, "userId": self.user_id, "sessionKey": self.session_key},
                    {"url": url, "userId": self.user_id},
                    {"url": url, "user_id": self.user_id, "session_key": self.session_key},
                    {"url": url},
                ],
            )
            return self._extract_tab_id(payload)
        except Exception as error:
            logger.error("CamoFox MCP create_tab failed: %s", error)
            return None

    async def list_tabs(self) -> list[str]:
        try:
            payload = await self._call_tool_variants(
                ["list_tabs", "get_tabs"],
                [{}, {"userId": self.user_id}, {"user_id": self.user_id}],
            )
            tab_ids: list[str] = []
            tabs = payload.get("tabs") or payload.get("results") or []
            if not isinstance(tabs, list):
                return tab_ids
            for item in tabs:
                if isinstance(item, dict):
                    tab_id = self._extract_tab_id(item)
                    if tab_id:
                        tab_ids.append(tab_id)
            return tab_ids
        except Exception as error:
            logger.warning("CamoFox MCP list_tabs failed: %s", error)
            return []

    async def close_all_tabs(self) -> int:
        closed_count = 0
        for tab_id in await self.list_tabs():
            if await self.close_tab(tab_id):
                closed_count += 1
        return closed_count

    async def search_google(self, tab_id: str, query: str) -> bool:
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        try:
            payload = await self._call_tool_variants(
                ["navigate_and_snapshot", "navigate"],
                [
                    {"tabId": tab_id, "url": search_url, "userId": self.user_id},
                    {"tabId": tab_id, "url": search_url, "waitForText": "Google"},
                    {"tab_id": tab_id, "url": search_url},
                    {"tabId": tab_id, "url": search_url},
                ],
            )
            self._snapshot_cache[tab_id] = payload
            return payload.get("ok", True) is not False
        except Exception as error:
            logger.error("CamoFox MCP search_google failed: %s", error)
            return False

    async def navigate(self, tab_id: str, url: str) -> bool:
        try:
            payload = await self._call_tool_variants(
                ["navigate_and_snapshot", "navigate"],
                [
                    {"tabId": tab_id, "url": url, "userId": self.user_id},
                    {"tabId": tab_id, "url": url},
                    {"tab_id": tab_id, "url": url},
                ],
            )
            self._snapshot_cache[tab_id] = payload
            return payload.get("ok", True) is not False
        except Exception as error:
            logger.error("CamoFox MCP navigate failed: %s", error)
            return False

    async def click(self, tab_id: str, ref: str) -> bool:
        try:
            payload = await self._call_tool_variants(
                ["click", "click_element", "click_ref"],
                [
                    {"tabId": tab_id, "ref": ref, "userId": self.user_id},
                    {"tabId": tab_id, "ref": ref},
                    {"tab_id": tab_id, "ref": ref},
                ],
            )
            self._snapshot_cache[tab_id] = payload
            return payload.get("ok", True) is not False
        except Exception as error:
            logger.error("CamoFox MCP click failed: %s", error)
            return False

    @staticmethod
    def _extract_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
        snapshot_value = (
            payload.get("snapshot")
            or payload.get("accessibilitySnapshot")
            or payload.get("text")
            or payload.get("result")
            or ""
        )
        if not isinstance(snapshot_value, str):
            snapshot_value = str(snapshot_value)

        return {
            "url": str(payload.get("url") or ""),
            "snapshot": snapshot_value,
            "refsCount": int(payload.get("refsCount") or payload.get("refs_count") or 0),
            "truncated": bool(payload.get("truncated", False)),
            "hasMore": bool(payload.get("hasMore", False)),
            "nextOffset": payload.get("nextOffset"),
        }

    async def get_snapshot(self, tab_id: str) -> Optional[str]:
        page = await self.get_snapshot_page(tab_id)
        if not page:
            return None
        return page.get("snapshot")

    async def get_snapshot_page(self, tab_id: str, offset: Optional[int] = None) -> Optional[dict]:
        argument_variants = [
            {"tabId": tab_id, "userId": self.user_id},
            {"tabId": tab_id},
            {"tab_id": tab_id},
        ]
        if offset is not None:
            for payload in argument_variants:
                payload["offset"] = offset

        try:
            payload = await self._call_tool_variants(
                ["snapshot", "get_snapshot", "take_snapshot"],
                argument_variants,
            )
            normalized = self._extract_snapshot_payload(payload)
            self._snapshot_cache[tab_id] = normalized
            return normalized
        except Exception as error:
            logger.warning("CamoFox MCP snapshot tool failed for tab=%s: %s", tab_id, error)
            cached = self._snapshot_cache.get(tab_id)
            if isinstance(cached, dict):
                return self._extract_snapshot_payload(cached)
            return None

    async def close_tab(self, tab_id: str) -> bool:
        try:
            payload = await self._call_tool_variants(
                ["close_tab", "close"],
                [
                    {"tabId": tab_id, "userId": self.user_id},
                    {"tabId": tab_id},
                    {"tab_id": tab_id},
                ],
            )
            self._snapshot_cache.pop(tab_id, None)
            return payload.get("ok", True) is not False
        except Exception as error:
            logger.error("CamoFox MCP close_tab failed: %s", error)
            return False
