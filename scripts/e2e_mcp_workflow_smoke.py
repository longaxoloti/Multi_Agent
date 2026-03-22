import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.camofox_mcp_client import CamoFoxMCPClient


async def run_smoke() -> int:
    chat_id = "e2e_mcp"
    session_key = "e2e_mcp_session"
    client = CamoFoxMCPClient(user_id=chat_id, session_key=session_key)

    checks: list[tuple[str, bool, str]] = []
    tab_id = None

    try:
        status_ok = await client.ping()
        checks.append(("server_status", status_ok, "ping camofox-mcp + camofox-browser"))
        if not status_ok:
            return _print_checks(checks)

        tab_id = await client.create_tab("https://example.com")
        checks.append(("create_tab", bool(tab_id), f"tab_id={tab_id or 'None'}"))
        if not tab_id:
            return _print_checks(checks)

        navigated = await client.navigate(tab_id, "https://example.com")
        checks.append(("navigate_and_snapshot", navigated, "navigate to https://example.com"))

        snapshot_page = await client.get_snapshot_page(tab_id)
        snapshot_text = (snapshot_page or {}).get("snapshot", "")
        has_example_text = "Example Domain" in snapshot_text
        checks.append(("snapshot_contains_text", has_example_text, "expect 'Example Domain' in snapshot"))

        click_attempt = await client.click(tab_id, "e1")
        checks.append(("click_ref_probe", click_attempt, "best-effort probe with ref e1"))

        close_ok = await client.close_tab(tab_id)
        checks.append(("close_tab", close_ok, "close smoke tab"))

        tab_id = None
    finally:
        if tab_id:
            await client.close_tab(tab_id)
        await client.close()

    return _print_checks(checks)


def _print_checks(checks: list[tuple[str, bool, str]]) -> int:
    print("=" * 90)
    print("MCP CAMOFOX SMOKE")
    print("=" * 90)

    all_required_ok = True
    required_steps = {"server_status", "create_tab", "navigate_and_snapshot", "snapshot_contains_text", "close_tab"}

    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"{marker:4} | {name:24} | {detail}")
        if name in required_steps and not ok:
            all_required_ok = False

    if all_required_ok:
        print("Result: PASS")
        return 0

    print("Result: FAIL")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    raise SystemExit(asyncio.run(run_smoke()))
