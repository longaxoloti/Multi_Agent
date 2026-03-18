#!/usr/bin/env python3
"""
Login Helper — Set up authenticated sessions for the agent.

Launches an interactive login flow for the configured auth backend.

- nodriver backend: opens per-site login pages and saves cookies.
- playwright backend: opens persistent Chromium profile as before.

Usage:
    conda run -n agent-stock python scripts/login_helper.py

Then:
    1. Chrome opens with Google loaded
    2. Navigate to any site (x.com, gemini.google.com, etc.)
    3. Log in manually
    4. Press Enter in the terminal when done
    5. Sessions are saved — the bot will reuse them automatically
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUTH_BROWSER_BACKEND
from tools.playwright_browser import PlaywrightBrowserAutomation
from tools.nodriver_browser import login_interactive


SITES_TO_LOGIN = [
    ("Google / Gemini", "https://accounts.google.com/"),
    ("X / Twitter", "https://x.com/i/flow/login"),
]


async def main():
    print("=" * 60)
    print("🔐 Agent Login Helper")
    print("=" * 60)
    print()
    print("This will open a Chrome window using the agent's persistent")
    print("browser profile. After you log in, the sessions will be")
    print("saved and reused by the bot automatically.")
    print()
    print("Suggested sites to log into:")
    for i, (name, url) in enumerate(SITES_TO_LOGIN, 1):
        print(f"  {i}. {name}: {url}")
    print()

    if AUTH_BROWSER_BACKEND == "nodriver":
        print(f"🔧 Auth backend: {AUTH_BROWSER_BACKEND} (recommended)")
        print("This will open one login window per site and save cookies.")
        print()
        for site_name, url in SITES_TO_LOGIN:
            key = site_name.lower().split("/")[0].strip().replace(" ", "")
            print(f"➡️ Open login for {site_name}: {url}")
            ok = await login_interactive(url=url, site_name=key, wait_seconds=90)
            if ok:
                print(f"✅ Saved cookies for {site_name}")
            else:
                print(f"❌ Login bootstrap failed for {site_name}")
        print("\n✅ nodriver login bootstrap finished.")
        return

    print(f"🔧 Auth backend: {AUTH_BROWSER_BACKEND} (playwright compatibility mode)")
    browser = PlaywrightBrowserAutomation(
        headless=False,
        use_persistent_context=True,
    )

    try:
        await browser.launch()
        print("✅ Chrome launched with the agent's persistent profile.")
        print(f"   Profile dir: {os.environ.get('BROWSER_USER_DATA_DIR', 'data/browser_profile')}")
        print()

        await browser.navigate("https://www.google.com/")

        print("📌 Instructions:")
        print("   1. Use the Chrome window to navigate to sites and log in")
        print("   2. You can open multiple tabs")
        print("   3. Once you're done logging in, come back here")
        print()
        input("   Press ENTER when you're finished logging in → ")

        await browser.save_storage_state()
        print()
        print("✅ Sessions saved! The bot will now have access to your")
        print("   logged-in accounts. You can close this window.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted — sessions may not be fully saved.")
    finally:
        await browser.close()
        print("✅ Chrome closed.")


if __name__ == "__main__":
    asyncio.run(main())
