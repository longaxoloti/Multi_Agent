import asyncio
import logging
import subprocess
import os
import random
import math
from typing import Any, Optional
from urllib.parse import quote_plus
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from main.config import CHROME_CDP_PORT

logger = logging.getLogger(__name__)

class ChromeCDPClient:
    def __init__(self, port: int = None):
        if port is None:
            port = CHROME_CDP_PORT
        self.cdp_url = f"http://127.0.0.1:{port}"
        self.port = port
        self.pw = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self._lock = asyncio.Lock()
        self._pages: dict[str, Page] = {}
        self._refs: dict[str, dict[str, str]] = {}
        self._initialized = False

    async def _try_launch_chrome(self) -> bool:
        logger.info(f"Open Chrome with port {self.port}...")
        
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        user_data_dir = "/tmp/chrome_debug"
        
        os.makedirs(user_data_dir, exist_ok=True)
        
        cmd = [
            chrome_path,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check"
        ]
        
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for i in range(10):
                await asyncio.sleep(1)
                if await self._check_port_listening():
                    return True
            return False
        except Exception as e:
            logger.error(f"Error when open Chrome: {e}")
            return False

    async def _check_port_listening(self) -> bool:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            
            try:
                self.pw = await async_playwright().start()
                
                if not await self._check_port_listening():
                    success = await self._try_launch_chrome()
                    if not success:
                        logger.warning("Cannot open Chrome automatively, try to connect directly...")

                try:
                    self.browser = await self.pw.chromium.connect_over_cdp(self.cdp_url)
                except Exception as e:
                    logger.error(f"Error when connect CDP: {e}")
                    raise e
                
                self.context = self.browser.contexts[0] if self.browser.contexts else None
                if not self.context:
                    raise RuntimeError("No browser context found via CDP.")
                self._initialized = True
                logger.info(f"Chrome CDP initialized on {self.cdp_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Chrome via CDP: {e}")

    async def close(self) -> None:
        async with self._lock:
            if self.browser:
                try:
                    await self.browser.close()
                except Exception:
                    pass
            if self.pw:
                try:
                    await self.pw.stop()
                except Exception:
                    pass
            self._initialized = False
            self._pages.clear()

    async def ping(self) -> bool:
        if not self._initialized:
            await self.initialize()
        return self._initialized and self.browser.is_connected()

    async def create_tab(self, url: str = "about:blank") -> Optional[str]:
        if not self._initialized:
            await self.initialize()
        if not self.browser or not self.browser.is_connected():
            return None
            
        try:
            page = await self.context.new_page()
            if url and url != "about:blank":
                await page.goto(url, wait_until="domcontentloaded")
            tab_id = str(id(page))
            self._pages[tab_id] = page
            self._refs[tab_id] = {}
            return tab_id
        except Exception as e:
            logger.error(f"CDP create_tab failed: {e}")
            return None

    async def list_tabs(self) -> list[str]:
        return list(self._pages.keys())

    async def close_all_tabs(self) -> int:
        count = 0
        for tab_id in list(self._pages.keys()):
            if await self.close_tab(tab_id):
                count += 1
        return count

    async def close_tab(self, tab_id: str) -> bool:
        page = self._pages.get(tab_id)
        if page:
            try:
                await page.close()
            except Exception:
                pass
            self._pages.pop(tab_id, None)
            self._refs.pop(tab_id, None)
            return True
        return False

    async def navigate(self, tab_id: str, url: str) -> bool:
        page = self._pages.get(tab_id)
        if not page:
            return False
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await self._inject_virtual_cursor(page)
            # Smooth scroll down a bit after loading to simulate human reading
            await self._smooth_scroll(page)
            return True
        except Exception as e:
            logger.error(f"CDP navigate failed: {e}")
            return False

    async def _inject_virtual_cursor(self, page: Page):
        """Injects a highly visible red arrow cursor into the page."""
        script = """
            if (!document.getElementById('playwright-custom-cursor')) {
                const cursor = document.createElement('div');
                cursor.id = 'playwright-custom-cursor';
                cursor.style.position = 'fixed';
                cursor.style.width = '24px';
                cursor.style.height = '24px';
                cursor.style.pointerEvents = 'none';
                cursor.style.zIndex = '2147483647';
                cursor.style.transition = 'top 0.05s linear, left 0.05s linear';
                
                // Red human-like arrow
                cursor.style.background = 'url("data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'24\\' height=\\'24\\' viewBox=\\'0 0 24 24\\' fill=\\'red\\' stroke=\\'white\\' stroke-width=\\'1\\'%3E%3Cpath d=\\'M3,3 L10.07,19.97 L13.58,13.58 L19.97,10.07 Z\\'/%3E%3C/svg%3E") no-repeat';
                cursor.style.backgroundSize = 'contain';
                
                document.body.appendChild(cursor);

                window.addEventListener('mousemove', (e) => {
                    const c = document.getElementById('playwright-custom-cursor');
                    if (c) {
                        c.style.left = e.clientX + 'px';
                        c.style.top = e.clientY + 'px';
                    }
                });
            }
        """
        try:
            await page.evaluate(script)
        except Exception:
            pass

    async def _smooth_scroll(self, page: Page):
        """Simulates human scrolling down."""
        try:
            scroll_steps = random.randint(2, 5)
            for _ in range(scroll_steps):
                y_delta = random.randint(100, 400)
                await page.mouse.wheel(0, y_delta)
                await asyncio.sleep(random.uniform(0.1, 0.4))
        except Exception:
            pass

    async def _move_mouse_human_like(self, page: Page, x: float, y: float):
        """Moves the mouse to a target x, y using a randomized curve trajectory."""
        try:
            # We must dispatch the events so the page knows where the mouse is and updates the custom cursor SVG.
            # Start from a random near-midpoint if previous position is unknown.
            # Playwright mouse position cannot be read reliably so we just start moving it.
            # Assuming current position is 100, 100 as a naive fallback, but it's better to just move linearly for the demo 
            # or jump to a start position.
            
            # Start point: random spot around the center of the screen
            start_x = random.randint(200, 800)
            start_y = random.randint(200, 600)
            await page.mouse.move(start_x, start_y)
            await asyncio.sleep(0.1)

            # Control point for bezier curve
            cp_x = (start_x + x) / 2 + random.randint(-200, 200)
            cp_y = (start_y + y) / 2 + random.randint(-200, 200)

            steps = random.randint(15, 30)
            for i in range(1, steps + 1):
                t = i / steps
                # Ease-in-out curve for t
                curve_t = t * t * (3 - 2 * t)
                
                # Quadratic bezier curve
                curr_x = (1 - curve_t) ** 2 * start_x + 2 * (1 - curve_t) * curve_t * cp_x + curve_t ** 2 * x
                curr_y = (1 - curve_t) ** 2 * start_y + 2 * (1 - curve_t) * curve_t * cp_y + curve_t ** 2 * y

                await page.mouse.move(curr_x, curr_y)
                await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # Ensure we end exactly on the target
            await page.mouse.move(x, y)
            await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"CDP mouse move failed: {e}")

    async def search_google(self, tab_id: str, query: str) -> bool:
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        return await self.navigate(tab_id, search_url)

    async def get_snapshot_page(self, tab_id: str, offset: Optional[int] = None) -> Optional[dict]:
        page = self._pages.get(tab_id)
        if not page:
            return None
        
        try:
            await page.wait_for_timeout(1000)
        except Exception:
            pass

        try:
            extract_script = """
            () => {
                let text = document.body.innerText || "";
                let links = Array.from(document.querySelectorAll('a[href]'));
                let refs = [];
                let urlMap = {};
                let count = 0;
                
                let ignoreDomains = ['google.com/support', 'accounts.google.com', 'support.google.com', 'maps.google.com', 'policies.google.com', 'youtube.com'];
                
                links.forEach((a) => {
                    let href = a.href;
                    if (!href.startsWith('http')) return;
                    if (ignoreDomains.some(domain => href.includes(domain))) return;
                    
                    let actualUrl = href;
                    if (href.includes('google.com/url?url=') || href.includes('google.com/url?q=')) {
                        try {
                            let urlObj = new URL(href);
                            let q = urlObj.searchParams.get('q') || urlObj.searchParams.get('url');
                            if (q) actualUrl = q;
                        } catch(e) {}
                    }
                    
                    if (actualUrl.includes('google.com/search') || actualUrl.includes('google.com/preferences')) return;
                    
                    let title = (a.innerText || "").trim().replace(/\\n/g, ' ').substring(0, 150);
                    // Filter out very short texts or just navigation elements
                    if(title.length > 10) {
                        let refId = 'e' + count;
                        refs.push(`[link ${refId}] ${title}`);
                        urlMap[refId] = actualUrl;
                        count++;
                    }
                });
                
                let snapshotText = text.substring(0, 6000) + "\\n\\n--- Source Links ---\\n" + refs.join('\\n');
                return {
                    text: snapshotText,
                    urlMap: urlMap,
                    url: document.location.href,
                    refsCount: count
                };
            }
            """
            result = await page.evaluate(extract_script)
            self._refs[tab_id] = result["urlMap"]
            
            return {
                "url": result["url"],
                "snapshot": result["text"],
                "refsCount": result["refsCount"],
                "truncated": False,
                "hasMore": False,
                "nextOffset": None
            }
        except Exception as e:
            logger.error(f"CDP get_snapshot_page failed: {e}")
            return None

    async def click(self, tab_id: str, ref: str) -> bool:
        url = self._refs.get(tab_id, {}).get(ref)
        if not url:
            return False
            
        page = self._pages.get(tab_id)
        if page:
            try:
                # Find all links that match the URL
                elements = await page.query_selector_all(f'a[href="{url}"]')
                if elements:
                    # Pick the first visible one
                    for element in elements:
                        if await element.is_visible():
                            box = await element.bounding_box()
                            if box:
                                target_x = box['x'] + box['width'] / 2
                                target_y = box['y'] + box['height'] / 2
                                await self._inject_virtual_cursor(page)
                                await self._move_mouse_human_like(page, target_x, target_y)
                                # Actually click using Playwright's mouse click at the coordinates to ensure the visual cursor clicks it
                                await page.mouse.click(target_x, target_y)
                                await asyncio.sleep(0.5)
                                return True
            except Exception as e:
                logger.error(f"Failed to perform human-like click: {e}")
                
        # Fallback to navigating to the URL if the element wasn't found or couldn't be clicked
        return await self.navigate(tab_id, url)
