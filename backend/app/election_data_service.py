import asyncio
import logging
import json
import time
import re
import asyncio
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for environments without zoneinfo
    ZoneInfo = None
from typing import List, Dict, Optional, Any
from pathlib import Path
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import httpx

logger = logging.getLogger(__name__)

class ElectionDataService:
    def __init__(self):
        self.base_url = "https://www.racetothewh.com"
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
        self.historical_data_path = Path(__file__).parent.parent / "data" / "election_historical_data.json"
        self.poll_snapshots_path = Path(__file__).parent.parent / "data" / "election_poll_snapshots.jsonl"
        self._historical_data_cache = None
        self._browser: Optional[Browser] = None
        self._inflight: Dict[str, asyncio.Task] = {}
        self._inflight_lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()
        self._last_snapshot_meta: Dict[str, Dict[str, Any]] = {}
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        
    async def _get_browser(self) -> Browser:
        """Get or create a browser instance."""
        if self._browser is None:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(headless=True)
        return self._browser
    
    async def close(self):
        """Close the browser instance and HTTP client."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        await self.http_client.aclose()

    async def get_polling_data(self, race_type: str, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch polling data for a specific race type.
        Supported race_types: 'approval', 'generic_ballot', 'senate', 'governor', 'house'
        """
        cache_key = f"polls_{race_type}_{state or 'national'}"
        if self._is_cached(cache_key):
            logger.info(f"Returning cached polling data for {cache_key}")
            return self.cache[cache_key]['data']

        async with self._inflight_lock:
            existing = self._inflight.get(cache_key)
            if existing:
                logger.info(f"Joining inflight polling request for {cache_key}")
                task = existing
            else:
                logger.info(f"Election polling request: race_type={race_type}, state={state or 'national'}")
                task = asyncio.create_task(self._fetch_polling_data(race_type, state, cache_key))
                self._inflight[cache_key] = task

        try:
            return await task
        finally:
            if task.done():
                async with self._inflight_lock:
                    if self._inflight.get(cache_key) is task:
                        del self._inflight[cache_key]

    async def _fetch_polling_data(self, race_type: str, state: Optional[str], cache_key: str) -> Dict[str, Any]:
        url = self._get_url_for_race(race_type, state)
        if not url:
            return {"error": "Invalid race type", "polls": []}

        try:
            logger.info(f"Scraping polling data from {url}")
            
            # For approval and generic_ballot, use Playwright to scrape Infogram iframes
            if race_type in ['approval', 'generic_ballot']:
                return await self._scrape_with_playwright(url, race_type, cache_key)
            
            # For Senate/Governor/House, scrape Infogram iframe with table rows
            else:
                browser = await self._get_browser()
                page = await browser.new_page()
                
                try:
                    polls = []
                    metadata = {}
                    best_score = None

                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        try:
                            await page.goto(url, wait_until="networkidle", timeout=45000)
                        except Exception:
                            await page.goto(url, wait_until="domcontentloaded", timeout=45000)

                        try:
                            await page.wait_for_selector('iframe[src*="infogram.com"]', timeout=15000)
                        except Exception:
                            pass

                        # Scroll to load all iframes
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(3000)
                        await page.evaluate("window.scrollTo(0, 0)")
                        await page.wait_for_timeout(3000)

                        # Wait for iframes to load
                        await page.wait_for_timeout(8000)

                        # Find target Infogram iframe (prefer known polling embed if present)
                        frames = page.frames
                        target_frame = None

                        for frame in frames:
                            if 'infogram.com' in frame.url and 'Z4QXXevjUBVE8S5gzMwO' in frame.url:
                                target_frame = frame
                                logger.info(f"Found polling iframe: {frame.url[:80]}...")
                                break

                        # If we didn't find the known embed, scan Infogram frames for a poll table
                        candidate_frames = [f for f in frames if 'infogram.com' in f.url]
                        if not target_frame and not candidate_frames:
                            logger.warning(f"No Infogram iframes found on attempt {attempt}/{max_attempts}")
                            await page.wait_for_timeout(4000)
                            continue

                        polls = []
                        metadata = {}
                        best_score = None

                        frames_to_check = [target_frame] if target_frame else []
                        frames_to_check.extend([f for f in candidate_frames if f not in frames_to_check])

                        for frame in frames_to_check:
                            try:
                                try:
                                    await frame.wait_for_selector('table.igc-table.__dynamic tbody tr', timeout=8000)
                                except Exception:
                                    pass
                                frame_html = await frame.content()
                                parsed = await self._parse_infogram_live_poll_table(frame_html, race_type)
                                if not parsed:
                                    parsed = self._parse_infogram_poll_table(frame_html, race_type)
                                if not parsed:
                                    continue
                                frame_polls, frame_metadata, stats = parsed
                                if not frame_polls:
                                    continue

                                # Prefer tables with non-favorability rows, then matchups
                                score = (stats.get("non_favor_rows", 0) * 1000) + stats.get("matchup_rows", 0)
                                title = (frame_metadata.get("title") or "").lower()
                                if "approval" in title or "favor" in title:
                                    score -= 500

                                if best_score is None or score > best_score:
                                    best_score = score
                                    polls = frame_polls
                                    metadata = frame_metadata
                                    logger.info(
                                        f"Using Infogram iframe with polling table: {frame.url[:80]}... "
                                        f"(rows={stats.get('total_rows')}, matchups={stats.get('matchup_rows')}, favor={stats.get('favor_rows')})"
                                    )
                            except Exception as e:
                                logger.warning(f"Error parsing frame {frame.url[:80]}...: {e}")

                        if polls:
                            break

                        logger.warning(f"No polls parsed on attempt {attempt}/{max_attempts}; retrying...")
                        await page.wait_for_timeout(4000)

                    if not polls:
                        logger.warning("No polls parsed from any Infogram table HTML")
                        return {"error": "No polling data found", "polls": [], "metadata": {}}
                    
                    data = {
                        "race_type": race_type,
                        "polls": polls,
                        "metadata": metadata,
                        "last_updated": time.time()
                    }
                    
                    logger.info(f"Scraped {len(polls)} polls from Infogram table")
                    self._cache_data(cache_key, data)
                    return data
                    
                finally:
                    await page.close()
                
        except Exception as e:
            logger.error(f"Error scraping polling data: {e}", exc_info=True)
            return {"error": str(e), "polls": []}
    
    async def _scrape_with_playwright(self, url: str, race_type: str, cache_key: str) -> Dict[str, Any]:
        """Use Playwright to scrape Infogram iframes for approval/generic_ballot."""
        browser = await self._get_browser()
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(8000)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find Infogram iframes
            iframes = soup.find_all('iframe')
            infogram_iframes = [iframe for iframe in iframes if 'infogram' in iframe.get('src', '')]
            
            if not infogram_iframes:
                logger.warning(f"No Infogram iframes found on {url}")
                return {"race_type": race_type, "polls": [], "error": "No data found"}
            
            # Scrape first iframe
            iframe_url = infogram_iframes[0].get('src')
            if not iframe_url.startswith('http'):
                iframe_url = 'https:' + iframe_url
            
            polls, metadata = await self._scrape_infogram_iframe(iframe_url, race_type)
            
            data = {
                "race_type": race_type,
                "polls": polls,
                "metadata": metadata,
                "last_updated": time.time()
            }
            
            self._cache_data(cache_key, data)
            return data
            
        finally:
            await page.close()
    
    def _parse_html_table_row(self, cells: List[str], headers: List[str], race_type: str) -> Optional[Dict[str, Any]]:
        """Parse a row from the HTML table on senate/26polls page."""
        try:
            # Typical headers: ['#', 'Added', 'Race or Candidate', 'Pollster', 'Lead', 'Dem or Fav', 'GOP or Unfav']
            if len(cells) < 4:
                return None
            
            # Find column indices
            race_idx = next((i for i, h in enumerate(headers) if 'race' in h.lower() or 'candidate' in h.lower()), 2)
            pollster_idx = next((i for i, h in enumerate(headers) if 'pollster' in h.lower()), 3)
            lead_idx = next((i for i, h in enumerate(headers) if 'lead' in h.lower()), 4)
            dem_idx = next((i for i, h in enumerate(headers) if 'dem' in h.lower() or 'fav' in h.lower()), 5)
            gop_idx = next((i for i, h in enumerate(headers) if 'gop' in h.lower() or 'unfav' in h.lower()), 6)
            added_idx = next((i for i, h in enumerate(headers) if 'added' in h.lower()), 1)
            
            race_name = cells[race_idx] if race_idx < len(cells) else ""
            pollster = cells[pollster_idx] if pollster_idx < len(cells) else ""
            lead = cells[lead_idx] if lead_idx < len(cells) else ""
            dem_or_fav = cells[dem_idx] if dem_idx < len(cells) else ""
            gop_or_unfav = cells[gop_idx] if gop_idx < len(cells) else ""
            date_added = cells[added_idx] if added_idx < len(cells) else ""
            
            # Skip empty rows
            if not race_name or not pollster:
                return None
            
            # Determine if this is a candidate race or favorability
            is_favorability = 'fav' in race_name.lower() and '(' in race_name
            
            poll = {
                "race": race_name,
                "pollster": pollster,
                "date_added": date_added,
                "date_range": "",  # Not in HTML table
                "sample": "",  # Not in HTML table
                "grade": "",  # Not in HTML table
                "link": "",
                "margin": lead
            }
            
            if is_favorability:
                poll["results"] = {
                    "Favorable": dem_or_fav,
                    "Unfavorable": gop_or_unfav
                }
            else:
                # Extract candidate names from race string
                # Format: "MI - McMorrow v. Rogers" or "Texas Dem Primary"
                if ' v. ' in race_name or ' vs. ' in race_name or ' vs ' in race_name:
                    # Head-to-head matchup
                    parts = race_name.split(' v. ' if ' v. ' in race_name else (' vs. ' if ' vs. ' in race_name else ' vs '))
                    if len(parts) == 2:
                        # Extract just candidate last names
                        cand1 = parts[0].split(' - ')[-1].strip() if ' - ' in parts[0] else parts[0].strip()
                        cand2 = parts[1].strip()
                        poll["results"] = {
                            cand1: dem_or_fav,
                            cand2: gop_or_unfav
                        }
                    else:
                        poll["results"] = {
                            "Dem": dem_or_fav,
                            "GOP": gop_or_unfav
                        }
                else:
                    # Primary or unclear format
                    poll["results"] = {
                        "Result1": dem_or_fav,
                        "Result2": gop_or_unfav
                    }
            
            return poll
            
        except Exception as e:
            logger.warning(f"Error parsing HTML table row: {e}")
            return None

    def _parse_infogram_poll_table(self, html: str, race_type: str) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]]:
        """Parse an Infogram poll table from HTML for senate/house/governor."""
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.select('table.igc-table.__dynamic')
        if not tables:
            return None

        best_polls: List[Dict[str, Any]] = []
        best_headers: List[str] = []
        best_candidates: List[str] = []
        best_stats: Dict[str, int] = {}
        best_score: Optional[int] = None

        for table in tables:
            header_cells = table.select('thead th')
            headers = [cell.get_text(strip=True) for cell in header_cells]
            header_text = " ".join(headers).lower()
            if 'pollster' in header_text and ('race' in header_text or 'candidate' in header_text):
                header_map = self._build_header_index(headers)
                rows = table.select('tbody tr')

                polls: List[Dict[str, Any]] = []
                candidates_seen: List[str] = []
                favorability_rows_detected = False
                total_rows = 0
                matchup_rows = 0
                favor_rows = 0

                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue

                    cell_texts = [c.get_text(strip=True) for c in cells]

                    date_added = self._safe_cell(cell_texts, header_map.get('added'))
                    row_num = self._safe_cell(cell_texts, header_map.get('row'))
                    race = self._safe_cell(cell_texts, header_map.get('race'))
                    state_cell = self._safe_cell(cell_texts, header_map.get('state'))
                    pollster_raw = self._safe_cell(cell_texts, header_map.get('pollster'))
                    margin = self._safe_cell(cell_texts, header_map.get('lead'))
                    dem_or_fav = self._safe_cell(cell_texts, header_map.get('dem'))
                    gop_or_unfav = self._safe_cell(cell_texts, header_map.get('gop'))

                    # Row-level fallback: if we didn't get both result cells by header (e.g. row has fewer cells),
                    # use the last two cells in the row as the result pair so we always show both columns when present.
                    if len(cell_texts) >= 2 and (not self._has_numeric_value(dem_or_fav) or not self._has_numeric_value(gop_or_unfav)):
                        last1, last2 = cell_texts[-2], cell_texts[-1]
                        if self._has_numeric_value(last1) or self._has_numeric_value(last2):
                            if not self._has_numeric_value(dem_or_fav):
                                dem_or_fav = last1
                            if not self._has_numeric_value(gop_or_unfav):
                                gop_or_unfav = last2

                    if not pollster_raw and not race:
                        continue

                    total_rows += 1
                    pollster, link = self._extract_pollster_and_link(cells, header_map.get('pollster'), pollster_raw)

                    cand_pair = self._extract_candidate_pair(race)
                    is_matchup = cand_pair is not None
                    is_favorability_row = False
                    if not is_matchup:
                        margin_lower = (margin or "").lower()
                        race_lower = (race or "").lower()
                        if "fav" in margin_lower or "unfav" in margin_lower or "favor" in race_lower or "approval" in race_lower:
                            is_favorability_row = True
                            favorability_rows_detected = True

                    has_percentages = self._has_numeric_value(dem_or_fav) and self._has_numeric_value(gop_or_unfav)
                    lead_only_results = None
                    if not is_favorability_row and not has_percentages:
                        if self._has_numeric_value(margin):
                            label = race or "Leader"
                            lead_only_results = {label: margin}
                        else:
                            continue

                    if is_matchup:
                        matchup_rows += 1
                    if is_favorability_row:
                        favor_rows += 1

                    if lead_only_results is not None:
                        results = lead_only_results
                        margin = ""
                    else:
                        dem_idx = header_map.get('dem')
                        gop_idx = header_map.get('gop')
                        dem_header = headers[dem_idx] if dem_idx is not None and dem_idx < len(headers) else ""
                        gop_header = headers[gop_idx] if gop_idx is not None and gop_idx < len(headers) else ""
                        results = self._build_results_from_race(
                            race=race,
                            dem_value=dem_or_fav,
                            gop_value=gop_or_unfav,
                            margin=margin,
                            dem_header=dem_header,
                            gop_header=gop_header,
                            favorability_row=is_favorability_row
                        )

                    for name in results.keys():
                        if name not in candidates_seen:
                            candidates_seen.append(name)

                    state_val = (state_cell or "").strip() if state_cell else None
                    if not state_val and race:
                        from .election_db import _state_from_race_text
                        state_val = _state_from_race_text(race)
                    if state_val and len(state_val) == 2 and state_val.isalpha():
                        state_val = state_val.upper()
                    poll = {
                        "row_num": row_num,
                        "added": date_added,
                        "race": race,
                        "state": state_val or None,
                        "pollster": pollster,
                        "lead": margin,
                        "dem_or_fav": dem_or_fav,
                        "gop_or_unfav": gop_or_unfav,
                        "date_added": date_added,
                        "date_range": date_added or "",
                        "sample": "",
                        "grade": "",
                        "link": link,
                        "margin": margin,
                        "results": results
                    }

                    polls.append(poll)

                if not polls:
                    continue

                stats = {
                    "total_rows": total_rows,
                    "matchup_rows": matchup_rows,
                    "favor_rows": favor_rows,
                    "non_favor_rows": max(total_rows - favor_rows, 0)
                }

                # Prefer tables with the largest total rows, then matchups
                score = (stats["total_rows"] * 1000) + stats["matchup_rows"]
                if best_score is None or score > best_score:
                    best_score = score
                    best_polls = polls
                    best_headers = headers
                    best_candidates = candidates_seen
                    best_stats = stats

        if not best_polls:
            return None

        page_title = soup.title.get_text(strip=True) if soup.title else ""
        metadata = {
            "title": page_title or f"2026 {race_type.title()} Polls",
            "headers": best_headers,
            "candidates": best_candidates,
            "has_favorability": any(
                isinstance(p.get("results"), dict) and
                any(k.lower() in ("favorable", "unfavorable") for k in p["results"].keys())
                for p in best_polls
            )
        }

        return best_polls, metadata, best_stats

    async def _parse_infogram_live_poll_table(
        self,
        html: str,
        race_type: str
    ) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]]:
        """Attempt to fetch live Infogram data (Google Sheets) for the poll table."""
        infographic = self._extract_infographic_payload(html)
        if not infographic:
            return None

        live_cfg = self._select_infogram_live_chart(infographic)
        if not live_cfg:
            return None

        base_url = self._extract_live_data_url(html) or "https://live-data.jifo.co/"
        provider = live_cfg.get("provider")
        key = live_cfg.get("key")
        chart_id = live_cfg.get("chart_id")
        sheet_name = live_cfg.get("sheet_name")

        if not provider or not key:
            return None

        table = await self._fetch_infogram_live_table(
            base_url=base_url,
            provider=provider,
            key=key,
            chart_id=chart_id,
            sheet_name=sheet_name
        )
        if not table:
            return None

        headers, rows = table
        if not headers or not rows:
            return None

        polls, metadata, stats = self._parse_infogram_poll_rows(headers, rows, race_type)
        return polls, metadata, stats

    def _extract_infographic_payload(self, html: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"window\\.infographicData=(\\{.*?\\});</script>", html, re.DOTALL)
        if not match:
            return None
        raw = match.group(1)
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Failed to parse Infogram JSON: {e}")
            return None

    def _extract_live_data_url(self, html: str) -> Optional[str]:
        match = re.search(r"window\\.publicViewConfig\\s*=\\s*(\\{.*?\\});", html, re.DOTALL)
        if not match:
            return None
        try:
            cfg = json.loads(match.group(1))
            return cfg.get("liveDataURL")
        except Exception:
            return None

    def _select_infogram_live_chart(self, infographic: Dict[str, Any]) -> Optional[Dict[str, str]]:
        entities = (
            infographic.get("elements", {})
            .get("content", {})
            .get("content", {})
            .get("entities", {})
        )
        best = None
        best_score = None

        for ent in entities.values():
            if ent.get("type") != "CHART":
                continue
            chart_data = ent.get("props", {}).get("chartData", {})
            chart_type = chart_data.get("chart_type_nr")
            if chart_type not in (19,):  # table
                continue

            live = chart_data.get("custom", {}).get("live") or chart_data.get("live") or {}
            if not isinstance(live, dict) or not live.get("key"):
                continue

            title = (live.get("title") or "").lower()
            score = 0
            if "poll" in title:
                score += 50
            if "latest" in title:
                score += 40
            if "added" in title:
                score += 20
            if "list" in title:
                score += 10
            if live.get("sheetNames") == ["Sheet1"]:
                score += 5

            if best_score is None or score > best_score:
                best_score = score
                sheet_names = live.get("sheetNames") or []
                sheet_selected = live.get("sheetSelected")
                sheet_name = None
                if isinstance(sheet_selected, int) and sheet_names and 0 <= sheet_selected < len(sheet_names):
                    sheet_name = sheet_names[sheet_selected]
                elif sheet_names:
                    sheet_name = sheet_names[0]

                best = {
                    "provider": live.get("provider"),
                    "key": live.get("key"),
                    "chart_id": live.get("chartId"),
                    "sheet_name": sheet_name,
                    "title": live.get("title")
                }

        return best

    async def _fetch_infogram_live_table(
        self,
        base_url: str,
        provider: str,
        key: str,
        chart_id: Optional[str] = None,
        sheet_name: Optional[str] = None
    ) -> Optional[tuple[List[str], List[List[str]]]]:
        base = (base_url or "").strip()
        if not base:
            base = "https://live-data.jifo.co/"
        if not base.endswith("/"):
            base += "/"

        paths = [
            "{provider}/{key}",
            "{provider}/{key}/data",
            "api/{provider}/{key}",
            "api/{provider}/{key}/data",
            "api/1.1/{provider}/{key}",
            "api/1.1/{provider}/{key}/data",
            "v1/{provider}/{key}",
            "v1/{provider}/{key}/data",
        ]

        queries = [""]
        if chart_id:
            queries.append(f"?chartId={chart_id}")
        if sheet_name:
            queries.append(f"?sheet={sheet_name}")
        if chart_id and sheet_name:
            queries.append(f"?chartId={chart_id}&sheet={sheet_name}")

        for path in paths:
            for query in queries:
                url = f"{base}{path.format(provider=provider, key=key)}{query}"
                try:
                    resp = await self.http_client.get(url)
                    if resp.status_code != 200:
                        continue
                    table = self._extract_table_from_live_payload(resp)
                    if table:
                        return table
                except Exception:
                    continue

        return None

    def _extract_table_from_live_payload(self, resp: httpx.Response) -> Optional[tuple[List[str], List[List[str]]]]:
        try:
            payload = resp.json()
        except Exception:
            # Try CSV
            text = resp.text.strip()
            if not text:
                return None
            try:
                import csv
                from io import StringIO

                reader = csv.reader(StringIO(text))
                rows = [r for r in reader if r]
                if len(rows) >= 2:
                    headers = rows[0]
                    return headers, rows[1:]
            except Exception:
                return None
            return None

        table = self._find_table_in_payload(payload)
        if not table:
            return None

        if table and isinstance(table[0], dict):
            headers = list(table[0].keys())
            rows = [[str(row.get(h, "")) for h in headers] for row in table]
            return headers, rows

        if table and isinstance(table[0], list):
            headers = [str(h) for h in table[0]]
            rows = [[str(c) for c in row] for row in table[1:]]
            return headers, rows

        return None

    def _find_table_in_payload(self, payload: Any) -> Optional[List[Any]]:
        def is_table(obj: Any) -> bool:
            if not isinstance(obj, list) or not obj:
                return False
            if all(isinstance(r, list) for r in obj):
                return True
            if all(isinstance(r, dict) for r in obj):
                return True
            return False

        if is_table(payload):
            return payload

        if isinstance(payload, dict):
            for key in ("data", "result", "sheet", "sheets", "table", "tables", "payload"):
                if key in payload:
                    found = self._find_table_in_payload(payload[key])
                    if found:
                        return found
            for val in payload.values():
                found = self._find_table_in_payload(val)
                if found:
                    return found

        if isinstance(payload, list):
            for val in payload:
                found = self._find_table_in_payload(val)
                if found:
                    return found

        return None

    def _parse_infogram_poll_rows(
        self,
        headers: List[str],
        rows: List[List[str]],
        race_type: str
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
        header_map = self._build_header_index(headers)
        polls: List[Dict[str, Any]] = []
        candidates_seen: List[str] = []
        favorability_rows_detected = False
        total_rows = 0
        matchup_rows = 0
        favor_rows = 0

        for row in rows:
            if not row:
                continue
            cell_texts = [str(c).strip() if c is not None else "" for c in row]

            date_added = self._safe_cell(cell_texts, header_map.get('added'))
            row_num = self._safe_cell(cell_texts, header_map.get('row'))
            race = self._safe_cell(cell_texts, header_map.get('race'))
            pollster_raw = self._safe_cell(cell_texts, header_map.get('pollster'))
            margin = self._safe_cell(cell_texts, header_map.get('lead'))
            dem_or_fav = self._safe_cell(cell_texts, header_map.get('dem'))
            gop_or_unfav = self._safe_cell(cell_texts, header_map.get('gop'))

            if len(cell_texts) >= 2 and (not self._has_numeric_value(dem_or_fav) or not self._has_numeric_value(gop_or_unfav)):
                last1, last2 = cell_texts[-2], cell_texts[-1]
                if self._has_numeric_value(last1) or self._has_numeric_value(last2):
                    if not self._has_numeric_value(dem_or_fav):
                        dem_or_fav = last1
                    if not self._has_numeric_value(gop_or_unfav):
                        gop_or_unfav = last2

            if not pollster_raw and not race:
                continue

            total_rows += 1
            pollster = pollster_raw
            link = ""
            if "Link:" in pollster:
                left, right = pollster.split("Link:", 1)
                pollster = left.strip()
                link = right.strip()

            # Clean/Fix Race Name
            # If the race is just a number (e.g. "1") or "District 1", try to make it more descriptive
            # by adding the leader's name if available (e.g. "Finstad (Dist 1)")
            race_clean = (race or "").strip()
            
            # Case 1: Just a number or "District X" (e.g. "1", "District 5")
            if race_clean.isdigit() or (race_clean.startswith("District ") and race_clean[9:].strip().isdigit()):
                dist_num = race_clean if race_clean.isdigit() else race_clean[9:].strip()
                leader_name, _ = self._extract_leader_info(margin)
                if leader_name:
                    race = f"{leader_name} (Dist {dist_num})"
                else:
                    race = f"District {dist_num}"

            cand_pair = self._extract_candidate_pair(race)
            is_matchup = cand_pair is not None
            is_favorability_row = False
            if not is_matchup:
                margin_lower = (margin or "").lower()
                race_lower = (race or "").lower()
                if "fav" in margin_lower or "unfav" in margin_lower or "favor" in race_lower or "approval" in race_lower:
                    is_favorability_row = True
                    favorability_rows_detected = True

            has_percentages = self._has_numeric_value(dem_or_fav) and self._has_numeric_value(gop_or_unfav)
            lead_only_results = None
            if not is_favorability_row and not has_percentages:
                if self._has_numeric_value(margin):
                    label = race or "Leader"
                    lead_only_results = {label: margin}
                else:
                    continue

            if is_matchup:
                matchup_rows += 1
            if is_favorability_row:
                favor_rows += 1

            results = None
            candidate_values = self._extract_candidate_values(headers, cell_texts, header_map)
            if candidate_values and len(candidate_values) >= 2:
                candidate_values.sort(key=lambda x: x[2], reverse=True)
                top_two = candidate_values[:2]
                results = {top_two[0][0]: top_two[0][1], top_two[1][0]: top_two[1][1]}
                dem_or_fav = top_two[0][1]
                gop_or_unfav = top_two[1][1]

            if results is None:
                if lead_only_results is not None:
                    results = lead_only_results
                    margin = ""
                else:
                    dem_idx = header_map.get('dem')
                    gop_idx = header_map.get('gop')
                    dem_header = headers[dem_idx] if dem_idx is not None and dem_idx < len(headers) else ""
                    gop_header = headers[gop_idx] if gop_idx is not None and gop_idx < len(headers) else ""
                    results = self._build_results_from_race(
                        race=race,
                        dem_value=dem_or_fav,
                        gop_value=gop_or_unfav,
                        margin=margin,
                        dem_header=dem_header,
                        gop_header=gop_header,
                        favorability_row=is_favorability_row
                    )

            for name in results.keys():
                if name not in candidates_seen:
                    candidates_seen.append(name)

            state_val = None
            if race:
                from .election_db import _state_from_race_text
                state_val = _state_from_race_text(race)
            if state_val and len(state_val) == 2 and state_val.isalpha():
                state_val = state_val.upper()
            poll = {
                "row_num": row_num,
                "added": date_added,
                "race": race,
                "state": state_val,
                "pollster": pollster,
                "lead": margin,
                "dem_or_fav": dem_or_fav,
                "gop_or_unfav": gop_or_unfav,
                "date_added": date_added,
                "date_range": date_added or "",
                "sample": "",
                "grade": "",
                "link": link,
                "margin": margin,
                "results": results
            }

            polls.append(poll)

        stats = {
            "total_rows": total_rows,
            "matchup_rows": matchup_rows,
            "favor_rows": favor_rows,
            "non_favor_rows": max(total_rows - favor_rows, 0)
        }

        metadata = {
            "title": f"2026 {race_type.title()} Polls",
            "headers": headers,
            "candidates": candidates_seen,
            "has_favorability": favorability_rows_detected
        }

        return polls, metadata, stats

    def _extract_candidate_values(
        self,
        headers: List[str],
        cell_texts: List[str],
        header_map: Dict[str, Optional[int]]
    ) -> List[tuple[str, str, float]]:
        skip_tokens = [
            "added", "poll", "sample", "grade", "margin", "leader", "lead", "favor",
            "unfav", "race", "candidate", "pollster", "date", "#", "dem", "gop", "rep"
        ]
        candidate_values = []
        for idx, header in enumerate(headers):
            h = (header or "").strip()
            hl = h.lower()
            if not h:
                continue
            if any(tok in hl for tok in skip_tokens):
                continue
            if idx == header_map.get("dem") or idx == header_map.get("gop"):
                continue
            if idx >= len(cell_texts):
                continue
            value = cell_texts[idx]
            num = self._parse_numeric_value(value)
            if num is None:
                continue
            candidate_values.append((h, value, num))
        return candidate_values

    def _parse_numeric_value(self, value: str) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"(-?\\d+(?:\\.\\d+)?)", value)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    def _build_header_index(self, headers: List[str]) -> Dict[str, Optional[int]]:
        def find_index(tokens: List[str]) -> Optional[int]:
            for i, h in enumerate(headers):
                hl = h.lower()
                if any(t in hl for t in tokens):
                    return i
            return None

        def find_race_index() -> Optional[int]:
            """
            Find the Race/Candidate column index.
            Strategy:
            1. Try to find explicit 'Race' or 'Candidate' header.
            2. If not found, check column before Pollster (common in Infogram), but VALIDATE it
               to ensure it's not 'District', 'State', or 'Added'.
            """
            # 1. Look for explicit header first (most reliable)
            for i, h in enumerate(headers):
                hl = (h or "").strip().lower()
                if not hl: continue
                # Skip numeric/district columns
                if hl in ("#", "no.", "num") or "district" in hl or "candidate 1" in hl or "candidate 2" in hl:
                    continue
                # Found explicit match
                if "race or candidate" in hl or ("race" in hl and "district" not in hl) or "candidate" in hl:
                    return i

            # 2. Fallback: Check column before Pollster, but be careful
            pollster_idx = find_index(["pollster", "poll"])
            if pollster_idx is not None and pollster_idx > 0:
                prev_idx = pollster_idx - 1
                prev_header = headers[prev_idx].lower().strip()
                # If the column before pollster is "District", "State", "Added", or "#", it's NOT the race.
                # In that case, we might need to look further back or just fail to find it here.
                invalid_prev = ["district", "state", "added", "date", "#", "no.", "num"]
                if not any(inv in prev_header for inv in invalid_prev):
                    return prev_idx
            
            return None

        n = len(headers)
        lead_idx = find_index(["lead"])
        dem_idx = find_index(["dem", "fav", "candidate 1"])
        gop_idx = find_index(["gop", "unfav", "rep", "candidate 2"])
        first_after_lead = (lead_idx + 1) if (lead_idx is not None and lead_idx + 1 < n) else None
        second_after_lead = (lead_idx + 2) if (lead_idx is not None and lead_idx + 2 < n) else None
        last_two_first = (n - 2) if n >= 6 else None
        last_two_second = (n - 1) if n >= 6 else None
        if dem_idx is not None and dem_idx == gop_idx:
            gop_idx = None
        if dem_idx is None:
            dem_idx = first_after_lead or last_two_first
        if gop_idx is None:
            gop_idx = second_after_lead or last_two_second
        if dem_idx is not None and gop_idx is not None and dem_idx == gop_idx:
            gop_idx = second_after_lead or last_two_second

        idx = {
            "row": find_index(["#"]),
            "added": find_index(["added"]),
            "race": find_race_index(),
            "state": find_index(["state"]),
            "pollster": find_index(["pollster", "poll"]),
            "lead": lead_idx,
            "dem": dem_idx,
            "gop": gop_idx,
        }
        return idx

    def _safe_cell(self, cells: List[str], idx: Optional[int]) -> str:
        if idx is None:
            return ""
        if idx < 0 or idx >= len(cells):
            return ""
        return cells[idx]

    def _has_numeric_value(self, value: str) -> bool:
        if not value:
            return False
        return bool(re.search(r"\d", value))

    def _extract_pollster_and_link(self, cells: List[Any], idx: Optional[int], fallback_text: str) -> tuple[str, str]:
        pollster = fallback_text or ""
        link = ""

        if idx is not None and idx < len(cells):
            cell = cells[idx]
            link_el = cell.find('a')
            if link_el:
                link = link_el.get('data-href') or link_el.get('href') or ""

        if "Link:" in pollster:
            left, right = pollster.split("Link:", 1)
            pollster = left.strip()
            if not link:
                link = right.strip()

        return pollster, link

    def _build_results_from_race(
        self,
        race: str,
        dem_value: str,
        gop_value: str,
        margin: str,
        dem_header: str,
        gop_header: str,
        favorability_row: bool
    ) -> Dict[str, str]:
        cand_pair = self._extract_candidate_pair(race)
        leader_name, leader_party = self._extract_leader_info(margin)

        if cand_pair:
            cand1, cand2 = cand_pair
            return {cand1: dem_value, cand2: gop_value}

        # If headers look like candidate names, use them directly.
        if not favorability_row:
            dem_header = (dem_header or "").strip()
            gop_header = (gop_header or "").strip()
            if dem_header and gop_header:
                if not self._is_generic_result_header(dem_header) and not self._is_generic_result_header(gop_header):
                    return {dem_header: dem_value, gop_header: gop_value}

        # No candidate names found; use header labels
        if favorability_row:
            dem_label = self._label_from_header(dem_header, "Favorable")
            gop_label = self._label_from_header(gop_header, "Unfavorable")
        else:
            race_lower = (race or "").lower()
            if "generic" in race_lower and "ballot" in race_lower:
                dem_label = "Dem"
                gop_label = "GOP"
            elif "gop" in race_lower or "republican" in race_lower:
                dem_label = "GOP"
                gop_label = "Opponent"
            elif "dem" in race_lower or "democrat" in race_lower:
                dem_label = "Dem"
                gop_label = "Opponent"
            else:
                dem_label = "Candidate 1"
                gop_label = "Candidate 2"

        # If margin indicates a leader with party, label that side
        if leader_name and leader_party:
            if leader_party == "D":
                dem_label = f"{leader_name} (D)"
                if gop_label == "GOP":
                    gop_label = "Opponent (R)"
            elif leader_party == "R":
                gop_label = f"{leader_name} (R)"
                if dem_label == "Dem":
                    dem_label = "Opponent (D)"

        return {
            dem_label or "Candidate 1": dem_value,
            gop_label or "Candidate 2": gop_value
        }

    def _extract_candidate_pair(self, race: str) -> Optional[tuple[str, str]]:
        if not race:
            return None
        race_clean = race.strip()

        # Remove leading state/district prefixes if present
        if " - " in race_clean:
            race_clean = race_clean.split(" - ", 1)[1].strip()
        if ": " in race_clean and len(race_clean.split(": ", 1)[0]) <= 6:
            race_clean = race_clean.split(": ", 1)[1].strip()

        separators = [" v. ", " vs. ", " vs ", " v "]
        for sep in separators:
            if sep in race_clean:
                left, right = race_clean.split(sep, 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return left, right
        return None

    def _extract_leader_info(self, margin: str) -> tuple[Optional[str], Optional[str]]:
        if not margin:
            return None, None
        margin_clean = margin.strip()
        if "tie" in margin_clean.lower():
            return None, None

        match = re.search(r"([A-Za-z0-9.\-' ]+)\s*\((D|R|I)\)", margin_clean)
        if match:
            name = match.group(1).strip()
            party = match.group(2).strip()
            return name, party
        return None, None

    def _label_from_header(self, header: str, fallback: str) -> str:
        h = (header or "").lower()
        if "favor" in h:
            return "Favorable"
        if "unfav" in h:
            return "Unfavorable"
        if "dem" in h:
            return "Dem"
        if "gop" in h or "rep" in h:
            return "GOP"
        return fallback

    def _is_generic_result_header(self, header: str) -> bool:
        if not header:
            return True
        h = header.lower()
        return any(
            token in h
            for token in ("dem", "gop", "rep", "favor", "unfav", "candidate", "margin", "lead", "poll")
        )
    
    async def _scrape_infogram_iframe(self, iframe_url: str, race_type: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Scrape poll data from an Infogram iframe. Returns (polls, metadata)."""
        browser = await self._get_browser()
        iframe_page = await browser.new_page()
        
        try:
            await iframe_page.goto(iframe_url, wait_until="domcontentloaded", timeout=30000)
            await iframe_page.wait_for_timeout(15000)  # Wait for Infogram JS to populate tables
            
            # Extract page title for state/race context
            page_title = await iframe_page.title()
            
            iframe_content = await iframe_page.content()
            iframe_soup = BeautifulSoup(iframe_content, 'html.parser')
            
            # Find all tables
            tables = iframe_soup.find_all('table')
            
            if not tables:
                logger.warning(f"No tables found in Infogram iframe (waited 15s)")
                return [], {}
            
            # The last table usually has the poll data
            last_table = tables[-1]
            rows = last_table.find_all('tr')
            
            if len(rows) < 2:
                logger.warning(f"Table has insufficient rows: {len(rows)}")
                return [], {}
            
            # Parse header
            header_row = rows[0]
            header_cells = header_row.find_all(['td', 'th'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            
            # Extract metadata
            metadata = {
                "title": page_title,
                "headers": headers,
                "candidates": [],
                "has_favorability": False
            }
            
            # Detect if this is favorability data (useless for candidate races)
            metadata["has_favorability"] = any(
                'favor' in h.lower() or 'unfav' in h.lower() 
                for h in headers
            )
            
            # Extract candidate names from headers
            # Skip: "Added", "Poll", "Sample", "Grade", "Margin", "Favor.", "Unfav.", "Leader:"
            skip_headers = ['added', 'poll', 'sample', 'grade', 'margin', 'favor', 'unfav', 'leader', 'date', '']
            for h in headers[2:]:  # Skip first 2 columns (usually Added and Poll)
                h_lower = h.lower().strip(':. ')
                if h and not any(skip in h_lower for skip in skip_headers):
                    metadata["candidates"].append(h)
            
            logger.info(f"Metadata extracted: title={page_title}, has_favorability={metadata['has_favorability']}, candidates={metadata['candidates']}")
            
            # Parse data rows
            polls = []
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) < 4:  # Need at least date, poll, and results
                    continue
                
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                
                # Skip rows with no meaningful data
                if not cell_texts[1] or cell_texts[1] == '-':
                    continue
                
                # Parse poll details
                poll = self._parse_poll_row(cell_texts, race_type, headers)
                if poll:
                    polls.append(poll)
            
            logger.info(f"Scraped {len(polls)} polls from iframe")
            logger.info(f"Metadata: title={metadata.get('title')}, has_favorability={metadata.get('has_favorability')}, candidates={metadata.get('candidates')}")
            return polls, metadata
            
        except Exception as e:
            logger.error(f"Error scraping Infogram iframe: {e}", exc_info=True)
            return [], {}
        finally:
            await iframe_page.close()
    
    def _parse_poll_row(self, cells: List[str], race_type: str, headers: List[str] = None) -> Optional[Dict[str, Any]]:
        """Parse a single poll row into structured data."""
        try:
            # cells[0] = Added date (may be empty if same as previous)
            # cells[1] = Poll details: "Feb 2 - 8: Morning Consult (C-), 45000 LV"
            # cells[2+] = Results (varies by race type)
            
            if len(cells) < 4:
                return None
            
            date_added = cells[0] if cells[0] else None
            poll_details = cells[1]
            
            # Parse poll details
            poll_info = self._parse_poll_details(poll_details)
            if not poll_info:
                return None
            
            # Build base poll object
            poll = {
                "date_added": date_added,
                "pollster": poll_info['pollster'],
                "date_range": poll_info['date_range'],
                "sample": poll_info['sample'],
                "grade": poll_info['grade'],
                "link": poll_info['link']
            }
            
            # Add race-specific results
            if race_type == 'approval':
                poll["approve"] = cells[2]
                poll["disapprove"] = cells[3]
                poll["margin"] = cells[4] if len(cells) > 4 else ""
            elif race_type == 'generic_ballot':
                poll["dem"] = cells[2]
                poll["gop"] = cells[3]
                poll["margin"] = cells[4] if len(cells) > 4 else ""
            else:
                # For candidate races, map results to headers if available
                result_cells = cells[2:]
                if headers and len(headers) > 2:
                    # Map result cells to header names
                    result_headers = headers[2:]  # Skip "Added" and poll columns
                    poll["results"] = {}
                    
                    # Filter headers to only include candidates (skip favor/unfav/margin columns)
                    skip_headers = ['favor', 'unfav', 'leader', 'margin', 'added', 'poll', 'sample', 'grade', 'date', '']
                    
                    for i, value in enumerate(result_cells):
                        if i < len(result_headers):
                            header = result_headers[i]
                            header_lower = header.lower().strip(':. ')
                            
                            # Skip non-candidate columns
                            if any(skip in header_lower for skip in skip_headers):
                                continue
                            
                            if header and value:
                                poll["results"][header] = value
                    
                    # Store margin separately if it exists
                    if 'margin' in [h.lower() for h in result_headers]:
                        margin_idx = [h.lower() for h in result_headers].index('margin')
                        if margin_idx < len(result_cells):
                            poll["margin"] = result_cells[margin_idx]
                    elif len(result_cells) > 0:
                        # Last cell might be margin
                        last_cell = result_cells[-1]
                        if '+' in last_cell or 'tie' in last_cell.lower():
                            poll["margin"] = last_cell
                else:
                    # Fallback to array
                    poll["results"] = result_cells
            
            return poll
            
        except Exception as e:
            logger.warning(f"Error parsing poll row: {e}")
            return None
    
    def _parse_poll_details(self, details: str) -> Optional[Dict[str, str]]:
        """
        Parse poll details string like:
        "Feb 2 - 8: Morning Consult (C-), 45000 LVLink: https://..."
        """
        try:
            # Split on "Link:" to separate poll info from URL
            parts = details.split('Link:')
            poll_text = parts[0].strip()
            link = parts[1].strip() if len(parts) > 1 else ""
            
            # Pattern: "Date range: Pollster (Grade), Sample"
            # Example: "Feb 2 - 8: Morning Consult (C-), 45000 LV"
            
            # Split on first colon to get date range
            if ':' not in poll_text:
                return None
            
            date_range, rest = poll_text.split(':', 1)
            date_range = date_range.strip()
            
            # Fix abbreviated years: "Dec 8 - 11, 25" -> "Dec 8 - 11, 2025"
            if ', ' in date_range:
                date_part, year_part = date_range.rsplit(', ', 1)
                if year_part.isdigit() and len(year_part) == 2:
                    # Convert 2-digit year to 4-digit
                    year_int = int(year_part)
                    full_year = 2000 + year_int if year_int >= 20 else 1900 + year_int
                    date_range = f"{date_part}, {full_year}"
            
            rest = rest.strip()
            
            # Extract pollster and grade (before the comma)
            if ',' in rest:
                pollster_part, sample = rest.split(',', 1)
                sample = sample.strip()
            else:
                pollster_part = rest
                sample = ""
            
            # Extract grade from parentheses
            grade_match = re.search(r'\(([^)]+)\)', pollster_part)
            if grade_match:
                grade = grade_match.group(1)
                pollster = pollster_part[:grade_match.start()].strip()
            else:
                grade = ""
                pollster = pollster_part.strip()
            
            return {
                "date_range": date_range,
                "pollster": pollster,
                "grade": grade,
                "sample": sample,
                "link": link
            }
            
        except Exception as e:
            logger.warning(f"Error parsing poll details '{details}': {e}")
            return None

    async def get_election_news(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Agent-driven news search. Return { query, articles } for the News tab."""
        from urllib.parse import urlparse
        from .election_ai_service import election_ai_service

        search_query = (query or "").strip()
        result = await election_ai_service.ask_news(search_query or None, model=model)
        if result.get("error"):
            fallback_articles = self._extract_articles_from_tool_steps(result.get("tool_steps") or [])
            if fallback_articles:
                return {
                    "query": search_query,
                    "articles": fallback_articles,
                    "warning": result["error"]
                }
            return {"query": search_query, "articles": [], "error": result["error"]}

        raw = (result.get("answer") or "").strip()
        parsed: Optional[Dict[str, Any]] = None
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                match = re.search(r"```(?:json)?\s*({.*})\s*```", raw, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        parsed = None
            if parsed is None:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end > start:
                    try:
                        parsed = json.loads(raw[start:end + 1])
                    except Exception:
                        parsed = None

        if not isinstance(parsed, dict):
            fallback_articles = self._extract_articles_from_tool_steps(result.get("tool_steps") or [])
            if fallback_articles:
                return {
                    "query": search_query,
                    "articles": fallback_articles,
                    "warning": "News agent returned an invalid response."
                }
            return {
                "query": search_query,
                "articles": [],
                "error": "News agent returned an invalid response."
            }

        articles_in = parsed.get("articles") or []
        normalized: List[Dict[str, Any]] = []
        for item in articles_in:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip() or "No title"
            snippet = (item.get("snippet") or "").strip()
            source = (item.get("publisher") or item.get("source") or "").strip()
            if not source and url:
                try:
                    parsed_url = urlparse(url)
                    netloc = (parsed_url.netloc or "").strip().replace("www.", "")
                    if netloc and "news.google" not in netloc.lower() and "bing.com" not in netloc.lower():
                        source = netloc
                except Exception:
                    pass
            if not source:
                source = "Web"
            normalized.append({
                "title": title,
                "source": source,
                "snippet": snippet[:300],
                "url": url
            })

        if not normalized:
            normalized = self._extract_articles_from_tool_steps(result.get("tool_steps") or [])

        return {
            "query": (parsed.get("query") or search_query).strip(),
            "articles": normalized
        }

    def _extract_articles_from_tool_steps(self, tool_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: parse tool response text into articles."""
        from urllib.parse import urlparse

        articles: List[Dict[str, Any]] = []
        seen: set = set()

        for step in tool_steps:
            text = (step.get("result") or step.get("result_preview") or "").strip()
            if not text:
                continue
            if "No results found" in text:
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            current: Optional[Dict[str, Any]] = None

            for line in lines:
                match = re.match(r"^\[(\d+)\]\s+(.*)$", line)
                if match:
                    if current and current.get("title"):
                        articles.append(current)
                    current = {"title": match.group(2).strip(), "url": "", "snippet": "", "source": ""}
                    continue

                if not current:
                    continue

                if line.startswith("Source: "):
                    current["url"] = line[len("Source: "):].strip()
                    continue
                if line.startswith("URL: "):
                    current["url"] = line[len("URL: "):].strip()
                    continue
                if line.startswith("Publisher: "):
                    current["source"] = line[len("Publisher: "):].strip()
                    continue
                if line.startswith("Snippet: ") and not current["snippet"]:
                    current["snippet"] = line[len("Snippet: "):].strip()
                    continue
                if line.startswith("Content: ") and not current["snippet"]:
                    current["snippet"] = line[len("Content: "):].strip()
                    continue

                if not current["snippet"] and not line.startswith("Search completed") and not line.startswith("Intent:") and not line.startswith("Found "):
                    current["snippet"] = line

            if current and current.get("title"):
                articles.append(current)

        normalized: List[Dict[str, Any]] = []
        for item in articles:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if not title:
                continue
            source = (item.get("source") or "").strip()
            if not source and url:
                try:
                    parsed_url = urlparse(url)
                    netloc = (parsed_url.netloc or "").strip().replace("www.", "")
                    if netloc and "news.google" not in netloc.lower() and "bing.com" not in netloc.lower():
                        source = netloc
                except Exception:
                    pass
            if not source:
                source = "Web"
            dedupe_key = (title.lower(), url.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append({
                "title": title,
                "source": source,
                "snippet": snippet[:300],
                "url": url
            })
            if len(normalized) >= 10:
                break

        return normalized

    def _poll_quality_weights(self, poll_list: List[Dict[str, Any]]) -> List[float]:
        """Weights for quality-weighted average: grade, sample size, recency. Returns list of float weights."""
        import math
        from datetime import date
        try:
            from .pollster_ratings import get_rating
        except ImportError:
            get_rating = None
        weights = []
        today = date.today()
        for p in poll_list:
            w_grade = 0.5
            if get_rating:
                r = get_rating(p.get("pollster") or "")
                if r and r.get("numeric_grade") is not None:
                    g = float(r["numeric_grade"])
                    w_grade = (g / 3.0) ** 0.5 if 0 <= g <= 3 else 0.5
            n = 600
            raw = (p.get("sample_size") or p.get("sample") or "").strip()
            if raw:
                num = re.sub(r"[^0-9]", "", raw)
                if num:
                    n = max(100, min(50000, int(num)))
            w_n = min(1.5, (n / 600) ** 0.5)
            w_time = 1.0
            iso = (p.get("date_iso") or "").strip()
            if iso and len(iso) >= 10:
                try:
                    from datetime import datetime
                    end = datetime.strptime(iso[:10], "%Y-%m-%d").date()
                    days_old = (today - end).days
                    w_time = math.exp(-days_old / 14.0)
                except Exception:
                    pass
            weights.append(w_grade * w_n * w_time)
        return weights if weights else []

    def _poll_date_for_weight(self, p: Dict[str, Any]) -> Optional[str]:
        """Parse date_iso or date_range to YYYY-MM-DD for recency weighting. Returns None if unparseable."""
        iso = (p.get("date_iso") or "").strip()
        if iso and len(iso) >= 10:
            return iso[:10]
        dr = (p.get("date_range") or "").strip()
        if not dr:
            return None
        # "Feb 2 - 8", "Dec 8 - 11, 2025"
        from datetime import date
        try:
            end_day_match = re.search(r"(\d+)\s*$", re.sub(r",?\s*\d{4}$", "", dr).strip())
            if not end_day_match:
                return None
            end_day = int(end_day_match.group(1))
            year_match = re.search(r",\s*(\d{4})$", dr)
            year = int(year_match.group(1)) if year_match else date.today().year
            # Find month: "Feb", "December", etc.
            month_names = "jan feb mar apr may jun jul aug sep oct nov dec"
            for i, name in enumerate(month_names.split(), 1):
                if name in dr.lower()[:20]:
                    d = date(year, i, min(end_day, 28 if i == 2 else 31))
                    return d.isoformat()
            return None
        except Exception:
            return None

    def compute_quality_weighted_generic_ballot(self, data: Dict[str, Any]) -> tuple:
        """
        Quality-weighted (538 grade, sample size, recency) generic ballot Dem share.
        data = response from get_polling_data('generic_ballot') with 'polls' or 'data' list.
        Returns (dem_share_0_to_100 or None, n_used, metadata).
        """
        polls_list = data.get("polls") or data.get("data") or []
        if not polls_list:
            return (None, 0, {"source": "generic_ballot", "n": 0})
        # Build list for weighting: same keys as _poll_quality_weights expects
        slim = []
        values = []
        for p in polls_list:
            dem_val = p.get("dem") or p.get("Democrat") or p.get("Candidate 1")
            gop_val = p.get("gop") or p.get("Republican") or p.get("Candidate 2")
            if dem_val is None or gop_val is None:
                continue
            try:
                d = float(str(dem_val).replace("%", "").strip())
                g = float(str(gop_val).replace("%", "").strip())
            except (ValueError, TypeError):
                continue
            if d + g <= 0:
                continue
            dem_share = 100.0 * d / (d + g)
            values.append(dem_share)
            slim.append({
                "pollster": p.get("pollster") or "",
                "sample": p.get("sample") or p.get("sample_size") or "",
                "date_iso": self._poll_date_for_weight(p) or "",
            })
        if not values or not slim:
            return (None, 0, {"source": "generic_ballot", "n": 0})
        weights = self._poll_quality_weights(slim)
        if len(weights) != len(values) or sum(weights) <= 0:
            # fallback simple average
            avg = sum(values) / len(values)
            return (round(avg, 2), len(values), {"source": "generic_ballot", "n": len(values), "weighted": False})
        total_w = sum(weights)
        weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_w
        return (round(weighted_avg, 2), len(values), {"source": "generic_ballot", "n": len(values), "weighted": True})

    def compute_quality_weighted_approval(self, data: Dict[str, Any]) -> tuple:
        """
        Quality-weighted (538 grade, sample size, recency) presidential approval net (approve - disapprove).
        data = response from get_polling_data('approval'). Returns (net_approval_pts or None, n_used, metadata).
        """
        polls_list = data.get("polls") or data.get("data") or []
        if not polls_list:
            return (None, 0, {"source": "approval", "n": 0})
        slim = []
        values = []
        for p in polls_list:
            app = p.get("approve") or p.get("Approve")
            dis = p.get("disapprove") or p.get("Disapprove")
            if app is None or dis is None:
                continue
            try:
                a = float(str(app).replace("%", "").strip())
                d = float(str(dis).replace("%", "").strip())
            except (ValueError, TypeError):
                continue
            net = a - d
            values.append(net)
            slim.append({
                "pollster": p.get("pollster") or "",
                "sample": p.get("sample") or p.get("sample_size") or "",
                "date_iso": self._poll_date_for_weight(p) or "",
            })
        if not values or not slim:
            return (None, 0, {"source": "approval", "n": 0})
        weights = self._poll_quality_weights(slim)
        if len(weights) != len(values) or sum(weights) <= 0:
            avg = sum(values) / len(values)
            return (round(avg, 2), len(values), {"source": "approval", "n": len(values), "weighted": False})
        total_w = sum(weights)
        weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_w
        return (round(weighted_avg, 2), len(values), {"source": "approval", "n": len(values), "weighted": True})

    def compute_state_averages_from_polls(
        self,
        polls: List[Dict[str, Any]],
        use_quality_weights: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build state-level averages from a list of poll dicts (same data as Polls tab).
        Derives state from poll['state'] or from poll['race'] when missing.
        When use_quality_weights=True, weights polls by 538 grade, sample size, and recency.
        Returns dict keyed by 2-letter state: { state: { dem_avg, gop_avg, poll_count, effective_n? } }.
        """
        from .election_db import _state_from_race_text
        # Same result keys as trends/map DB query
        def _dem_from_results(r: Dict) -> Optional[float]:
            if not r:
                return None
            v = r.get("Dem") or r.get("Democrat") or r.get("Candidate 1") or r.get("Result1")
            if v is None:
                return None
            try:
                return float(str(v).replace("%", "").strip())
            except (ValueError, TypeError):
                return None

        def _gop_from_results(r: Dict) -> Optional[float]:
            if not r:
                return None
            v = r.get("GOP") or r.get("Republican") or r.get("Candidate 2") or r.get("Result2") or r.get("Rep")
            if v is None:
                return None
            try:
                return float(str(v).replace("%", "").strip())
            except (ValueError, TypeError):
                return None

        def _parse_margin_leader(margin_str: Optional[str]) -> Optional[tuple]:
            """Return ('D', lead) or ('R', lead) from margin e.g. 'Abbott (R) +8', 'R+8', 'Democrats +4'. None if tie or unparseable."""
            if not margin_str or not isinstance(margin_str, str):
                return None
            m = str(margin_str).strip()
            if not m or "tie" in m.lower():
                return None
            # (R) +8, (D) +4, R+8, D+4, Republican +8, Democrats +4
            num_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*%?", m)
            lead = float(num_match.group(1).replace(" ", "").replace("+", "")) if num_match else None
            if lead is None:
                return None
            m_lower = m.lower()
            if "(r)" in m_lower or "republican" in m_lower or m.strip().startswith("r+") or " r+" in m_lower:
                return ("R", lead)
            if "(d)" in m_lower or "democrat" in m_lower or m.strip().startswith("d+") or " d+" in m_lower:
                return ("D", lead)
            # Candidate name + number: leader is the one with higher share; we use lead sign if present
            if "+" in m and "(" in m:
                if "(r)" in m_lower or "(r )" in m_lower:
                    return ("R", lead)
                if "(d)" in m_lower or "(d )" in m_lower:
                    return ("D", lead)
            return None

        def _dem_gop_from_results(
            r: Dict,
            margin_str: Optional[str] = None,
            state: Optional[str] = None,
            office: Optional[str] = None,
        ) -> tuple:
            """Return (dem_pct, gop_pct). Prefer 2026 candidate roster, then margin string, else column order."""
            dem = _dem_from_results(r)
            gop = _gop_from_results(r)
            if dem is not None and gop is not None:
                return (dem, gop)
            names_vals = []
            for k, v in (r or {}).items():
                try:
                    pct = float(str(v).replace("%", "").strip())
                    names_vals.append((str(k).strip(), pct))
                except (ValueError, TypeError):
                    continue
            if len(names_vals) >= 2:
                try:
                    from .election_candidates import get_dem_gop_from_candidate_names
                    pair = get_dem_gop_from_candidate_names(
                        [(n, p) for n, p in names_vals[:2]],
                        state=state,
                        office=office,
                    )
                    if pair is not None:
                        return pair
                except Exception:
                    pass
                leader = _parse_margin_leader(margin_str)
                if leader is not None:
                    low, high = min(v for _, v in names_vals[:2]), max(v for _, v in names_vals[:2])
                    if leader[0] == "R":
                        return (low, high)
                    return (high, low)
                return (None, None)
            if len(names_vals) == 1:
                return (names_vals[0][1], None) if dem is None and gop is None else (dem, gop)
            return (dem, gop)

        by_state: Dict[str, List[tuple]] = {}
        poll_slims_by_state: Dict[str, List[Dict[str, Any]]] = {}
        for p in polls or []:
            race_text = (p.get("race") or p.get("race_key") or "").lower()
            if "primary" in race_text:
                continue
            state = (p.get("state") or "").strip()
            if not state and p.get("race"):
                state = _state_from_race_text(str(p.get("race", "")).strip()) or ""
            state = (state or "").upper().strip()
            if len(state) != 2 or not state.isalpha():
                continue
            results = p.get("results") or {}
            margin_str = p.get("margin") or p.get("lead") or ""
            office = (p.get("race_type") or "").strip().lower() or None
            dem, gop = _dem_gop_from_results(results, margin_str, state=state, office=office)
            if dem is None and gop is None:
                continue
            if state not in by_state:
                by_state[state] = []
                poll_slims_by_state[state] = []
            by_state[state].append((dem, gop))
            poll_slims_by_state[state].append({
                "pollster": p.get("pollster") or "",
                "race": p.get("race") or p.get("race_key") or "",
                "margin": p.get("margin") or p.get("lead") or "",
                "added": p.get("added") or p.get("date_added") or "",
                "date_iso": p.get("date_iso") or "",
                "sample_size": p.get("sample_size") or p.get("sample") or "",
                "dem_pct": dem,
                "gop_pct": gop,
                "results": results,
            })
        # Pollster-based confidence (FiveThirtyEight ratings); does not change averages
        try:
            from .pollster_ratings import get_state_confidence_from_pollsters, get_rating
        except ImportError:
            get_state_confidence_from_pollsters = None
            get_rating = None

        out = {}
        for state, pairs in by_state.items():
            dems = [x[0] for x in pairs if x[0] is not None]
            gops = [x[1] for x in pairs if x[1] is not None]
            poll_list = poll_slims_by_state.get(state, [])
            dem_avg = sum(dems) / len(dems) if dems else None
            gop_avg = sum(gops) / len(gops) if gops else None
            effective_n = float(len(pairs))
            if use_quality_weights and pairs and poll_list:
                weights = self._poll_quality_weights(poll_list)
                if len(weights) == len(pairs) and sum(weights) > 0:
                    total_w = sum(weights)
                    dem_num = sum((d or 0) * w for (d, _), w in zip(pairs, weights))
                    gop_num = sum((g or 0) * w for (_, g), w in zip(pairs, weights))
                    dem_avg = dem_num / total_w
                    gop_avg = gop_num / total_w
                    effective_n = total_w
            entry = {
                "dem_avg": dem_avg,
                "gop_avg": gop_avg,
                "poll_count": len(pairs),
                "polls": poll_list,
            }
            if use_quality_weights:
                entry["effective_n"] = round(effective_n, 2)
            if get_state_confidence_from_pollsters:
                pollster_names = [p.get("pollster") or "" for p in poll_list if p.get("pollster")]
                conf, note = get_state_confidence_from_pollsters(pollster_names)
                entry["pollster_confidence"] = conf
                entry["pollster_confidence_note"] = note
            else:
                entry["pollster_confidence"] = None
                entry["pollster_confidence_note"] = None
            out[state] = entry
        return out

    async def get_historical_data(self, year: Optional[str] = None, race_type: Optional[str] = None) -> Dict[str, Any]:
        """Return historical election results from local JSON file."""
        if not self._historical_data_cache:
            try:
                if self.historical_data_path.exists():
                    with open(self.historical_data_path, 'r') as f:
                        self._historical_data_cache = json.load(f)
                else:
                    logger.warning("Historical data file not found")
                    return {}
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                return {}

        data = self._historical_data_cache

        # Default to president if not specified
        target_race = race_type or 'president'
        data = data.get(target_race, {})

        if year and isinstance(data, dict):
            data = data.get(str(year), {})

        # Transform president year payload to frontend shape: dem, gop, popular_dem, popular_gop
        if isinstance(data, dict) and data.get("electoral_votes") and data.get("popular_vote") and data.get("winner"):
            ev = data["electoral_votes"]
            pv = data["popular_vote"]
            winner = data["winner"]
            party = (data.get("party") or "").upper()
            other = next((k for k in ev if k != winner), None)
            if other is not None:
                if party == "REP":
                    dem_ev, gop_ev = ev.get(other, 0), ev.get(winner, 0)
                    dem_pv, gop_pv = pv.get(other, 0), pv.get(winner, 0)
                else:
                    dem_ev, gop_ev = ev.get(winner, 0), ev.get(other, 0)
                    dem_pv, gop_pv = pv.get(winner, 0), pv.get(other, 0)
                data = {**data, "dem": dem_ev, "gop": gop_ev}
                data["popular_dem"] = f"{dem_pv / 1e6:.1f}M" if dem_pv else ""
                data["popular_gop"] = f"{gop_pv / 1e6:.1f}M" if gop_pv else ""

        return data

    def _get_url_for_race(self, race_type: str, state: Optional[str]) -> Optional[str]:
        """Construct the URL for RaceToTheWH based on race type."""
        race_type = race_type.lower()
        
        if race_type == 'approval':
            return f"{self.base_url}/trump"
        elif race_type == 'generic_ballot':
            return f"{self.base_url}/polls/genericballot"
        elif race_type == 'senate' or race_type == 'senate_primary':
            return f"{self.base_url}/senate/26polls"
        elif race_type == 'governor' or race_type == 'governor_primary':
            return f"{self.base_url}/governor/26polls"
        elif race_type == 'house' or race_type == 'house_primary':
            return f"{self.base_url}/house/26polls"
        
        return None

    def _is_cached(self, key: str) -> bool:
        """Check if a cache key is valid."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return True
        return False

    def clear_poll_cache(self, race_type: str, state: Optional[str] = None) -> None:
        """Clear in-memory poll cache for a race type so the next fetch re-scrapes. Call before manual refresh."""
        key = f"polls_{race_type}_{state or 'national'}"
        self.cache.pop(key, None)
        logger.info(f"Cleared poll cache for {key}")

    def _cache_data(self, key: str, data: Any):
        """Cache data with current timestamp."""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

    def _get_local_time(self) -> datetime:
        if ZoneInfo:
            try:
                return datetime.now(ZoneInfo("America/New_York"))
            except Exception:
                pass
        return datetime.now().astimezone()

    async def _maybe_save_poll_snapshot(self, cache_key: str, url: str, data: Dict[str, Any]):
        """Persist a snapshot of the latest polls for trend analysis."""
        if not data or data.get("error"):
            return
        polls = data.get("polls") or []
        if not polls:
            return

        race_type = data.get("race_type") or cache_key.replace("polls_", "")
        local_now = self._get_local_time()
        utc_now = datetime.now(timezone.utc)

        payload = {
            "race_type": race_type,
            "cache_key": cache_key,
            "source_url": url,
            "timestamp_utc": utc_now.isoformat(),
            "timestamp_local": local_now.isoformat(),
            "date_local": local_now.strftime("%Y-%m-%d"),
            "poll_count": len(polls),
            "metadata": data.get("metadata") or {},
            "polls": polls
        }

        # Hash to avoid spamming duplicates
        try:
            polls_hash = hash(json.dumps(polls, sort_keys=True))
        except Exception:
            polls_hash = None

        async with self._snapshot_lock:
            last = self._last_snapshot_meta.get(race_type)
            if last and polls_hash is not None and last.get("hash") == polls_hash:
                # Data unchanged: do not store a duplicate snapshot
                return

            self.poll_snapshots_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.poll_snapshots_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload))
                f.write("\n")

            self._last_snapshot_meta[race_type] = {
                "timestamp": time.time(),
                "hash": polls_hash
            }

    async def get_poll_snapshots(self, race_type: Optional[str] = None, days: int = 7, limit: int = 200) -> Dict[str, Any]:
        """Read poll snapshots for a given race type and recent days."""
        if not self.poll_snapshots_path.exists():
            return {
                "race_type": race_type,
                "days": days,
                "snapshots": [],
                "error": "No snapshots available"
            }

        local_now = self._get_local_time()
        cutoff_date = (local_now.date() - timedelta(days=max(days - 1, 0))).strftime("%Y-%m-%d")

        snapshots: List[Dict[str, Any]] = []
        try:
            with open(self.poll_snapshots_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snap = json.loads(line)
                    except Exception:
                        continue

                    snap_race = snap.get("race_type")
                    if race_type and snap_race != race_type:
                        continue

                    date_local = snap.get("date_local") or ""
                    if date_local and date_local < cutoff_date:
                        continue

                    snapshots.append(snap)

            if limit and len(snapshots) > limit:
                snapshots = snapshots[-limit:]

            return {
                "race_type": race_type,
                "days": days,
                "snapshots": snapshots
            }
        except Exception as e:
            logger.error(f"Error reading poll snapshots: {e}")
            return {
                "race_type": race_type,
                "days": days,
                "snapshots": [],
                "error": str(e)
            }

def _state_from_rtwh_race(race: str) -> Optional[str]:
    """Extract state from RTWH race string. Uses shared logic from election_db for (TX-R), 'Alabama GOP Primary', 'OH - ...'."""
    from .election_db import _state_from_race_text
    return _state_from_race_text(race)


def normalize_rtwh_polls(data: Dict[str, Any], race_type: str) -> List[Dict[str, Any]]:
    """Convert RaceToTheWH API response to DB schema for upsert."""
    from datetime import datetime, timezone
    from .election_db import make_poll_id
    polls_raw = data.get("polls") or []
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for p in polls_raw:
        results = p.get("results") or {}
        race_key = p.get("race") or p.get("race_key") or f"RTWH-{race_type}"
        # If race is just a district number (e.g. "1", "7"), show as "District 1" so Race or Candidate column isn't broken
        rk_strip = (race_key or "").strip()
        if rk_strip.isdigit():
            race_key = f"District {rk_strip}"
        start_date = p.get("date_added") or p.get("date_range") or ""
        end_date = p.get("date_added") or ""
        margin = p.get("margin") or p.get("lead") or ""
        poll_id = make_poll_id("racetothewh", p.get("pollster", ""), start_date, race_key)
        state = (p.get("state") or "").strip() or _state_from_rtwh_race(race_key) or _state_from_rtwh_race(p.get("race") or p.get("race_key") or "")
        if state and len(state) == 2 and state.isalpha():
            state = state.upper()
        out.append({
            "id": poll_id,
            "source": "racetothewh",
            "race_type": race_type,
            "race_key": race_key,
            "state": state,
            "pollster": p.get("pollster", ""),
            "poll_url": p.get("link", ""),
            "start_date": start_date,
            "end_date": end_date,
            "date_added": p.get("date_added") or p.get("added", ""),
            "sample_size": p.get("sample", "") or p.get("sample_size", ""),
            "population": "",
            "grade": p.get("grade", ""),
            "margin": margin,
            "results": results,
            "raw_data": p,
            "fetched_at": now,
        })
    return out


# Global instance
election_service = ElectionDataService()
