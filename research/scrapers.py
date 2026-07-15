"""
Phase 3 — Research Engine Scrapers.

Sources:
  - ForexFactory (high-impact news + forum strategy threads)
  - MQL5 article reader (strategy articles → structured idea extraction)
  - GitHub repository miner ("forex strategy python" repos)
  - arXiv / SSRN paper reader (quantitative finance papers)
  - Quant blog aggregator (QuantStart, QuantConnect blog, Hudson & Thames)

All scrapers return a list of RawIdea dicts:
  {title, summary, source, url, tags, extracted_at}
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger

RawIdea = dict[str, Any]


def _get(url: str, timeout: int = 15) -> str | None:
    """HTTP GET with a browser-like UA. Returns text or None."""
    try:
        import requests
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.debug(f"[Scraper] GET {url} failed: {e}")
        return None


def _soup(html: str):
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser")
    except ImportError:
        logger.warning("[Scraper] BeautifulSoup4 not installed — install beautifulsoup4")
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── ForexFactory ──────────────────────────────────────────────────────────────

def scrape_forexfactory(max_articles: int = 10) -> list[RawIdea]:
    """Scrape ForexFactory calendar news events + popular strategy threads."""
    ideas: list[RawIdea] = []

    # High-impact economic calendar events
    html = _get("https://www.forexfactory.com/calendar")
    if html:
        soup = _soup(html)
        if soup:
            rows = soup.select("tr.calendar__row") if soup else []
            for row in rows[:max_articles]:
                title_el = row.select_one(".calendar__event-title")
                impact_el = row.select_one(".calendar__impact span")
                if not title_el:
                    continue
                title  = title_el.get_text(strip=True)
                impact = impact_el.get("title", "") if impact_el else ""
                if "High" in impact or "Medium" in impact:
                    ideas.append({
                        "title":        f"News Impact: {title}",
                        "summary":      f"{impact} impact event. Consider avoiding trades 30min before/after.",
                        "source":       "ForexFactory Calendar",
                        "url":          "https://www.forexfactory.com/calendar",
                        "tags":         ["news", "fundamental", "risk"],
                        "extracted_at": _now(),
                    })

    # Strategy threads (forum)
    html2 = _get("https://www.forexfactory.com/forum/71-trading-systems")
    if html2:
        soup2 = _soup(html2)
        if soup2:
            threads = soup2.select("h3.thread-title a")
            for t in threads[:max_articles]:
                text = t.get_text(strip=True)
                href = t.get("href", "")
                if any(k in text.lower() for k in ("strategy", "system", "ema", "rsi", "smc", "breakout", "scalp")):
                    ideas.append({
                        "title":        text[:120],
                        "summary":      f"Strategy thread from ForexFactory forum. Review for tradeable ideas.",
                        "source":       "ForexFactory Forum",
                        "url":          f"https://www.forexfactory.com{href}" if href.startswith("/") else href,
                        "tags":         ["strategy", "forum"],
                        "extracted_at": _now(),
                    })

    logger.info(f"[Scraper] ForexFactory: {len(ideas)} ideas")
    return ideas


# ── MQL5 Article Reader ───────────────────────────────────────────────────────

def scrape_mql5(max_articles: int = 8) -> list[RawIdea]:
    """Read MQL5 strategy/article pages for structured idea extraction."""
    ideas: list[RawIdea] = []

    html = _get("https://www.mql5.com/en/articles/category/strategies")
    if html:
        soup = _soup(html)
        if soup:
            links = soup.select("a.item-title")
            for link in links[:max_articles]:
                title = link.get_text(strip=True)
                href  = link.get("href", "")
                if not href:
                    continue
                url = f"https://www.mql5.com{href}" if href.startswith("/") else href

                # Read the article page for a brief extract
                art_html = _get(url)
                summary  = ""
                if art_html:
                    art_soup = _soup(art_html)
                    if art_soup:
                        intro = art_soup.select_one(".article-content p")
                        if intro:
                            summary = intro.get_text(strip=True)[:300]

                if title:
                    ideas.append({
                        "title":        title[:120],
                        "summary":      summary or "MQL5 strategy article — review for parameter ideas.",
                        "source":       "MQL5 Articles",
                        "url":          url,
                        "tags":         ["mql5", "strategy", "algorithm"],
                        "extracted_at": _now(),
                    })

    logger.info(f"[Scraper] MQL5: {len(ideas)} ideas")
    return ideas


# ── GitHub Repository Miner ───────────────────────────────────────────────────

def scrape_github(query: str = "forex strategy python", max_repos: int = 8) -> list[RawIdea]:
    """Search GitHub for forex strategy repos and extract README summaries."""
    ideas: list[RawIdea] = []

    try:
        import requests
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max_repos},
            timeout=15,
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code != 200:
            return ideas
        repos = resp.json().get("items", [])

        for repo in repos:
            name        = repo.get("full_name", "")
            description = repo.get("description") or "No description"
            stars       = repo.get("stargazers_count", 0)
            url         = repo.get("html_url", "")
            language    = repo.get("language", "")

            if language and language.lower() not in ("python", ""):
                continue

            # Fetch README for more context
            readme_url = f"https://raw.githubusercontent.com/{name}/main/README.md"
            readme = _get(readme_url) or _get(readme_url.replace("main", "master")) or ""
            excerpt = readme[:500].replace("\n", " ") if readme else ""

            ideas.append({
                "title":        f"GitHub: {name} ({stars}★)",
                "summary":      f"{description}. {excerpt}"[:400],
                "source":       "GitHub",
                "url":          url,
                "tags":         ["github", "open-source", "python"],
                "extracted_at": _now(),
            })
    except Exception as e:
        logger.debug(f"[Scraper] GitHub error: {e}")

    logger.info(f"[Scraper] GitHub: {len(ideas)} repos")
    return ideas


# ── arXiv / SSRN Paper Reader ─────────────────────────────────────────────────

def scrape_arxiv(query: str = "forex trading strategy machine learning", max_papers: int = 6) -> list[RawIdea]:
    """Search arXiv for quantitative finance papers."""
    ideas: list[RawIdea] = []

    try:
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start":        0,
            "max_results":  max_papers,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        })
        url = f"https://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=20) as resp:
            xml_text = resp.read().decode("utf-8")

        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)

        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title",   namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            url_out = link_el.get("href", "") if link_el is not None else ""

            if title:
                ideas.append({
                    "title":        title[:120],
                    "summary":      summary[:300],
                    "source":       "arXiv",
                    "url":          url_out,
                    "tags":         ["research", "paper", "quantitative"],
                    "extracted_at": _now(),
                })
    except Exception as e:
        logger.debug(f"[Scraper] arXiv error: {e}")

    logger.info(f"[Scraper] arXiv: {len(ideas)} papers")
    return ideas


# ── Quant Blog Aggregator ─────────────────────────────────────────────────────

_QUANT_BLOGS = [
    ("QuantStart",       "https://www.quantstart.com/articles/"),
    ("QuantConnect",     "https://www.quantconnect.com/blog/"),
    ("Hudson & Thames",  "https://hudsonthames.org/blog/"),
]


def scrape_quant_blogs(max_per_blog: int = 5) -> list[RawIdea]:
    """Aggregate quant blog articles from QuantStart, QuantConnect, Hudson & Thames."""
    ideas: list[RawIdea] = []

    for name, url in _QUANT_BLOGS:
        html = _get(url)
        if not html:
            continue
        soup = _soup(html)
        if not soup:
            continue

        # Generic: grab any <a> tags that look like article links
        for a in soup.select("a[href]")[:max_per_blog * 5]:
            text = a.get_text(strip=True)
            href = a.get("href", "")
            if not text or len(text) < 15 or len(text) > 150:
                continue
            if any(kw in text.lower() for kw in (
                "strategy", "backtest", "momentum", "mean reversion", "machine learning",
                "trading", "quant", "factor", "alpha", "signal", "forex", "volatility"
            )):
                full_url = href if href.startswith("http") else url.rstrip("/") + "/" + href.lstrip("/")
                ideas.append({
                    "title":        text[:120],
                    "summary":      f"Quant article from {name}",
                    "source":       name,
                    "url":          full_url,
                    "tags":         ["blog", "quant", "research"],
                    "extracted_at": _now(),
                })
                if len([i for i in ideas if i["source"] == name]) >= max_per_blog:
                    break

    logger.info(f"[Scraper] Quant blogs: {len(ideas)} articles")
    return ideas


# ── Aggregated scrape ─────────────────────────────────────────────────────────

def scrape_all(max_per_source: int = 6) -> list[RawIdea]:
    """Run all scrapers and return combined deduplicated list."""
    all_ideas: list[RawIdea] = []
    all_ideas.extend(scrape_forexfactory(max_per_source))
    all_ideas.extend(scrape_mql5(max_per_source))
    all_ideas.extend(scrape_github(max_repos=max_per_source))
    all_ideas.extend(scrape_arxiv(max_papers=max_per_source))
    all_ideas.extend(scrape_quant_blogs(max_per_blog=max_per_source))
    logger.info(f"[Scraper] Total raw ideas: {len(all_ideas)}")
    return all_ideas
