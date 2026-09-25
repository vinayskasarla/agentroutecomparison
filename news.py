"""Model news: launches, pricing changes and retirements from official feeds, checked against our catalog.

Sources live in knowledge/news_sources.json. Each is fetched on demand (cached for 30 minutes); a source that
can't be reached is reported as such rather than hiding the rest.
"""
import asyncio
import datetime
import email.utils
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import httpx

import catalog

SOURCES = json.load(open(os.path.join(catalog.KNOWLEDGE_DIR, "news_sources.json")))["sources"]
CACHE: dict = {"at": 0, "data": None}
TTL = 30 * 60

MODEL_WORDS = re.compile(r"\b(model|models|llm|gpt|claude|gemini|grok|llama|nova|mistral|deepseek|qwen|phi|o\d|"
                         r"pricing|price|launch|introduc|release|deprecat|retire|context window|tokens?)\b", re.I)
CATEGORY = [
    ("retirement", re.compile(r"deprecat|retire|sunset|end of life|shut(ting)? down|no longer available", re.I)),
    ("new_model", re.compile(r"launch|introduc|now available|generally available|\bga\b|new model|releas|announc", re.I)),
    ("pricing", re.compile(r"pric|\$\s?\d|per (million|mtok)|cost|discount|cheaper", re.I)),
]
# Model names as they appear in announcements, e.g. "Claude Opus 5.5", "GPT-5.6 Sol", "Gemini 3.1 Pro", "Grok 4.7"
MODEL_NAME = re.compile(
    r"\b(Claude (?:Opus|Sonnet|Haiku|Fable|Mythos) \d+(?:\.\d+)?|GPT-\d+(?:\.\d+)?(?:[- ](?:mini|nano|Sol|Terra|Luna|Astra|Pro))?|"
    r"Gemini \d+(?:\.\d+)? (?:Pro|Flash(?:-Lite)?|Ultra)|Grok \d+(?:\.\d+)?(?: Fast)?|Llama \d+(?:\.\d+)? \w+|"
    r"(?:Amazon )?Nova (?:\d\.\d )?(?:Micro|Lite|Pro|Premier|Omni)|Mistral (?:Large|Medium|Small) \d*|DeepSeek[- ]v?\d+(?:\.\d+)?|"
    r"Qwen\d+(?:\.\d+)?[\w-]*)\b", re.I)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("amazon", ""))


def _in_catalog(name):
    n = _norm(name)
    return any(n == _norm(m["label"]) for m in catalog.MODELS)


def _date(s):
    if not s:
        return None
    try:
        return email.utils.parsedate_to_datetime(s).astimezone(datetime.timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


def _clean(text, limit=280):
    text = html.unescape(re.sub(r"<[^>]+>|\[([^\]]+)\]\([^)]+\)|`|\*\*", lambda m: m.group(1) or "", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def parse_rss(text, src):
    root = ET.fromstring(text.lstrip("﻿").encode())
    items = []
    atom = "{http://www.w3.org/2005/Atom}"
    for it in root.iter("item"):
        items.append({"title": it.findtext("title"), "link": it.findtext("link"), "date": _date(it.findtext("pubDate")),
                      "summary": it.findtext("description"), "tags": [c.text for c in it.findall("category") if c.text]})
    for e in root.iter(f"{atom}entry"):
        link = e.find(f"{atom}link")
        items.append({"title": e.findtext(f"{atom}title"), "link": link.get("href") if link is not None else src.get("url"),
                      "date": _date(e.findtext(f"{atom}updated") or e.findtext(f"{atom}published")),
                      "summary": e.findtext(f"{atom}summary") or e.findtext(f"{atom}content"), "tags": []})
    return items


def parse_release_notes(text, src):
    items = []
    for block in re.split(r"\n### ", text)[1:]:
        heading, _, body = block.partition("\n")
        try:
            date = datetime.datetime.strptime(heading.strip(), "%B %d, %Y").date().isoformat()
        except ValueError:
            continue
        for bullet in re.findall(r"^\* (.+)$", body, re.M):
            plain = _clean(bullet, 1000)
            title = re.split(r"(?<=[.!?])\s", plain, 1)[0]
            items.append({"title": title[:180], "link": src.get("link", src["url"]), "date": date, "summary": plain, "tags": []})
    return items


def classify(item):
    text = f"{item['title']} {item.get('summary') or ''}"
    cat = next((c for c, rx in CATEGORY if rx.search(text)), "feature")
    names = sorted({re.sub(r"\s+", " ", m.group(0)).strip() for m in MODEL_NAME.finditer(text)})
    return cat, [{"name": n, "in_catalog": _in_catalog(n)} for n in names]


async def fetch_source(client, src):
    try:
        r = await client.get(src["url"])
        r.raise_for_status()
        raw = parse_release_notes(r.text, src) if src["type"] == "markdown_release_notes" else parse_rss(r.text, src)
    except Exception as exc:
        return src, [], f"{type(exc).__name__}: {str(exc)[:120]}"
    out = []
    must = re.compile(src["must_match"], re.I) if src.get("must_match") else None
    for it in raw:
        text = f"{it['title'] or ''} {it.get('summary') or ''} {' '.join(it.get('tags') or [])}"
        if not MODEL_WORDS.search(text) or (must and not must.search(text)):
            continue
        cat, models = classify(it)
        out.append({"source": src["name"], "provider": src["provider"], "title": _clean(it["title"], 200), "link": it["link"],
                    "date": it["date"], "summary": _clean(it.get("summary")), "category": cat, "models": models})
    return src, out[:40], None


async def get_news(refresh=False):
    if not refresh and CACHE["data"] and time.time() - CACHE["at"] < TTL:
        return CACHE["data"]
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": "architecture-advisor/1.0"}) as client:
        results = await asyncio.gather(*[fetch_source(client, s) for s in SOURCES])
    items = sorted([i for _, its, _ in results for i in its], key=lambda i: i["date"] or "", reverse=True)
    new_names = sorted({m["name"] for i in items for m in i["models"] if not m["in_catalog"]})
    data = {"fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "sources": [{"name": s["name"], "provider": s["provider"], "url": s["url"], "ok": err is None, "items": len(its), "error": err}
                        for s, its, err in results],
            "items": items, "not_in_catalog": new_names}
    CACHE.update(at=time.time(), data=data)
    return data
