"""网络搜索封装 — 多引擎后备

优先 DuckDuckGo（国际），如果被墙则尝试 Bing。
在终端输出搜索来源方便调试。
"""

import re
from ddgs import DDGS
import requests


def _search_ddg(query: str, max_results: int) -> list[dict] | None:
    """DuckDuckGo 搜索（静默失败，切到备用引擎）"""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
            for r in raw
        ]
    except Exception:
        return None

#用得是原生request库拼接
def _search_bing(query: str, max_results: int) -> list[dict]:
    """Bing 搜索（在国内可用）"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = "https://www.bing.com/search"
        resp = requests.get(url, params={"q": query, "count": max_results}, headers=headers, timeout=10)
        resp.raise_for_status()

        results = []
        for li in re.findall(r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)[:max_results]:
            href_match = re.search(r'<a[^>]*href="(https?://[^"]+)"', li)
            title_match = re.search(r'<a[^>]*>(.*?)</a>', li)
            body_match = re.search(r'<p[^>]*>(.*?)</p>', li, re.DOTALL)
            if href_match:
                results.append({
                    "title": _clean_html(title_match.group(1)) if title_match else "",
                    "href": href_match.group(1),
                    "body": _clean_html(body_match.group(1)) if body_match else "",
                })
        return results
    except Exception as e:
        print(f"  [搜索] Bing 不可用 ({e.__class__.__name__})")
        return []


def _clean_html(text: str) -> str:
    """去除 HTML 标签和多余空白"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """搜索网络，返回标题/链接/摘要列表

    引擎优先级：DuckDuckGo → Bing → 空结果
    Args:
        query: 搜索关键词
        max_results: 最多返回几条结果（默认 5）
    Returns:
        [{title, href, body}, ...]
    """
    results = _search_ddg(query, max_results)
    if results is not None:
        return results
    return _search_bing(query, max_results)
