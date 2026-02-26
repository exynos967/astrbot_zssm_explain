from __future__ import annotations

from datetime import datetime
from html import unescape
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlsplit, urlunsplit

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

WECHAT_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Mobile/15E148 MicroMessenger/8.0.40(0x1800282b) "
    "NetType/WIFI Language/zh_CN"
)


def is_wechat_article_url(url: str) -> bool:
    """Return whether URL is a WeChat article URL."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        p = urlparse(url.strip())
        host = (p.netloc or "").lower()
        path = p.path or ""
        if not host.endswith("mp.weixin.qq.com"):
            return False
        return path == "/s" or path.startswith("/s/")
    except Exception:
        return False


def ensure_mobile_article_url(url: str) -> str:
    sp = urlsplit(url)
    query_map = parse_qs(sp.query, keep_blank_values=True)
    query_map["nwr_flag"] = ["1"]
    query = urlencode([(k, v) for k, vals in query_map.items() for v in vals])
    return urlunsplit((sp.scheme or "https", sp.netloc, sp.path, query, ""))


def _wechat_headers(referer: str = "https://mp.weixin.qq.com/") -> Dict[str, str]:
    return {
        "User-Agent": WECHAT_MOBILE_UA,
        "Referer": referer,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "X-Requested-With": "com.tencent.mm",
    }


def _normalize_url(url: str) -> str:
    url = unescape((url or "").strip())
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin("https://mp.weixin.qq.com", url)
    return url


def _extract_js_var(page_html: str, name: str) -> str:
    pattern = rf"\bvar\s+{re.escape(name)}\s*=\s*([^;]+);"
    match = re.search(pattern, page_html)
    if not match:
        return ""
    expr = match.group(1).strip()
    quoted = re.search(r'"([^"]*)"|\'([^\']*)\'', expr)
    if quoted:
        return quoted.group(1) or quoted.group(2) or ""
    number = re.search(r"-?\d+", expr)
    return number.group(0) if number else ""


def _strip_html(raw_html: str) -> str:
    raw_html = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    raw_html = re.sub(r"<style[\s\S]*?</style>", " ", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text_by_patterns(page_html: str, patterns: List[str]) -> str:
    for pat in patterns:
        m = re.search(pat, page_html, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        text = _strip_html(m.group(1))
        if text:
            return text
    return ""


def _extract_content_html(page_html: str) -> str:
    open_tag = re.search(
        r'<div[^>]*\bid=["\']js_content["\'][^>]*>',
        page_html,
        flags=re.IGNORECASE,
    )
    if not open_tag:
        return ""
    start = open_tag.end()
    depth = 1
    for m in re.finditer(r"</?div\b[^>]*>", page_html[start:], flags=re.IGNORECASE):
        tag = m.group(0).lower()
        if tag.startswith("</div"):
            depth -= 1
        else:
            depth += 1
        if depth == 0:
            end = start + m.start()
            return page_html[start:end]
    return ""


def _html_to_text_keep_lines(content_html: str) -> str:
    text = re.sub(r"<\s*br\s*/?>", "\n", content_html, flags=re.IGNORECASE)
    text = re.sub(
        r"</\s*(p|div|li|h1|h2|h3|h4|h5|h6|section|article)\s*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_image_urls(content_html: str) -> List[str]:
    urls: List[str] = []
    seen = set()
    for m in re.finditer(r"<img\b[^>]*>", content_html, flags=re.IGNORECASE):
        tag = m.group(0)
        src_match = re.search(
            r'\bdata-src=["\']([^"\']+)["\']|\bsrc=["\']([^"\']+)["\']',
            tag,
            flags=re.IGNORECASE,
        )
        if not src_match:
            continue
        src = src_match.group(1) or src_match.group(2) or ""
        src = _normalize_url(src)
        if not src or src in seen:
            continue
        seen.add(src)
        urls.append(src)
    return urls


def _parse_markdown(article_url: str, page_html: str) -> Dict[str, str]:
    title = _extract_text_by_patterns(
        page_html,
        [
            r'<h1[^>]*id=["\']activity-name["\'][^>]*>(.*?)</h1>',
            r'<h1[^>]*class=["\'][^"\']*rich_media_title[^"\']*["\'][^>]*>(.*?)</h1>',
            r"<title[^>]*>(.*?)</title>",
        ],
    )
    account = _extract_text_by_patterns(
        page_html,
        [
            r'<a[^>]*id=["\']js_name["\'][^>]*>(.*?)</a>',
            r'<span[^>]*id=["\']js_name["\'][^>]*>(.*?)</span>',
        ],
    )
    author = _extract_text_by_patterns(
        page_html,
        [r'<span[^>]*class=["\'][^"\']*rich_media_meta_text[^"\']*["\'][^>]*>(.*?)</span>'],
    )
    nickname = _extract_js_var(page_html, "nickname")
    if not account and nickname:
        account = nickname

    ct = _extract_js_var(page_html, "ct")
    publish_time = ""
    if ct.isdigit():
        try:
            publish_time = datetime.fromtimestamp(int(ct)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            publish_time = ""

    content_html = _extract_content_html(page_html)
    content_text = _html_to_text_keep_lines(content_html) if content_html else ""
    image_urls = _extract_image_urls(content_html) if content_html else []

    lines = [f"# {title or '微信公众号文章'}", ""]
    if account:
        lines.append(f"- 公众号: {account}")
    if author and author != account:
        lines.append(f"- 作者: {author}")
    if publish_time:
        lines.append(f"- 发布时间: {publish_time}")
    lines.append(f"- 原文链接: {article_url}")
    lines.extend(["", "## 正文", "", content_text or "(未提取到正文)", ""])
    if image_urls:
        lines.extend(["## 原文图片", ""])
        for idx, u in enumerate(image_urls, 1):
            lines.append(f"{idx}. {u}")
        lines.append("")

    meta_parts: List[str] = []
    if account:
        meta_parts.append(f"公众号: {account}")
    if author:
        meta_parts.append(f"作者: {author}")
    if publish_time:
        meta_parts.append(f"发布时间: {publish_time}")

    return {
        "title": title or "微信公众号文章",
        "desc": "；".join(meta_parts) if meta_parts else "微信公众号文章单篇正文",
        "markdown": "\n".join(lines),
    }


async def fetch_wechat_article_markdown(
    url: str,
    timeout_sec: int,
    last_fetch_info: Dict[str, Any],
    *,
    max_chars: int,
    user_prompt_template: str,
) -> Optional[Tuple[str, Optional[str], List[str]]]:
    """Fetch one WeChat article and build markdown snippet for prompt usage."""
    article_url = ensure_mobile_article_url(url)

    def _mark(
        *,
        status: Optional[int] = None,
        via: str = "",
        final_url: Optional[str] = None,
        error: Optional[str] = None,
        captcha: bool = False,
    ) -> None:
        last_fetch_info.clear()
        last_fetch_info.update(
            {
                "url": final_url or article_url,
                "status": status,
                "cloudflare": False,
                "via": via,
                "error": error,
                "wechat": True,
                "wechat_captcha": bool(captcha),
            }
        )

    async def _aiohttp_fetch() -> Optional[Tuple[str, str]]:
        if aiohttp is None:
            return None
        try:
            async with aiohttp.ClientSession(headers=_wechat_headers()) as session:
                async with session.get(
                    article_url, timeout=timeout_sec, allow_redirects=True
                ) as resp:
                    status = int(resp.status)
                    final_url = str(getattr(resp, "url", article_url))
                    if "wappoc_appmsgcaptcha" in final_url:
                        _mark(
                            status=status,
                            via="wechat_aiohttp",
                            final_url=final_url,
                            error="wechat_captcha",
                            captcha=True,
                        )
                        return None
                    if not (200 <= status < 400):
                        _mark(
                            status=status,
                            via="wechat_aiohttp",
                            final_url=final_url,
                            error=f"status={status}",
                        )
                        return None
                    text = await resp.text()
                    _mark(status=status, via="wechat_aiohttp", final_url=final_url)
                    return text, final_url
        except Exception as e:
            _mark(via="wechat_aiohttp", error=str(e))
            return None

    async def _urllib_fetch() -> Optional[Tuple[str, str]]:
        import urllib.request

        def _do() -> Optional[Tuple[str, str]]:
            try:
                req = urllib.request.Request(article_url, headers=_wechat_headers())
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    status = int(getattr(resp, "status", 200))
                    final_url = str(getattr(resp, "url", article_url))
                    if "wappoc_appmsgcaptcha" in final_url:
                        _mark(
                            status=status,
                            via="wechat_urllib",
                            final_url=final_url,
                            error="wechat_captcha",
                            captcha=True,
                        )
                        return None
                    if not (200 <= status < 400):
                        _mark(
                            status=status,
                            via="wechat_urllib",
                            final_url=final_url,
                            error=f"status={status}",
                        )
                        return None
                    body = resp.read()
                    enc = resp.headers.get_content_charset() or "utf-8"
                    text = body.decode(enc, errors="replace")
                    _mark(status=status, via="wechat_urllib", final_url=final_url)
                    return text, final_url
            except Exception as e:
                _mark(via="wechat_urllib", error=str(e))
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do)

    page = await _aiohttp_fetch()
    if page is None:
        page = await _urllib_fetch()
    if page is None:
        return None

    page_html, final_url = page
    parsed = _parse_markdown(final_url, page_html)
    md_text = parsed["markdown"]
    snippet = md_text[: max(0, int(max_chars))]
    user_prompt = user_prompt_template.format(
        url=final_url,
        title=parsed["title"],
        desc=parsed["desc"],
        snippet=snippet,
    )
    return (user_prompt, md_text, [])
