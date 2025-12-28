from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import asyncio
import os
import re
from html import unescape
from urllib.parse import quote, urljoin, urlparse

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger

from .file_preview_utils import pdf_bytes_to_markdown


def extract_urls_from_text(text: Optional[str]) -> List[str]:
    """从文本中提取 URL 列表，保持顺序去重。"""
    if not isinstance(text, str) or not text:
        return []
    url_pattern = re.compile(
        r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)", re.IGNORECASE
    )
    urls = [m.group(1) for m in url_pattern.finditer(text)]
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def strip_html(html: str) -> str:
    """基础 HTML 文本提取：去 script/style 与标签，归一空白。"""
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return unescape(re.sub(r"\s+", " ", m.group(1)).strip())
    return ""


def extract_meta_desc(html: str) -> str:
    for name in [
        r'name="description"',
        r'property="og:description"',
        r'name="twitter:description"',
    ]:
        m = re.search(
            rf"<meta[^>]+{name}[^>]+content=\"(.*?)\"[^>]*>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return unescape(re.sub(r"\s+", " ", m.group(1)).strip())
    return ""


def build_cf_screenshot_url(
    url: str,
    width: int,
    height: int,
) -> str:
    """构造 urlscan 截图 URL。"""
    try:
        encoded = quote(url, safe="")
    except Exception:
        encoded = url
    return f"https://urlscan.io/liveshot/?width={width}&height={height}&url={encoded}"


def extract_first_img_src(html: str) -> Optional[str]:
    if not isinstance(html, str) or not html:
        return None
    m = re.search(
        r'<img[^>]+src=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if m:
        return unescape(m.group(1).strip())
    return None


async def fetch_html(
    url: str, timeout_sec: int, last_fetch_info: Dict[str, Any]
) -> Optional[str]:
    """获取网页 HTML 文本并记录 Cloudflare 相关信息。"""

    def _mark(
        status: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        text_hint: Optional[str] = None,
        via: str = "",
        error: Optional[str] = None,
    ):
        headers = headers or {}
        server = str(headers.get("server", "")).lower()
        cf_header = (
            any(h.lower().startswith("cf-") for h in headers.keys())
            if headers
            else False
        )
        text_has_cf = False
        if isinstance(text_hint, str):
            tl = text_hint.lower()
            if (
                "cloudflare" in tl
                or "attention required" in tl
                or "enable javascript and cookies" in tl
            ):
                text_has_cf = True
        is_cf = ("cloudflare" in server) or cf_header or text_has_cf
        last_fetch_info.clear()
        last_fetch_info.update(
            {
                "url": url,
                "status": status,
                "cloudflare": is_cf,
                "via": via,
                "error": error,
            }
        )

    async def _aiohttp_fetch() -> Optional[str]:
        if aiohttp is None:
            return None
        try:
            async with aiohttp.ClientSession(
                headers={
                    "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)"
                }
            ) as session:
                async with session.get(
                    url, timeout=timeout_sec, allow_redirects=True
                ) as resp:
                    status = int(resp.status)
                    hdrs = {k: v for k, v in resp.headers.items()}
                    if 200 <= status < 400:
                        text = await resp.text()
                        _mark(
                            status=status,
                            headers=hdrs,
                            text_hint=text[:512],
                            via="aiohttp",
                        )
                        return text
                    _mark(status=status, headers=hdrs, text_hint=None, via="aiohttp")
                    return None
        except Exception as e:  # pragma: no cover - 网络环境相关
            logger.warning(f"zssm_explain: aiohttp fetch failed: {e}")
            _mark(
                status=None, headers=None, text_hint=None, via="aiohttp", error=str(e)
            )
            return None

    async def _urllib_fetch() -> Optional[str]:
        import urllib.request
        import urllib.error

        def _do() -> Optional[str]:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)",
                    },
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = resp.read()
                    enc = resp.headers.get_content_charset() or "utf-8"
                    try:
                        text = data.decode(enc, errors="replace")
                        _mark(
                            status=getattr(resp, "status", 200),
                            headers=dict(resp.headers),
                            text_hint=text[:512],
                            via="urllib",
                        )
                        return text
                    except Exception:
                        text = data.decode("utf-8", errors="replace")
                        _mark(
                            status=getattr(resp, "status", 200),
                            headers=dict(resp.headers),
                            text_hint=text[:512],
                            via="urllib",
                        )
                        return text
            except urllib.error.HTTPError as e:
                try:
                    body = e.read() or b""
                    hint = body.decode("utf-8", errors="ignore")[:512]
                except Exception:
                    hint = None
                hdrs = dict(getattr(e, "headers", {}) or {})
                _mark(
                    status=getattr(e, "code", None),
                    headers=hdrs,
                    text_hint=hint,
                    via="urllib",
                    error=str(e),
                )
                logger.warning(f"zssm_explain: urllib fetch failed: {e}")
                return None
            except Exception as e:  # pragma: no cover
                _mark(
                    status=None,
                    headers=None,
                    text_hint=None,
                    via="urllib",
                    error=str(e),
                )
                logger.warning(f"zssm_explain: urllib fetch failed: {e}")
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do)

    html = await _aiohttp_fetch()
    if html is not None:
        return html
    return await _urllib_fetch()


async def fetch_pdf_bytes(
    url: str, timeout_sec: int, max_bytes: int
) -> Optional[bytes]:
    """抓取 PDF 二进制内容（做体积限制），用于 URL 场景的 PDF 摘要。"""
    if not (isinstance(url, str) and url.strip()):
        return None
    timeout_sec = max(2, int(timeout_sec))
    max_bytes = max(1, int(max_bytes))
    headers = {
        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)"
    }

    async def _aiohttp_fetch() -> Optional[bytes]:
        if aiohttp is None:
            return None
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    url, timeout=timeout_sec, allow_redirects=True
                ) as resp:
                    status = int(resp.status)
                    if not (200 <= status < 400):
                        return None
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        logger.warning(
                            "zssm_explain: pdf over size limit when fetching url=%s",
                            url,
                        )
                        return None
                    data = await resp.content.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        logger.warning(
                            "zssm_explain: pdf over size limit when fetching url=%s",
                            url,
                        )
                        return None
                    return data
        except Exception as e:
            logger.warning(f"zssm_explain: aiohttp fetch pdf failed: {e}")
            return None

    data = await _aiohttp_fetch()
    if data is not None:
        return data

    import urllib.error
    import urllib.request

    def _do() -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                status = getattr(resp, "status", 200)
                if not (200 <= int(status) < 400):
                    return None
                chunks: List[bytes] = []
                remaining = max_bytes + 1
                while remaining > 0:
                    chunk = resp.read(min(8192, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                    if remaining <= 0:
                        logger.warning(
                            "zssm_explain: pdf over size limit in urllib fetch url=%s",
                            url,
                        )
                        return None
                return b"".join(chunks)
        except urllib.error.HTTPError as e:
            logger.warning(f"zssm_explain: urllib fetch pdf failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"zssm_explain: urllib fetch pdf failed: {e}")
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)


def build_url_user_prompt(
    url: str,
    html: str,
    max_chars: int,
    user_prompt_template: str,
) -> Tuple[str, str]:
    title = extract_title(html)
    desc = extract_meta_desc(html)
    plain = strip_html(html)
    snippet = plain[: max(0, int(max_chars))]
    user_prompt = user_prompt_template.format(
        url=url,
        title=title or "(无)",
        desc=desc or "(无)",
        snippet=snippet,
    )
    return user_prompt, title or ""


def build_url_brief_for_forward(html: str, max_chars: int) -> Tuple[str, str, str]:
    """为合并转发场景构造网址的精简信息摘要（标题/描述/正文片段）。"""
    title = extract_title(html)
    desc = extract_meta_desc(html)
    plain = strip_html(html)
    snippet = plain[: max(0, int(max_chars))]
    return title or "", desc or "", snippet


async def prepare_url_prompt(
    url: str,
    timeout_sec: int,
    last_fetch_info: Dict[str, Any],
    *,
    max_chars: int,
    cf_screenshot_enable: bool,
    cf_screenshot_width: int,
    cf_screenshot_height: int,
    file_preview_max_bytes: int,
    user_prompt_template: str,
) -> Optional[Tuple[str, Optional[str], List[str]]]:
    """统一处理网页抓取：成功返回摘要提示词；若因 Cloudflare 被拦截则回退到截图模式。

    返回值：
    - user_prompt: 给 LLM 的用户提示词
    - text: 当前实现不返回正文（保持与旧逻辑一致，返回 None）
    - images: 可选的本地截图路径（Cloudflare 降级）
    """
    # 1) 特判 PDF 链接：直接按 PDF 处理并生成 Markdown 片段
    try:
        path = urlparse(url).path
        _, ext = os.path.splitext(str(path).lower())
    except Exception:
        ext = ""

    if ext == ".pdf":
        max_bytes = (
            int(file_preview_max_bytes)
            if isinstance(file_preview_max_bytes, int) and file_preview_max_bytes > 0
            else 2 * 1024 * 1024
        )
        pdf_bytes = await fetch_pdf_bytes(url, timeout_sec, max_bytes)
        if pdf_bytes:
            text = pdf_bytes_to_markdown(pdf_bytes)
            if text:
                snippet = text[: max(0, int(max_chars))]
                user_prompt = user_prompt_template.format(
                    url=url,
                    title="(PDF 文档)",
                    desc="从 PDF 正文中提取的文本内容。",
                    snippet=snippet,
                )
                return (user_prompt, None, [])

    # 2) 常规 HTML 场景
    html = await fetch_html(url, timeout_sec, last_fetch_info)
    if html:
        user_prompt, _title = build_url_user_prompt(
            url, html, max_chars, user_prompt_template
        )
        return (user_prompt, None, [])

    # 3) Cloudflare 截图降级
    info = last_fetch_info or {}
    is_cf = bool(info.get("cloudflare"))
    if is_cf and cf_screenshot_enable:
        screenshot_url = build_cf_screenshot_url(
            url, int(cf_screenshot_width), int(cf_screenshot_height)
        )
        if screenshot_url:
            logger.warning(
                "zssm_explain: Cloudflare detected for %s (status=%s, via=%s); fallback to urlscan screenshot",
                url,
                info.get("status"),
                info.get("via"),
            )
            ready = await wait_cf_screenshot_ready(screenshot_url, last_fetch_info)
            if not ready:
                try:
                    last_fetch_info["cf_screenshot_ready"] = False
                except Exception:
                    pass
                logger.warning(
                    "zssm_explain: urlscan screenshot still unavailable, aborting image fallback"
                )
                return None
            try:
                last_fetch_info["used_cf_screenshot"] = True
                last_fetch_info["cf_screenshot_ready"] = True
            except Exception:
                pass

            final_image_url = await resolve_liveshot_image_url(screenshot_url)
            if not final_image_url:
                logger.warning(
                    "zssm_explain: failed to resolve liveshot image from html response"
                )
                return None
            local_image_path = await download_image_to_temp(final_image_url)
            if not local_image_path:
                logger.warning(
                    "zssm_explain: failed to download liveshot image to temp file"
                )
                return None
            cf_prompt = user_prompt_template.format(
                url=url,
                title="(Cloudflare 截图)",
                desc="目标站点启用 Cloudflare，已改用 urlscan 截图作为依据。",
                snippet="由于无法直接抓取 HTML，请结合截图内容输出网页摘要。",
            )
            return (cf_prompt, None, [local_image_path])

    return None


def build_url_failure_message(
    last_fetch_info: Dict[str, Any], cf_screenshot_enable: bool
) -> str:
    info = last_fetch_info or {}
    if info.get("cloudflare"):
        if cf_screenshot_enable:
            return "目标站点启用 Cloudflare 防护，截图降级失败，请稍后重试或改为发送手动截图/摘录内容。"
        return "目标站点启用 Cloudflare 防护，因未启用截图降级无法抓取，请开启 cf_screenshot_enable 或稍后再试。"
    return "网页获取失败或不受支持，请稍后重试并确认链接可访问。"


async def probe_screenshot_url(url: str, per_request_timeout: int = 6) -> bool:
    """尝试访问截图 URL，确认资源已经生成。"""
    if not url:
        return False
    headers = {
        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)",
        "Range": "bytes=0-256",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }
    if aiohttp is not None:
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    url, timeout=per_request_timeout, allow_redirects=True
                ) as resp:
                    if 200 <= int(resp.status) < 400:
                        await resp.content.readexactly(1)
                        return True
        except Exception:
            pass
    import urllib.request
    import urllib.error

    def _do() -> bool:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=per_request_timeout) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= int(status) < 400:
                    resp.read(1)
                    return True
        except Exception:
            return False
        return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)


async def wait_cf_screenshot_ready(
    url: str,
    last_fetch_info: Dict[str, Any],
    overall_timeout: float = 12.0,
    interval_sec: float = 1.5,
) -> bool:
    """轮询 urlscan 截图是否已经可访问。"""
    if not url:
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(overall_timeout, 3.0)
    attempt = 0
    while True:
        attempt += 1
        if await probe_screenshot_url(url):
            try:
                last_fetch_info["cf_screenshot_ready_attempts"] = attempt
            except Exception:
                pass
            return True
        if loop.time() >= deadline:
            logger.warning(
                "zssm_explain: urlscan screenshot not ready after %s attempts", attempt
            )
            break
        await asyncio.sleep(interval_sec)
    return False


async def download_image_to_temp(url: str, timeout_sec: int = 15) -> Optional[str]:
    """下载图片到临时文件并返回路径。"""
    if not url:
        return None
    headers = {
        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }

    async def _fetch() -> Tuple[Optional[bytes], Optional[str]]:
        if aiohttp is not None:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(
                        url, timeout=timeout_sec, allow_redirects=True
                    ) as resp:
                        if 200 <= int(resp.status) < 400:
                            data = await resp.read()
                            return data, resp.headers.get("Content-Type")
            except Exception:
                pass
        import urllib.request
        import urllib.error

        def _do() -> Tuple[Optional[bytes], Optional[str]]:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    status = getattr(resp, "status", 200)
                    if 200 <= int(status) < 400:
                        data = resp.read()
                        return data, resp.headers.get("Content-Type")
            except Exception:
                return (None, None)
            return (None, None)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do)

    data, content_type = await _fetch()
    if not data:
        return None
    suffix = ".png"
    if isinstance(content_type, str):
        cl = content_type.lower()
        if "jpeg" in cl:
            suffix = ".jpg"
        elif "webp" in cl:
            suffix = ".webp"
    try:
        import tempfile

        fd, path = tempfile.mkstemp(prefix="zssm_cf_", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        logger.warning(f"zssm_explain: failed to save screenshot temp file: {e}")
        return None


async def resolve_liveshot_image_url(url: str, timeout_sec: int = 15) -> Optional[str]:
    """确保拿到真正的图片 URL：若返回 HTML，则解析 <img src>。"""
    headers = {
        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }

    async def _fetch() -> Tuple[Optional[bytes], Optional[str]]:
        if aiohttp is not None:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(
                        url, timeout=timeout_sec, allow_redirects=True
                    ) as resp:
                        if 200 <= int(resp.status) < 400:
                            data = await resp.read()
                            return data, resp.headers.get("Content-Type")
            except Exception:
                pass
        import urllib.request
        import urllib.error

        def _do() -> Tuple[Optional[bytes], Optional[str]]:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    status = getattr(resp, "status", 200)
                    if 200 <= int(status) < 400:
                        data = resp.read()
                        return data, resp.headers.get("Content-Type")
            except Exception:
                return (None, None)
            return (None, None)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do)

    data, content_type = await _fetch()
    if not data:
        return None
    if isinstance(content_type, str) and "image" in content_type.lower():
        return url
    try:
        html = data.decode("utf-8", errors="ignore")
    except Exception:
        html = ""
    img_src = extract_first_img_src(html)
    if not img_src:
        return None
    resolved = urljoin(url, img_src)
    if not resolved.startswith("http"):
        return None
    ok = await probe_screenshot_url(resolved)
    return resolved if ok else None
