from __future__ import annotations

from typing import Optional, Dict, Any

import json
import re
from urllib.parse import urlparse

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger

from .video_utils import is_http_url, download_video_to_temp


_BILI_B23_RE = re.compile(r"(https?://)?(b23\.tv|bili2233\.cn)/[\w]+", re.IGNORECASE)
_BILI_BV_RE = re.compile(r"BV1[0-9A-Za-z]{9}")
_BILI_AV_RE = re.compile(r"av\d+", re.IGNORECASE)

_BILI_AV2BV_TABLE = "fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF"
_BILI_AV2BV_S = [11, 10, 3, 8, 4, 6]
_BILI_AV2BV_XOR = 177451812
_BILI_AV2BV_ADD = 8728348608

_BILI_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AstrBot-zssm/1.0"
_BILI_REFERER = "https://www.bilibili.com/"


def is_bilibili_url(url: Optional[str]) -> bool:
    """判断是否为看起来是 B 站相关的 URL（含短链）。"""
    if not is_http_url(url):
        return False
    try:
        parsed = urlparse(str(url))
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    if not host:
        return False
    return (
        "bilibili.com" in host
        or "b23.tv" in host
        or "bili2233.cn" in host
    )


def _bili_av2bv(av: str) -> Optional[str]:
    """将 av 号字符串转换为 BV 号。"""
    m = re.search(r"\d+", av)
    if not m:
        return None
    try:
        x = (int(m.group()) ^ _BILI_AV2BV_XOR) + _BILI_AV2BV_ADD
    except Exception:
        return None
    r = list("BV1 0 4 1 7  ")
    for i in range(6):
        idx = (x // (58 ** i)) % 58
        r[_BILI_AV2BV_S[i]] = _BILI_AV2BV_TABLE[idx]
    return "".join(r).replace(" ", "0")


async def _bili_resolve_b23(url: str) -> Optional[str]:
    """解析 b23.tv / bili2233.cn 等短链，获取真实跳转后的 URL。"""
    if not isinstance(url, str) or not url:
        return None
    full = url.strip()
    if not full.lower().startswith(("http://", "https://")):
        full = "https://" + full.lstrip("/")
    if aiohttp is not None:
        try:
            async with aiohttp.ClientSession(headers={"User-Agent": _BILI_UA}) as session:
                async with session.get(full, timeout=15, allow_redirects=True) as resp:
                    return str(resp.url)
        except Exception:
            pass
    try:
        import urllib.request

        req = urllib.request.Request(full, headers={"User-Agent": _BILI_UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            try:
                return resp.geturl()
            except Exception:
                return full
    except Exception:
        return None


def _bili_extract_bvid_from_url(url: str) -> Optional[str]:
    """从各种形式的 B 站链接中抽取 BV 号（或 av 号并转换）。"""
    if not isinstance(url, str):
        return None
    m_bv = _BILI_BV_RE.search(url)
    if m_bv:
        return m_bv.group(0)
    m_av = _BILI_AV_RE.search(url)
    if m_av:
        return _bili_av2bv(m_av.group(0))
    return None


async def _bili_request_json(url: str) -> Optional[Dict[str, Any]]:
    """以 JSON 形式请求 B 站 API，带简单 UA/Referer。"""
    headers = {
        "User-Agent": _BILI_UA,
        "Referer": _BILI_REFERER,
    }
    if aiohttp is not None:
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if 200 <= int(resp.status) < 400:
                        try:
                            data = await resp.json()
                        except Exception:
                            text = await resp.text()
                            try:
                                data = json.loads(text)
                            except Exception:
                                return None
                        return data if isinstance(data, dict) else None
        except Exception:
            pass
    try:
        import urllib.request

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            try:
                data = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                return None
            return data if isinstance(data, dict) else None
    except Exception:
        return None


async def resolve_bilibili_video_url(url: str, quality: int = 80) -> Optional[str]:
    """将 B 站页面/短链解析为可下载的视频直链 URL。

    仅依赖公开 API，不涉及登录态；解析失败时返回 None。
    """
    if not is_bilibili_url(url):
        return None

    candidate = url
    # 短链先展开
    if _BILI_B23_RE.search(candidate or ""):
        real = await _bili_resolve_b23(candidate)
        if isinstance(real, str) and real:
            candidate = real

    bvid = _bili_extract_bvid_from_url(candidate)
    if not bvid:
        return None

    view_api = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    view_data = await _bili_request_json(view_api)
    if not (isinstance(view_data, dict) and view_data.get("code") == 0 and isinstance(view_data.get("data"), dict)):
        return None
    info = view_data["data"]
    aid = info.get("aid")
    cid = info.get("cid")
    if aid is None or cid is None:
        return None

    play_api = (
        f"https://api.bilibili.com/x/player/playurl?"
        f"avid={aid}&cid={cid}&qn={quality}&type=mp4&platform=html5"
    )
    play_data = await _bili_request_json(play_api)
    if not (isinstance(play_data, dict) and play_data.get("code") == 0):
        return None
    pdata = play_data.get("data") or {}
    durl = pdata.get("durl")
    if isinstance(durl, list) and durl:
        first = durl[0]
        if isinstance(first, dict):
            v_url = first.get("url")
            if isinstance(v_url, str) and v_url:
                return v_url
    return None


async def download_bilibili_video_to_temp(url: str, size_mb_limit: int, quality: int = 80) -> Optional[str]:
    """解析 B 站视频链接并下载到临时文件。

    - 先通过 resolve_bilibili_video_url 获取真实文件地址；
    - 再附带 B 站 UA/Referer 头下载到临时文件；
    - 超过 size_mb_limit 或下载失败时返回 None。
    """
    stream_url = await resolve_bilibili_video_url(url, quality=quality)
    if not isinstance(stream_url, str) or not stream_url:
        return None
    headers = {
        "User-Agent": _BILI_UA,
        "Referer": _BILI_REFERER,
    }
    logger.info("zssm_explain: downloading bilibili stream url=%s", stream_url[:120])
    return await download_video_to_temp(stream_url, size_mb_limit, headers=headers)


__all__ = [
    "is_bilibili_url",
    "resolve_bilibili_video_url",
    "download_bilibili_video_to_temp",
]
