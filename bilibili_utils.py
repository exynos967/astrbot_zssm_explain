from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import json
import re
from urllib.parse import urlparse, parse_qs

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger

from .video_utils import is_http_url, download_video_to_temp


_BILI_B23_RE = re.compile(r"(https?://)?(b23\.tv|bili2233\.cn)/[\w]+", re.IGNORECASE)
_BILI_BV_RE = re.compile(r"BV[0-9a-zA-Z]{10}")
_BILI_AV_RE = re.compile(r"av(\d+)", re.IGNORECASE)

# 动态 URL 模式
_BILI_DYNAMIC_RE = re.compile(
    r"(?:t\.bilibili\.com|bilibili\.com/dynamic)/(\d+)", re.IGNORECASE
)
# 直播 URL 模式
_BILI_LIVE_RE = re.compile(r"live\.bilibili\.com/(\d+)", re.IGNORECASE)
# 专栏 URL 模式
_BILI_READ_RE = re.compile(r"bilibili\.com/read/cv(\d+)", re.IGNORECASE)
# 图文动态 URL 模式
_BILI_OPUS_RE = re.compile(r"bilibili\.com/opus/(\d+)", re.IGNORECASE)
# 收藏夹 URL 模式
_BILI_FAVLIST_RE = re.compile(r"bilibili\.com/favlist\?fid=(\d+)", re.IGNORECASE)

_BILI_AV2BV_TABLE = "fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF"
_BILI_AV2BV_S = [11, 10, 3, 8, 4, 6]
_BILI_AV2BV_XOR = 177451812
_BILI_AV2BV_ADD = 8728348608

_BILI_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AstrBot-zssm/1.0"
_BILI_REFERER = "https://www.bilibili.com/"


# 允许的 B 站域名白名单（精确匹配或后缀匹配）
_BILI_ALLOWED_HOSTS = frozenset(
    [
        "bilibili.com",
        "www.bilibili.com",
        "m.bilibili.com",
        "t.bilibili.com",
        "live.bilibili.com",
        "space.bilibili.com",
        "bilibilix.com",
        "www.bilibilix.com",
        "b23.tv",
        "bili2233.cn",
    ]
)


def _is_valid_bili_host(host: str) -> bool:
    """严格校验是否为合法的 B 站域名，防止 SSRF 绕过。

    处理顺序：先去掉 userinfo（@前的部分），再去掉端口，最后校验域名。
    防止 bilibili.com:80@evil.com 这类绕过。
    """
    if not host:
        return False
    h = host.lower()
    # 1. 先去掉 userinfo 部分（防止 user:pass@host 或 bilibili.com@evil.com 绕过）
    if "@" in h:
        h = h.split("@")[-1]
    # 2. 再去掉端口
    if ":" in h:
        h = h.rsplit(":", 1)[0]
    # 3. 去掉尾部点号（DNS 规范允许尾点）
    h = h.rstrip(".")
    # 4. 拒绝 IP 地址（防止 127.0.0.1 等）
    if h.replace(".", "").isdigit():
        return False
    if h.startswith("["):  # IPv6
        return False
    # 精确匹配
    if h in _BILI_ALLOWED_HOSTS:
        return True
    # 后缀匹配（子域名）
    for allowed in _BILI_ALLOWED_HOSTS:
        if h.endswith("." + allowed):
            return True
    return False


def is_bilibili_url(url: Optional[str]) -> bool:
    """判断是否为 B 站相关的 URL（含短链、bilibilix 直链解析）。"""
    if not is_http_url(url):
        return False
    try:
        parsed = urlparse(str(url))
    except Exception:
        return False
    host = parsed.netloc or ""
    return _is_valid_bili_host(host)


def get_bilibili_url_type(url: Optional[str]) -> Optional[str]:
    """判断 B 站 URL 类型。

    返回值：
    - "video": 视频（BV/av）
    - "dynamic": 动态
    - "live": 直播
    - "read": 专栏
    - "opus": 图文动态
    - "short": 短链（需展开后再判断）
    - None: 非 B 站链接或暂不支持的类型（如收藏夹）
    """
    if not is_bilibili_url(url):
        return None
    s = str(url)
    if _BILI_B23_RE.search(s):
        return "short"
    if _BILI_LIVE_RE.search(s):
        return "live"
    if _BILI_DYNAMIC_RE.search(s):
        return "dynamic"
    if _BILI_READ_RE.search(s):
        return "read"
    if _BILI_OPUS_RE.search(s):
        return "opus"
    # 收藏夹暂不支持，返回 None 让调用方走截图降级
    if _BILI_FAVLIST_RE.search(s):
        return None
    if _BILI_BV_RE.search(s) or _BILI_AV_RE.search(s):
        return "video"
    # bilibilix.com 视频直链
    if "bilibilix.com" in s.lower():
        return "video"
    return None


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
        idx = (x // (58**i)) % 58
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
            async with aiohttp.ClientSession(
                headers={"User-Agent": _BILI_UA}
            ) as session:
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


def _bili_extract_bvid_from_url(url: str) -> Tuple[Optional[str], int]:
    """从各种形式的 B 站链接中抽取 BV 号（或 av 号并转换），以及分P页码。

    返回 (bvid, page_index)，page_index 从 0 开始。
    """
    if not isinstance(url, str):
        return None, 0
    # 解析 ?p= 参数
    page_index = 0
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        p = qs.get("p", [None])[0]
        if p and str(p).isdigit() and int(p) >= 1:
            page_index = int(p) - 1
    except Exception:
        pass

    m_bv = _BILI_BV_RE.search(url)
    if m_bv:
        return m_bv.group(0), page_index
    m_av = _BILI_AV_RE.search(url)
    if m_av:
        return _bili_av2bv(m_av.group(0)), page_index
    return None, 0


async def _bili_request_json(
    url: str, cookie: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """以 JSON 形式请求 B 站 API，带简单 UA/Referer。

    Args:
        url: API 地址
        cookie: 可选的 Cookie 字符串，格式如 "SESSDATA=xxx; bili_jct=xxx"

    安全说明：
        - 带 Cookie 时禁用自动重定向，防止 Cookie 泄露到非 B 站域名
        - 无 Cookie 时允许重定向（API 通常不重定向）
    """
    headers = {
        "User-Agent": _BILI_UA,
        "Referer": _BILI_REFERER,
    }
    if cookie:
        headers["Cookie"] = cookie
    # 带 Cookie 时禁用重定向，防止 Cookie 泄露
    allow_redir = not bool(cookie)
    if aiohttp is not None:
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get(url, allow_redirects=allow_redir) as resp:
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

    支持：
    - bilibili.com/video/BVxxx 或 avxxx（含 ?p= 分P）
    - b23.tv / bili2233.cn 短链
    - bilibilix.com 直链解析服务

    仅依赖公开 API，不涉及登录态；解析失败时返回 None。
    """
    if not is_bilibili_url(url):
        return None

    candidate = url

    # bilibilix.com 链接：直接跟随重定向拿到视频直链
    try:
        parsed = urlparse(candidate)
        if "bilibilix.com" in (parsed.netloc or "").lower():
            direct = await _bili_fetch_bilibilix(candidate)
            if direct:
                return direct
            # bilibilix 失败，尝试转换为 bilibili.com 走常规解析
            candidate = candidate.replace("bilibilix.com", "bilibili.com")
    except Exception:
        pass

    # 短链先展开
    if _BILI_B23_RE.search(candidate or ""):
        real = await _bili_resolve_b23(candidate)
        if isinstance(real, str) and real:
            candidate = real

    bvid, page_index = _bili_extract_bvid_from_url(candidate)
    if not bvid:
        return None

    # 先尝试 bilibilix.com 直链解析（更快、更稳定）
    bilibilix_url = f"https://www.bilibilix.com/video/{bvid}"
    if page_index > 0:
        bilibilix_url += f"?p={page_index + 1}"
    direct = await _bili_fetch_bilibilix(bilibilix_url)
    if direct:
        logger.info("zssm_explain: bilibilix resolved => %s", direct[:120])
        return direct

    # bilibilix 失败，回退到官方 API
    view_api = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    view_data = await _bili_request_json(view_api)
    if not (
        isinstance(view_data, dict)
        and view_data.get("code") == 0
        and isinstance(view_data.get("data"), dict)
    ):
        return None
    info = view_data["data"]
    aid = info.get("aid")
    # 分P：从 pages 列表取对应 cid
    cid = None
    pages = info.get("pages")
    if isinstance(pages, list) and page_index < len(pages):
        cid = pages[page_index].get("cid")
    if cid is None:
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


# B 站 CDN 域名白名单（用于校验重定向目标）
_BILI_CDN_HOSTS = frozenset(
    [
        "bilivideo.com",
        "bilivideo.cn",
        "akamaized.net",
        "bilicdn1.com",
        "hdslb.com",
    ]
)


def _is_valid_bili_cdn_host(url: str) -> bool:
    """校验 URL 是否为合法的 B 站 CDN 域名。"""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if "@" in host:
            host = host.split("@")[-1]
        if ":" in host:
            host = host.rsplit(":", 1)[0]
        host = host.rstrip(".")
        for cdn in _BILI_CDN_HOSTS:
            if host == cdn or host.endswith("." + cdn):
                return True
    except Exception:
        pass
    return False


async def _bili_fetch_bilibilix(url: str) -> Optional[str]:
    """通过 bilibilix.com 获取视频直链（跟随重定向）。"""
    if aiohttp is not None:
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": _BILI_UA},
            ) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    final_url = str(resp.url)
                    # 校验重定向目标是否为合法 CDN 域名
                    if _is_valid_bili_cdn_host(final_url):
                        return final_url
                    # 检查响应头 Location
                    loc = resp.headers.get("Location")
                    if isinstance(loc, str) and _is_valid_bili_cdn_host(loc):
                        return loc
        except Exception as e:
            logger.debug("zssm_explain: bilibilix fetch failed: %s", e)
    return None


async def download_bilibili_video_to_temp(
    url: str, size_mb_limit: int, quality: int = 80
) -> Optional[str]:
    """解析 B 站视频链接并下载到临时文件。

    - 先通过 resolve_bilibili_video_url 获取真实文件地址；
    - 再附带 B 站 UA/Referer 头下载到临时文件；
    - 超过 size_mb_limit 时抛出 ValueError；
    - 解析/下载失败时返回 None。
    """
    stream_url = await resolve_bilibili_video_url(url, quality=quality)
    if not isinstance(stream_url, str) or not stream_url:
        return None
    headers = {
        "User-Agent": _BILI_UA,
        "Referer": _BILI_REFERER,
    }
    logger.info("zssm_explain: downloading bilibili stream url=%s", stream_url[:120])
    # B 站视频下载超时设为 180 秒（视频文件较大）
    result = await download_video_to_temp(
        stream_url, size_mb_limit, headers=headers, timeout_sec=180
    )
    # download_video_to_temp 返回 None 可能是大小超限或下载失败
    # 通过检查 Content-Length 区分
    if result is None:
        # 尝试 HEAD 请求检查大小
        try:
            if aiohttp is not None:
                async with aiohttp.ClientSession() as sess:
                    async with sess.head(
                        stream_url, headers=headers, timeout=10
                    ) as resp:
                        cl = resp.headers.get("Content-Length")
                        if cl and cl.isdigit():
                            size_mb = int(cl) / (1024 * 1024)
                            if size_mb > size_mb_limit:
                                raise ValueError(
                                    f"视频大小 {size_mb:.1f}MB 超过限制 {size_mb_limit}MB"
                                )
        except ValueError:
            raise
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# 动态解析
# ---------------------------------------------------------------------------


def _parse_dynamic_detail_response(
    data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """解析动态详情 API 响应，提取结构化信息。

    用于 dynamic 和 opus 类型，它们共用同一个 API。

    返回字典包含：
    - title: 标题（可能为空）
    - text: 文本内容
    - images: 图片 URL 列表
    - author: 作者名
    - avatar: 作者头像
    - timestamp: 发布时间戳
    """
    if not (isinstance(data, dict) and data.get("code") == 0):
        return None

    item = (data.get("data") or {}).get("item")
    if not isinstance(item, dict):
        return None

    modules = item.get("modules") or {}
    author_mod = modules.get("module_author") or {}
    dynamic_mod = modules.get("module_dynamic") or {}

    # 提取图片
    images: list[str] = []
    major = dynamic_mod.get("major") or {}
    major_type = major.get("type", "")

    if major_type == "MAJOR_TYPE_OPUS":
        opus = major.get("opus") or {}
        pics = opus.get("pics") or []
        images = [p.get("url") for p in pics if isinstance(p, dict) and p.get("url")]
    elif major_type == "MAJOR_TYPE_DRAW":
        draw = major.get("draw") or {}
        items = draw.get("items") or []
        images = [p.get("src") for p in items if isinstance(p, dict) and p.get("src")]
    elif major_type == "MAJOR_TYPE_ARCHIVE":
        archive = major.get("archive") or {}
        cover = archive.get("cover")
        if cover:
            images = [cover]

    # 提取文本
    text = ""
    desc = dynamic_mod.get("desc") or {}
    text = desc.get("text", "")
    if not text and major_type == "MAJOR_TYPE_OPUS":
        opus = major.get("opus") or {}
        summary = opus.get("summary") or {}
        text = summary.get("text", "")

    return {
        "title": None,
        "text": text,
        "images": images,
        "author": author_mod.get("name", ""),
        "avatar": author_mod.get("face", ""),
        "timestamp": author_mod.get("pub_ts", 0),
    }


async def resolve_bilibili_dynamic(
    url: str, cookie: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """解析 B 站动态，返回结构化信息。

    返回字典包含：
    - title: 标题（可能为空）
    - text: 文本内容
    - images: 图片 URL 列表
    - author: 作者名
    - avatar: 作者头像
    - timestamp: 发布时间戳

    Args:
        url: B 站动态链接
        cookie: 可选的 Cookie 字符串，格式如 "SESSDATA=xxx; bili_jct=xxx"
    """
    m = _BILI_DYNAMIC_RE.search(url or "")
    if not m:
        # 可能是短链，先展开
        if _BILI_B23_RE.search(url or ""):
            real = await _bili_resolve_b23(url)
            if real:
                m = _BILI_DYNAMIC_RE.search(real)
        if not m:
            return None

    dynamic_id = m.group(1)
    api = f"https://api.bilibili.com/x/polymer/web-dynamic/v1/detail?id={dynamic_id}"
    data = await _bili_request_json(api, cookie=cookie)
    return _parse_dynamic_detail_response(data)


# ---------------------------------------------------------------------------
# 直播解析
# ---------------------------------------------------------------------------


async def resolve_bilibili_live(url: str) -> Optional[Dict[str, Any]]:
    """解析 B 站直播间信息。

    返回字典包含：
    - title: 直播标题
    - text: 分区/标签信息
    - images: [封面, 关键帧]
    - author: 主播名
    - avatar: 主播头像
    - live_status: 直播状态 (0=未开播, 1=直播中, 2=轮播中)
    """
    m = _BILI_LIVE_RE.search(url or "")
    if not m:
        if _BILI_B23_RE.search(url or ""):
            real = await _bili_resolve_b23(url)
            if real:
                m = _BILI_LIVE_RE.search(real)
        if not m:
            return None

    room_id = m.group(1)
    api = f"https://api.live.bilibili.com/room/v1/Room/get_info?room_id={room_id}"
    data = await _bili_request_json(api)
    if not (isinstance(data, dict) and data.get("code") == 0):
        return None

    info = data.get("data") or {}
    uid = info.get("uid")

    # 获取主播信息
    author = ""
    avatar = ""
    if uid:
        user_api = f"https://api.live.bilibili.com/live_user/v1/Master/info?uid={uid}"
        user_data = await _bili_request_json(user_api)
        if isinstance(user_data, dict) and user_data.get("code") == 0:
            user_info = (user_data.get("data") or {}).get("info") or {}
            author = user_info.get("uname", "")
            avatar = user_info.get("face", "")

    images = []
    if info.get("user_cover"):
        images.append(info["user_cover"])
    if info.get("keyframe"):
        images.append(info["keyframe"])

    area = info.get("area_name", "")
    parent_area = info.get("parent_area_name", "")
    tags = info.get("tags", "")
    text_parts = []
    if parent_area or area:
        text_parts.append(f"分区: {parent_area} - {area}")
    if tags:
        text_parts.append(f"标签: {tags}")

    return {
        "title": f"直播 - {info.get('title', '')}",
        "text": "\n".join(text_parts),
        "images": images,
        "author": author,
        "avatar": avatar,
        "live_status": info.get("live_status", 0),
    }


# ---------------------------------------------------------------------------
# 专栏/图文解析
# ---------------------------------------------------------------------------


async def resolve_bilibili_read(url: str) -> Optional[Dict[str, Any]]:
    """解析 B 站专栏文章。

    返回字典包含：
    - title: 文章标题
    - text: 文章摘要/正文
    - images: 文章中的图片 URL 列表
    - author: 作者名
    - avatar: 作者头像
    - timestamp: 发布时间戳
    """
    m = _BILI_READ_RE.search(url or "")
    if not m:
        if _BILI_B23_RE.search(url or ""):
            real = await _bili_resolve_b23(url)
            if real:
                m = _BILI_READ_RE.search(real)
        if not m:
            return None

    cv_id = m.group(1)
    api = f"https://api.bilibili.com/x/article/viewinfo?id={cv_id}"
    data = await _bili_request_json(api)
    if not (isinstance(data, dict) and data.get("code") == 0):
        return None

    info = data.get("data") or {}
    images = info.get("image_urls") or []
    if not images and info.get("banner_url"):
        images = [info["banner_url"]]

    return {
        "title": info.get("title", ""),
        "text": info.get("summary", ""),
        "images": images,
        "author": info.get("author_name", ""),
        "avatar": info.get("author_face", ""),
        "timestamp": info.get("publish_time", 0),
    }


async def resolve_bilibili_opus(
    url: str, cookie: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """解析 B 站图文动态 (opus)。

    返回字典包含：
    - title: 标题（可能为空）
    - text: 文本内容
    - images: 图片 URL 列表
    - author: 作者名
    - avatar: 作者头像
    - timestamp: 发布时间戳

    Args:
        url: B 站 opus 链接
        cookie: 可选的 Cookie 字符串，格式如 "SESSDATA=xxx; bili_jct=xxx"
    """
    m = _BILI_OPUS_RE.search(url or "")
    if not m:
        if _BILI_B23_RE.search(url or ""):
            real = await _bili_resolve_b23(url)
            if real:
                m = _BILI_OPUS_RE.search(real)
        if not m:
            return None

    opus_id = m.group(1)
    # opus 使用动态 API，直接用 opus_id 作为 dynamic_id 查询
    api = f"https://api.bilibili.com/x/polymer/web-dynamic/v1/detail?id={opus_id}"
    data = await _bili_request_json(api, cookie=cookie)
    result = _parse_dynamic_detail_response(data)
    if not result:
        logger.debug(
            "zssm_explain: opus API failed, code=%s", data.get("code") if data else None
        )
    return result


# ---------------------------------------------------------------------------
# 视频信息（不下载，仅获取元数据）
# ---------------------------------------------------------------------------


async def resolve_bilibili_video_info(url: str) -> Optional[Dict[str, Any]]:
    """获取 B 站视频元信息（不下载视频）。

    返回字典包含：
    - title: 视频标题
    - text: 视频简介
    - images: [封面图]
    - author: UP主名
    - avatar: UP主头像
    - timestamp: 发布时间戳
    - duration: 时长（秒）
    - bvid: BV号
    - stats: 统计信息 {view, like, coin, favorite, share, reply, danmaku}
    """
    candidate = url
    if _BILI_B23_RE.search(url or ""):
        real = await _bili_resolve_b23(url)
        if real:
            candidate = real

    bvid, page_index = _bili_extract_bvid_from_url(candidate)
    if not bvid:
        return None

    api = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    data = await _bili_request_json(api)
    if not (isinstance(data, dict) and data.get("code") == 0):
        return None

    info = data.get("data") or {}
    owner = info.get("owner") or {}
    stat = info.get("stat") or {}

    # 处理分P
    title = info.get("title", "")
    duration = info.get("duration", 0)
    cover = info.get("pic", "")
    pages = info.get("pages") or []
    if pages and page_index < len(pages):
        page = pages[page_index]
        if len(pages) > 1:
            title = f"{title} | P{page_index + 1} - {page.get('part', '')}"
        duration = page.get("duration", duration)
        if page.get("first_frame"):
            cover = page["first_frame"]

    return {
        "title": title,
        "text": info.get("desc", ""),
        "images": [cover] if cover else [],
        "author": owner.get("name", ""),
        "avatar": owner.get("face", ""),
        "timestamp": info.get("pubdate", 0),
        "duration": duration,
        "bvid": bvid,
        "stats": {
            "view": stat.get("view", 0),
            "like": stat.get("like", 0),
            "coin": stat.get("coin", 0),
            "favorite": stat.get("favorite", 0),
            "share": stat.get("share", 0),
            "reply": stat.get("reply", 0),
            "danmaku": stat.get("danmaku", 0),
        },
    }


# ---------------------------------------------------------------------------
# 统一解析入口
# ---------------------------------------------------------------------------


async def resolve_bilibili_content(
    url: str, cookie: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """统一解析 B 站链接，自动识别类型并返回结构化内容。

    返回字典包含：
    - type: 内容类型 (video/dynamic/live/read/opus)
    - title: 标题
    - text: 文本内容
    - images: 图片 URL 列表
    - author: 作者名
    - avatar: 作者头像
    - timestamp: 时间戳
    - 其他类型特有字段

    Args:
        url: B 站链接
        cookie: 可选的 Cookie 字符串，格式如 "SESSDATA=xxx; bili_jct=xxx"
    """
    url_type = get_bilibili_url_type(url)
    if not url_type:
        return None

    # 短链先展开
    if url_type == "short":
        real = await _bili_resolve_b23(url)
        if not real:
            return None
        url = real
        url_type = get_bilibili_url_type(url)
        if not url_type:
            return None

    result: Optional[Dict[str, Any]] = None

    if url_type == "video":
        result = await resolve_bilibili_video_info(url)
    elif url_type == "dynamic":
        result = await resolve_bilibili_dynamic(url, cookie=cookie)
    elif url_type == "live":
        result = await resolve_bilibili_live(url)
    elif url_type == "read":
        result = await resolve_bilibili_read(url)
    elif url_type == "opus":
        result = await resolve_bilibili_opus(url, cookie=cookie)

    if result:
        result["type"] = url_type

    return result


__all__ = [
    "is_bilibili_url",
    "get_bilibili_url_type",
    "resolve_bilibili_video_url",
    "download_bilibili_video_to_temp",
    "resolve_bilibili_dynamic",
    "resolve_bilibili_live",
    "resolve_bilibili_read",
    "resolve_bilibili_opus",
    "resolve_bilibili_video_info",
    "resolve_bilibili_content",
    "resolve_bilibili_short_url",
]


async def resolve_bilibili_short_url(url: str) -> Optional[str]:
    """展开 B 站短链（b23.tv / bili2233.cn）。

    如果不是短链或展开失败，返回原 URL。
    """
    if not is_bilibili_url(url):
        return url
    if get_bilibili_url_type(url) != "short":
        return url
    real = await _bili_resolve_b23(url)
    return real if real else url
