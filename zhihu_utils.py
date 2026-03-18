from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional runtime dependency
    curl_requests = None


_ARTICLE_PATTERN = re.compile(
    r"^https?://zhuanlan\.zhihu\.com/p/(?P<article_id>\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)
_ANSWER_PATTERN = re.compile(
    r"^https?://www\.zhihu\.com/question/(?P<question_id>\d+)/answer/(?P<answer_id>\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)
_QUESTION_PATTERN = re.compile(
    r"^https?://www\.zhihu\.com/question/(?P<question_id>\d+)(?!/answer)(?:[/?#].*)?$",
    re.IGNORECASE,
)
_PIN_PATTERN = re.compile(
    r"^https?://www\.zhihu\.com/pin/(?P<pin_id>\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)

_MEDIA_ATTRS = (
    "src",
    "data-src",
    "data-original",
    "data-actualsrc",
    "data-default-watermark-src",
    "poster",
    "href",
)
_IMAGE_ATTRS = ("src", "data-src", "data-original", "data-actualsrc")
_VIDEO_TAGS = ("video", "source", "iframe")
_VIDEO_EXTENSIONS = (".mp4", ".m3u8", ".mov", ".webm")
_REQUEST_PROFILES = (
    ("desktop", "chrome"),
    ("ios", "safari_ios"),
    ("mobile", "chrome_android"),
)
_BASE_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "referer": "https://www.zhihu.com/",
    "origin": "https://www.zhihu.com",
    "cache-control": "no-cache",
    "pragma": "no-cache",
}


class ZhihuParseError(RuntimeError):
    """Raised when a Zhihu link cannot be parsed into structured context."""


@dataclass(frozen=True, slots=True)
class ZhihuMatch:
    kind: str
    url: str
    target_id: str
    question_id: str | None = None


@dataclass(eq=True, slots=True)
class ZhihuContext:
    kind: str
    title: str
    url: str
    author: str | None = None
    author_description: str | None = None
    created_at: str | None = None
    stats: list[tuple[str, str]] = field(default_factory=list)
    summary: str | None = None
    question_detail: str = ""
    body: str = ""
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)


@dataclass(eq=True, slots=True)
class ZhihuPreparedPrompt:
    prompt: str
    images: list[str]
    context: ZhihuContext


def match_zhihu_url(url: str) -> ZhihuMatch | None:
    if not isinstance(url, str):
        return None
    target = url.strip()
    if not target:
        return None

    if matched := _ARTICLE_PATTERN.match(target):
        return ZhihuMatch(
            kind="article", url=target, target_id=matched.group("article_id")
        )
    if matched := _ANSWER_PATTERN.match(target):
        return ZhihuMatch(
            kind="answer",
            url=target,
            target_id=matched.group("answer_id"),
            question_id=matched.group("question_id"),
        )
    if matched := _QUESTION_PATTERN.match(target):
        return ZhihuMatch(
            kind="question", url=target, target_id=matched.group("question_id")
        )
    if matched := _PIN_PATTERN.match(target):
        return ZhihuMatch(kind="pin", url=target, target_id=matched.group("pin_id"))
    return None


def build_zhihu_context(match: ZhihuMatch, payload: dict[str, Any]) -> ZhihuContext:
    if match.kind == "article":
        return _build_article_context(match, payload)
    if match.kind == "answer":
        return _build_answer_context(match, payload)
    if match.kind == "question":
        return _build_question_context(match, payload)
    if match.kind == "pin":
        return _build_pin_context(match, payload)
    raise ZhihuParseError("知乎链接类型暂不支持。")


def build_zhihu_prompt(context: ZhihuContext) -> str:
    kind_name = {
        "article": "文章",
        "answer": "回答",
        "question": "问题",
        "pin": "想法",
    }.get(context.kind, "内容")

    parts = [
        f"请解释下面的知乎{kind_name}内容。",
        "请概括核心观点、上下文、语气和潜在讨论背景；如果附带图片，请结合图片一起解释。",
        f"链接：{context.url}",
    ]
    if context.title:
        label = "标题" if context.kind != "question" else "问题"
        parts.append(f"{label}：{context.title}")
    if context.author:
        parts.append(f"作者：{context.author}")
    if context.author_description:
        parts.append(f"作者简介：{context.author_description}")
    if context.created_at:
        parts.append(f"发布时间：{context.created_at}")
    if context.stats:
        stats_text = " | ".join(f"{label} {value}" for label, value in context.stats)
        parts.append(f"统计：{stats_text}")
    if context.summary:
        parts.append(f"摘要：{context.summary}")
    if context.question_detail:
        parts.append(f"问题描述：\n{context.question_detail}")
    if context.body:
        body_label = "正文"
        if context.kind == "answer":
            body_label = "回答正文"
        elif context.kind == "question":
            body_label = "默认排序首条回答"
        parts.append(f"{body_label}：\n{context.body}")
    if context.images:
        parts.append(f"附图数量：{len(context.images)}")
    if context.videos:
        parts.append("视频链接：\n" + "\n".join(context.videos[:5]))
    return "\n\n".join(part for part in parts if part).strip()


async def prepare_zhihu_prompt(
    url: str,
    *,
    cookie: str,
    timeout_sec: int = 20,
    proxy: str | None = None,
) -> ZhihuPreparedPrompt:
    if not isinstance(cookie, str) or not cookie.strip():
        raise ZhihuParseError(
            "知乎链接解析需要有效 cookie，请在插件配置中填写 zhihu_cookie。"
        )

    matched = match_zhihu_url(url)
    if matched is None:
        raise ZhihuParseError("未识别到受支持的知乎链接。")

    payload = await fetch_zhihu_payload(
        matched,
        cookie=cookie.strip(),
        timeout_sec=max(int(timeout_sec), 2),
        proxy=proxy,
    )
    context = build_zhihu_context(matched, payload)
    return ZhihuPreparedPrompt(
        prompt=build_zhihu_prompt(context),
        images=context.images,
        context=context,
    )


async def fetch_zhihu_payload(
    matched: ZhihuMatch,
    *,
    cookie: str,
    timeout_sec: int,
    proxy: str | None = None,
) -> dict[str, Any]:
    if matched.kind == "pin":
        return await _fetch_pin_payload(
            matched.target_id,
            cookie=cookie,
            timeout_sec=timeout_sec,
            proxy=proxy,
        )
    return await _fetch_initial_data(
        matched.url,
        cookie=cookie,
        timeout_sec=timeout_sec,
        proxy=proxy,
    )


async def _fetch_initial_data(
    url: str,
    *,
    cookie: str,
    timeout_sec: int,
    proxy: str | None,
) -> dict[str, Any]:
    saw_challenge = False
    saw_login = False
    last_error: Exception | None = None

    for _profile_name, impersonate in _REQUEST_PROFILES:
        try:
            response = await _request_text(
                url,
                cookie=cookie,
                timeout_sec=timeout_sec,
                proxy=proxy,
                accept=_BASE_HEADERS["accept"],
                impersonate=impersonate,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            continue

        html_text = response["text"]
        final_url = response["final_url"]
        status_code = response["status_code"]
        if _is_challenge_page(html_text, status_code=status_code):
            saw_challenge = True
            continue
        if _is_login_page(final_url, html_text):
            saw_login = True
            continue

        payload = _extract_initial_data(html_text)
        if payload is not None:
            return payload

    if saw_challenge:
        raise ZhihuParseError("知乎抓取失败：当前 cookie 可能失效，或请求被风控拦截。")
    if saw_login:
        raise ZhihuParseError(
            "知乎抓取失败：当前请求被引导到登录页，请更新 zhihu_cookie。"
        )
    if last_error is not None:
        raise ZhihuParseError("知乎页面抓取失败。") from last_error
    raise ZhihuParseError("知乎页面抓取失败：未找到可解析数据。")


async def _fetch_pin_payload(
    pin_id: str,
    *,
    cookie: str,
    timeout_sec: int,
    proxy: str | None,
) -> dict[str, Any]:
    url = (
        "https://www.zhihu.com/api/v4/pins/"
        f"{pin_id}?include=content,content_html,created_time,updated_time,author,origin_pin"
    )
    saw_challenge = False
    saw_login = False
    last_error: Exception | None = None

    for _profile_name, impersonate in _REQUEST_PROFILES:
        try:
            response = await _request_text(
                url,
                cookie=cookie,
                timeout_sec=timeout_sec,
                proxy=proxy,
                accept="application/json, text/plain, */*",
                impersonate=impersonate,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            continue

        body_text = response["text"]
        final_url = response["final_url"]
        status_code = response["status_code"]
        content_type = response["content_type"]
        if _is_challenge_page(body_text, status_code=status_code) or status_code in (
            401,
            403,
        ):
            saw_challenge = True
            continue
        if _is_login_page(final_url, body_text):
            saw_login = True
            continue
        payload = _extract_json_payload(body_text, content_type=content_type)
        if payload is not None:
            return payload

    if saw_challenge:
        raise ZhihuParseError("知乎抓取失败：当前 cookie 可能失效，或请求被风控拦截。")
    if saw_login:
        raise ZhihuParseError(
            "知乎抓取失败：当前请求被引导到登录页，请更新 zhihu_cookie。"
        )
    if last_error is not None:
        raise ZhihuParseError("知乎想法抓取失败。") from last_error
    raise ZhihuParseError("知乎想法抓取失败：未返回有效 JSON。")


async def _request_text(
    url: str,
    *,
    cookie: str,
    timeout_sec: int,
    proxy: str | None,
    accept: str,
    impersonate: str,
) -> dict[str, Any]:
    if curl_requests is None:
        raise ZhihuParseError("当前环境缺少 curl_cffi 依赖，无法解析知乎链接。")

    headers = dict(_BASE_HEADERS)
    headers["accept"] = accept
    headers["cookie"] = cookie

    def do_request():
        return curl_requests.get(
            url,
            headers=headers,
            impersonate=impersonate,
            proxies={"https": proxy, "http": proxy} if proxy else None,
            timeout=timeout_sec,
            allow_redirects=True,
        )

    response = await asyncio.to_thread(do_request)
    return {
        "status_code": int(response.status_code),
        "final_url": str(response.url),
        "text": str(response.text),
        "content_type": str(response.headers.get("content-type", "")),
    }


def _build_article_context(match: ZhihuMatch, payload: dict[str, Any]) -> ZhihuContext:
    article = (_entities(payload).get("articles") or {}).get(match.target_id) or {}
    if not isinstance(article, dict) or not article:
        raise ZhihuParseError("知乎文章数据不存在。")

    content_html = str(article.get("content") or "")
    body = _html_to_text(content_html, keep_newlines=True)
    images = _extract_image_urls(content_html, match.url)
    videos = _extract_video_urls(content_html, match.url)
    summary = _pick_first_non_empty(
        _normalize_text(str(article.get("excerpt") or ""), keep_newlines=False),
        _truncate_text(body, 140),
    )
    return ZhihuContext(
        kind="article",
        title=_normalize_text(str(article.get("title") or ""), keep_newlines=False),
        url=match.url,
        author=_author_name(article.get("author")),
        author_description=_author_description(article.get("author")),
        created_at=_format_timestamp(article.get("created")),
        stats=_content_stats(
            article.get("voteupCount"),
            article.get("commentCount"),
            article.get("favlistsCount") or article.get("favoriteCount"),
            article.get("likedCount"),
            labels=("赞同", "评论", "收藏", "喜欢"),
        ),
        summary=summary,
        body=body,
        images=images,
        videos=videos,
    )


def _build_answer_context(match: ZhihuMatch, payload: dict[str, Any]) -> ZhihuContext:
    entities = _entities(payload)
    answer = (entities.get("answers") or {}).get(match.target_id) or {}
    question = (entities.get("questions") or {}).get(match.question_id or "") or {}
    if not isinstance(answer, dict) or not answer:
        raise ZhihuParseError("知乎回答数据不存在。")
    if not isinstance(question, dict) or not question:
        raise ZhihuParseError("知乎问题数据不存在。")

    content_html = str(answer.get("content") or "")
    body = _html_to_text(content_html, keep_newlines=True)
    images = _extract_image_urls(content_html, match.url)
    videos = _extract_video_urls(content_html, match.url)
    summary = _pick_first_non_empty(
        _normalize_text(str(answer.get("excerpt") or ""), keep_newlines=False),
        _truncate_text(body, 140),
    )
    return ZhihuContext(
        kind="answer",
        title=_normalize_text(str(question.get("title") or ""), keep_newlines=False),
        url=match.url,
        author=_author_name(answer.get("author")),
        author_description=_author_description(answer.get("author")),
        created_at=_format_timestamp(answer.get("createdTime")),
        stats=_content_stats(
            answer.get("voteupCount"),
            answer.get("commentCount"),
            answer.get("favlistsCount") or answer.get("favoriteCount"),
            answer.get("thanksCount") or answer.get("likedCount"),
            labels=("赞同", "评论", "收藏", "喜欢"),
        ),
        summary=summary,
        body=body,
        images=images,
        videos=videos,
    )


def _build_question_context(match: ZhihuMatch, payload: dict[str, Any]) -> ZhihuContext:
    entities = _entities(payload)
    question = (entities.get("questions") or {}).get(match.target_id) or {}
    if not isinstance(question, dict) or not question:
        raise ZhihuParseError("知乎问题数据不存在。")

    answer_id = _pick_first_answer_id(payload, match.target_id)
    if not answer_id:
        raise ZhihuParseError("知乎问题页未找到默认排序首条回答。")
    answer = (entities.get("answers") or {}).get(answer_id) or {}
    if not isinstance(answer, dict) or not answer:
        raise ZhihuParseError("知乎首条回答数据不存在。")

    detail_html = str(question.get("detail") or "")
    question_detail = _html_to_text(detail_html, keep_newlines=True)
    answer_html = str(answer.get("content") or "")
    body = _pick_first_non_empty(
        _html_to_text(answer_html, keep_newlines=True),
        _normalize_text(str(answer.get("excerpt") or ""), keep_newlines=True),
    )
    images = _dedupe(
        _extract_image_urls(detail_html, match.url)
        + _extract_image_urls(answer_html, match.url)
    )
    videos = _extract_video_urls(answer_html, match.url)
    summary = _pick_first_non_empty(
        _normalize_text(str(answer.get("excerpt") or ""), keep_newlines=False),
        _truncate_text(body, 140),
        _truncate_text(question_detail, 140),
    )
    return ZhihuContext(
        kind="question",
        title=_normalize_text(str(question.get("title") or ""), keep_newlines=False),
        url=match.url,
        author=_author_name(answer.get("author")),
        author_description=_author_description(answer.get("author")),
        created_at=_format_timestamp(answer.get("createdTime")),
        stats=_question_stats(question),
        summary=summary,
        question_detail=question_detail,
        body=body,
        images=images,
        videos=videos,
    )


def _build_pin_context(match: ZhihuMatch, payload: dict[str, Any]) -> ZhihuContext:
    if not isinstance(payload, dict) or not payload:
        raise ZhihuParseError("知乎想法数据不存在。")

    content_html = str(payload.get("content_html") or payload.get("contentHtml") or "")
    body = _pick_first_non_empty(
        _html_to_text(content_html, keep_newlines=True),
        _normalize_text(
            _find_text_value(payload, ("content", "text")), keep_newlines=True
        ),
    )
    images = _extract_image_urls(content_html, match.url)
    videos = _extract_video_urls(content_html, match.url)
    summary = _pick_first_non_empty(_truncate_text(body, 140), "知乎想法")
    return ZhihuContext(
        kind="pin",
        title="知乎想法",
        url=match.url,
        author=_author_name(payload.get("author")),
        author_description=_author_description(payload.get("author")),
        created_at=_format_timestamp(
            payload.get("created_time") or payload.get("updated_time")
        ),
        stats=_content_stats(
            payload.get("voteup_count") or payload.get("voteupCount"),
            payload.get("comment_count") or payload.get("commentCount"),
            None,
            None,
            labels=("赞同", "评论", "收藏", "喜欢"),
        ),
        summary=summary,
        body=body,
        images=images,
        videos=videos,
    )


def _extract_initial_data(html_text: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html_text, "html.parser")
    node = soup.select_one('script#js-initialData[type="text/json"]')
    if node is None:
        return None
    raw = node.get_text(strip=True)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    initial_state = payload.get("initialState")
    return payload if isinstance(initial_state, dict) else None


def _extract_json_payload(raw_text: str, *, content_type: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    if "application/json" not in content_type.lower() and not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _entities(initial_data: dict[str, Any]) -> dict[str, Any]:
    initial_state = initial_data.get("initialState") or {}
    entities = initial_state.get("entities") or {}
    return entities if isinstance(entities, dict) else {}


def _pick_first_answer_id(initial_data: dict[str, Any], question_id: str) -> str | None:
    initial_state = initial_data.get("initialState") or {}
    answers = ((initial_state.get("question") or {}).get("answers") or {}).get(
        question_id
    ) or {}
    ids = answers.get("ids") or []
    if not ids or not isinstance(ids[0], dict):
        return None
    target = ids[0].get("target")
    return str(target) if target else None


def _normalize_text(text: str, *, keep_newlines: bool) -> str:
    if not isinstance(text, str):
        return ""
    value = unescape(text).replace("\r", "\n")
    if keep_newlines:
        lines = [re.sub(r"\s+", " ", line).strip() for line in value.split("\n")]
        return "\n".join(line for line in lines if line).strip()
    return re.sub(r"\s+", " ", value).strip()


def _html_to_text(html_text: str, *, keep_newlines: bool) -> str:
    if not isinstance(html_text, str) or not html_text.strip():
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text("\n" if keep_newlines else " ", strip=True)
    return _normalize_text(text, keep_newlines=keep_newlines)


def _extract_image_urls(html_text: str, page_url: str) -> list[str]:
    if not isinstance(html_text, str) or not html_text.strip():
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    urls: list[str] = []
    for node in soup.find_all("img"):
        for key in _IMAGE_ATTRS:
            raw = node.get(key)
            normalized = _normalize_media_url(raw, page_url)
            if normalized:
                urls.append(normalized)
                break
    return _dedupe(urls)


def _extract_video_urls(html_text: str, page_url: str) -> list[str]:
    if not isinstance(html_text, str) or not html_text.strip():
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    urls: list[str] = []
    for node in soup.find_all(_VIDEO_TAGS):
        for key in _MEDIA_ATTRS:
            raw = node.get(key)
            normalized = _normalize_media_url(raw, page_url)
            if normalized and _looks_like_video_url(normalized):
                urls.append(normalized)
                break
    return _dedupe(urls)


def _normalize_media_url(raw: Any, page_url: str) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or value.startswith("data:"):
        return None
    if value.startswith("//"):
        return "https:" + value
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/"):
        return urljoin(page_url, value)
    return urljoin(page_url, value)


def _looks_like_video_url(url: str) -> bool:
    lowered = url.lower()
    return (
        any(lowered.endswith(ext) for ext in _VIDEO_EXTENSIONS) or "/video/" in lowered
    )


def _author_name(author_data: Any) -> str | None:
    if not isinstance(author_data, dict):
        return None
    name = str(author_data.get("name") or "").strip()
    return name or None


def _author_description(author_data: Any) -> str | None:
    if not isinstance(author_data, dict):
        return None
    description = _normalize_text(
        str(author_data.get("headline") or author_data.get("description") or ""),
        keep_newlines=False,
    )
    return description or None


def _content_stats(
    voteup: Any,
    comment: Any,
    favorite: Any,
    liked: Any,
    *,
    labels: tuple[str, str, str, str],
) -> list[tuple[str, str]]:
    stats: list[tuple[str, str]] = []
    for label, value in zip(labels, (voteup, comment, favorite, liked), strict=True):
        formatted = _format_count(value)
        if formatted:
            stats.append((label, formatted))
    return stats


def _question_stats(question: dict[str, Any]) -> list[tuple[str, str]]:
    stats: list[tuple[str, str]] = []
    for label, value in (
        ("回答", question.get("answerCount")),
        ("关注", question.get("followerCount")),
        ("浏览", question.get("visitCount")),
    ):
        formatted = _format_count(value)
        if formatted:
            stats.append((label, formatted))
    return stats


def _format_count(value: Any) -> str | None:
    number = _safe_int(value)
    if number is None:
        return None
    if abs(number) >= 100000000:
        text = f"{number / 100000000:.1f}".rstrip("0").rstrip(".")
        return f"{text}亿"
    if abs(number) >= 10000:
        text = f"{number / 10000:.1f}".rstrip("0").rstrip(".")
        return f"{text}万"
    return str(number)


def _format_timestamp(value: Any) -> str | None:
    timestamp = _safe_int(value)
    if timestamp is None or timestamp <= 0:
        return None
    if timestamp >= 10**12:
        timestamp //= 1000
    try:
        dt = datetime.fromtimestamp(timestamp)
    except Exception:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return None
    try:
        return int(value)
    except Exception:
        return None


def _truncate_text(text: str, limit: int) -> str:
    value = _normalize_text(text, keep_newlines=False)
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit].rstrip(" ，,；;。！？!?、") + "…"


def _pick_first_non_empty(*values: str | None) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _find_text_value(data: Any, keys: tuple[str, ...]) -> str:
    queue = [data]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            for key in keys:
                value = current.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return ""


def _is_challenge_page(html_text: str, *, status_code: int) -> bool:
    lowered = html_text.lower()
    return (
        'id="zh-zse-ck"' in lowered
        or "static.zhihu.com/zse-ck/" in lowered
        or 'appname":"zse_ck"' in lowered
        or (status_code == 403 and "zse-ck" in lowered)
    )


def _is_login_page(final_url: str, html_text: str) -> bool:
    lowered_url = final_url.lower()
    lowered_html = html_text.lower()
    return (
        "/signin" in lowered_url
        or "/signup" in lowered_url
        or "<title>知乎 - 有问题，就会有答案</title>" in lowered_html
    )
