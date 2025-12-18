from __future__ import annotations

from typing import Iterable, Optional, Set, Dict, Any, List
import os
import io
import re

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import fitz  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    import PyPDF2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    PyPDF2 = None  # type: ignore[assignment]

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

from .message_utils import get_reply_message_id, ob_data, call_get_forward_msg


def build_text_exts_from_config(raw: str, default_exts: Iterable[str]) -> Set[str]:
    """根据配置字符串构造文本扩展名集合。

    - raw: 类似 'txt,md,log' 的配置值，可为空。
    - default_exts: 代码内置的默认扩展名集合。
    """
    base: Set[str] = set()
    for ext in default_exts:
        e = str(ext).strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        base.add(e)
    if not isinstance(raw, str) or not raw.strip():
        return base
    for part in raw.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if not p.startswith("."):
            p = "." + p
        base.add(p)
    return base


def _normalize_pdf_page_text(raw: Optional[str]) -> str:
    """将单页 PDF 提取出的原始文本整理为更适合 LLM 处理的 Markdown 段落。

    - 合并同一段落内的换行，保留空行作为段落分隔
    - 尝试保留简单的列表/编号结构
    """
    if not isinstance(raw, str):
        return ""
    lines = [ln.rstrip() for ln in raw.splitlines()]
    blocks: List[str] = []
    current: List[str] = []

    bullet_pattern = re.compile(r"^(\s*[-*•·]\s+|\s*\d{1,3}[.)]\s+)")

    def flush_paragraph() -> None:
        if not current:
            return
        joined = " ".join(s.strip() for s in current if s.strip())
        if joined:
            blocks.append(joined)
        current.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush_paragraph()
            continue
        # 列表/编号行：直接单独成段，避免被错误合并
        if bullet_pattern.match(s):
            flush_paragraph()
            blocks.append(s)
            continue
        current.append(s)

    flush_paragraph()
    return "\n\n".join(blocks).strip()


def pdf_bytes_to_markdown(data: bytes, max_pages: Optional[int] = None) -> str:
    """将 PDF 二进制内容转换为简单 Markdown 文本。

    - 按页生成 `### 第 N 页` 标题
    - 每页内部按段落拆分，合并过多换行
    - max_pages 为正数时，仅转换前若干页
    """
    if not data:
        return ""

    # 优先使用 PyMuPDF 的 markdown 输出，保留原有章节标题结构
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")  # type: ignore[arg-type]
            page_count = int(getattr(doc, "page_count", len(doc)))  # type: ignore[arg-type]
            md_pages: List[str] = []
            for idx in range(page_count):
                page_no = idx + 1
                if isinstance(max_pages, int) and max_pages > 0 and page_no > max_pages:
                    break
                try:
                    page = doc.load_page(idx)
                    # 直接使用 PyMuPDF 的 markdown 模式，保留标题/列表等结构
                    md = page.get_text("markdown") or page.get_text("text")
                except Exception:
                    md = ""
                if isinstance(md, str):
                    md = md.strip()
                if md:
                    md_pages.append(f"### 第 {page_no} 页\n\n{md}")
            if md_pages:
                return "\n\n---\n\n".join(md_pages).strip()
        except Exception as e:  # pragma: no cover - 环境可能无 PyMuPDF
            logger.warning(f"zssm_explain: PyMuPDF markdown extract failed: {e}")

    # 回退到 PyPDF2 的纯文本提取，再做简单 Markdown 规整
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))  # type: ignore[call-arg]
    except Exception as e:
        logger.warning(f"zssm_explain: pdf read failed: {e}")
        return ""

    md_pages_fallback: List[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        if isinstance(max_pages, int) and max_pages > 0 and idx > max_pages:
            break
        try:
            raw = page.extract_text()  # type: ignore[call-arg]
        except Exception:
            raw = None
        text = _normalize_pdf_page_text(raw)
        if text:
            md_pages_fallback.append(f"### 第 {idx} 页\n\n{text}")
    return "\n\n---\n\n".join(md_pages_fallback).strip()


def _find_first_file_in_message_list(msg_list: List[Any]) -> Optional[Dict[str, Any]]:
    """在 OneBot/Napcat 消息段列表中查找首个 type=file 段，支持简单嵌套 content/message。"""
    if not isinstance(msg_list, list):
        return None
    for seg in msg_list:
        try:
            if not isinstance(seg, dict):
                continue
            t = seg.get("type")
            d = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            if t == "file" and isinstance(d, dict):
                return seg
            content = seg.get("content") or seg.get("message")
            if isinstance(content, list):
                inner = _find_first_file_in_message_list(content)
                if inner is not None:
                    return inner
        except Exception:
            continue
    return None


def _find_first_file_in_forward_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """在 OneBot get_forward_msg 返回的结构中查找首个文件段。"""
    data = ob_data(payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return None
    msgs = (
        data.get("messages")
        or data.get("message")
        or data.get("nodes")
        or data.get("nodeList")
    )
    if not isinstance(msgs, list):
        return None
    for node in msgs:
        try:
            content = None
            if isinstance(node, dict):
                content = node.get("content") or node.get("message")
            if isinstance(content, list):
                seg = _find_first_file_in_message_list(content)
                if seg is not None:
                    return seg
        except Exception:
            continue
    return None


async def extract_file_preview_from_reply(
    event: AstrMessageEvent,
    text_exts: Set[str],
    max_size_bytes: Optional[int] = None,
) -> Optional[str]:
    """尝试从被回复的 Napcat 文件消息中构造文件内容预览文本。

    仅在 OneBot/Napcat 平台 (aiocqhttp) 且存在 Reply 组件时生效。
    """
    try:
        platform = event.get_platform_name()
    except Exception:
        platform = None
    if platform != "aiocqhttp" or not hasattr(event, "bot"):
        return None

    # 定位 Reply 组件
    try:
        chain = event.get_messages()
    except Exception:
        chain = getattr(event.message_obj, "message", []) if hasattr(event, "message_obj") else []
    reply_comp = None
    for seg in chain:
        try:
            if isinstance(seg, Comp.Reply):
                reply_comp = seg
                break
        except Exception:
            continue
    if not reply_comp:
        return None

    reply_id = get_reply_message_id(reply_comp)
    if not reply_id:
        return None

    # 调用 get_msg 获取原始消息，查找其中的 file 段
    try:
        ret: Dict[str, Any] = await event.bot.api.call_action("get_msg", message_id=reply_id)
    except Exception:
        return None
    data = ob_data(ret) if isinstance(ret, dict) else {}
    if not isinstance(data, dict):
        return None
    msg_list = data.get("message") or data.get("messages")
    if not isinstance(msg_list, list):
        return None

    # 1) 优先在原始消息的顶层段中查找文件
    file_seg = _find_first_file_in_message_list(msg_list)

    # 2) 若未找到文件，尝试处理 Napcat “合并转发”场景：查找 forward 段并通过 get_forward_msg 拉取节点
    if not file_seg:
        for seg in msg_list:
            try:
                if not isinstance(seg, dict):
                    continue
                t = seg.get("type")
                if t not in ("forward", "forward_msg", "nodes"):
                    continue
                d = seg.get("data") if isinstance(seg.get("data"), dict) else {}
                fid = d.get("id")
                if not isinstance(fid, str) or not fid:
                    continue
                try:
                    fwd = await call_get_forward_msg(event, str(fid))
                except Exception as fe:
                    logger.warning(f"zssm_explain: get_forward_msg for file preview failed: {fe}")
                    continue
                inner = _find_first_file_in_forward_payload(fwd)
                if inner is not None:
                    file_seg = inner
                    break
            except Exception:
                continue

    if not file_seg:
        return None

    d = file_seg.get("data") or {}
    if not isinstance(d, dict):
        return None
    file_id = d.get("file")
    file_name = d.get("name") or d.get("file") or "未命名文件"
    summary = d.get("summary") or ""
    if not isinstance(file_id, str) or not file_id:
        return None

    return await build_group_file_preview(
        event=event,
        file_id=file_id,
        file_name=file_name,
        summary=summary,
        text_exts=text_exts,
        max_size_bytes=max_size_bytes,
    )


async def build_group_file_preview(
    event: AstrMessageEvent,
    file_id: str,
    file_name: str,
    summary: str,
    text_exts: Set[str],
    max_size_bytes: Optional[int] = None,
) -> Optional[str]:
    """获取群文件下载链接，尝试读取文本内容片段并构造预览。

    text_exts 为允许尝试内容预览的扩展名集合（包含点，如 '.txt'）。
    """
    # 仅支持群聊场景，私聊暂不处理
    try:
        gid = event.get_group_id()
    except Exception:
        gid = None

    url: Optional[str] = None

    # 群聊：优先使用 get_group_file_url
    if gid:
        try:
            group_id = int(gid)
        except Exception:
            group_id = None
        if group_id is not None:
            try:
                url_result = await event.bot.api.call_action(
                    "get_group_file_url",
                    group_id=group_id,
                    file_id=file_id,
                )
                url = url_result.get("url") if isinstance(url_result, dict) else None
            except Exception as e:
                logger.warning(f"zssm_explain: get_group_file_url failed: {e}")

    # 私聊或群聊兜底：使用 get_private_file_url
    if not url:
        try:
            url_result = await event.bot.api.call_action(
                "get_private_file_url",
                file_id=file_id,
            )
            data = url_result.get("data") if isinstance(url_result, dict) else None
            if isinstance(data, dict):
                url = data.get("url")
        except Exception as e:
            logger.warning(f"zssm_explain: get_private_file_url failed: {e}")

    # 元信息部分（即使无法下载，也可以使用）
    meta_lines: List[str] = [f"[群文件] {file_name}"]
    if summary:
        meta_lines.append(f"说明: {summary}")

    # 仅对配置允许的文本扩展名或 PDF 尝试内容预览
    if not url or aiohttp is None:
        return "\n".join(meta_lines)

    name_lower = str(file_name).lower()
    _, ext = os.path.splitext(name_lower)
    is_pdf = ext == ".pdf"
    if ext not in text_exts and not is_pdf:
        # 非文本类/非 PDF 文件暂不尝试解析内容
        return "\n".join(meta_lines)

    snippet = ""
    size_hint = ""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logger.warning(
                        "zssm_explain: fetch group file failed, status=%s", resp.status
                    )
                else:
                    cl = resp.headers.get("Content-Length")
                    sz = None
                    if cl and cl.isdigit():
                        sz = int(cl)
                        if sz >= 0:
                            if sz < 1024:
                                size_hint = f"{sz} B"
                            elif sz < 1024 * 1024:
                                size_hint = f"{sz / 1024:.1f} KB"
                            else:
                                size_hint = f"{sz / 1024 / 1024:.2f} MB"
                            # 若为非 PDF 文件且配置了最大文件大小，且当前文件超出阈值，则仅返回元信息
                            if not is_pdf and isinstance(max_size_bytes, int) and max_size_bytes > 0 and sz > max_size_bytes:
                                meta_lines.append(f"大小: {size_hint}")
                                meta_lines.append("（文件体积较大，已跳过内容预览，仅展示元信息）")
                                return "\n".join(meta_lines)
                    # PDF：尝试使用 PyPDF2 将内容转换为 Markdown 文本
                    if is_pdf and PyPDF2 is not None:
                        # 固定读取不超过约 2MB 的二进制内容，再按转换后的 Markdown 文本大小与全局限制比较
                        limit = 2 * 1024 * 1024
                        buf = io.BytesIO()
                        total = 0
                        async for chunk in resp.content.iter_chunked(8192):
                            if not chunk:
                                break
                            total += len(chunk)
                            if isinstance(limit, int) and total > limit:
                                break
                            buf.write(chunk)
                        try:
                            pdf_bytes = buf.getvalue()
                            text = pdf_bytes_to_markdown(pdf_bytes)
                        except Exception as e:
                            logger.warning(f"zssm_explain: pdf text extract failed: {e}")
                            text = ""
                        if text:
                            # 对于 PDF，按提取出的 Markdown 文本大小进行限制，而非 PDF 文件体积
                            if isinstance(max_size_bytes, int) and max_size_bytes > 0:
                                try:
                                    txt_bytes = len(text.encode("utf-8", errors="ignore"))
                                except Exception:
                                    txt_bytes = len(text)
                                if txt_bytes > max_size_bytes:
                                    if size_hint:
                                        meta_lines.append(f"大小: {size_hint}")
                                    meta_lines.append("（PDF 文本内容较长，已跳过内容预览，仅展示元信息）")
                                    return "\n".join(meta_lines)
                            snippet = text if len(text) <= 400 else (text[:400] + " ...")
                    else:
                        # 纯文本类文件：读取前 4KB 作为预览
                        max_bytes = 4096
                        data = await resp.content.read(max_bytes)
                        try:
                            text = data.decode("utf-8", errors="ignore")
                        except Exception:
                            text = ""
                        text = text.strip()
                        if text:
                            snippet = text if len(text) <= 400 else (text[:400] + " ...")
    except Exception as e:
        logger.warning(f"zssm_explain: preview group file content failed: {e}")

    if size_hint:
        meta_lines.append(f"大小: {size_hint}")
    if snippet:
        meta_lines.append("内容片段（截取部分，可能不完整）:")
        meta_lines.append(snippet)

    return "\n".join(meta_lines)
