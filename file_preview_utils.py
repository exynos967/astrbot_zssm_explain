from __future__ import annotations

from typing import Iterable, Optional, Set, Dict, Any, List
import os

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

from .message_utils import get_reply_message_id, ob_data


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


async def extract_file_preview_from_reply(
    event: AstrMessageEvent,
    text_exts: Set[str],
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

    file_seg = None
    for seg in msg_list:
        try:
            if not isinstance(seg, dict):
                continue
            if seg.get("type") == "file":
                file_seg = seg
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

    return await build_group_file_preview(event, file_id, file_name, summary, text_exts)


async def build_group_file_preview(
    event: AstrMessageEvent,
    file_id: str,
    file_name: str,
    summary: str,
    text_exts: Set[str],
) -> Optional[str]:
    """获取群文件下载链接，尝试读取文本内容片段并构造预览。

    text_exts 为允许尝试内容预览的扩展名集合（包含点，如 '.txt'）。
    """
    # 仅支持群聊场景，私聊暂不处理
    try:
        gid = event.get_group_id()
    except Exception:
        gid = None
    if not gid:
        return None

    try:
        group_id = int(gid)
    except Exception:
        return None

    # 调用 Napcat get_group_file_url 获取下载链接
    try:
        url_result = await event.bot.api.call_action(
            "get_group_file_url",
            group_id=group_id,
            file_id=file_id,
        )
        url = url_result.get("url") if isinstance(url_result, dict) else None
    except Exception as e:
        logger.warning(f"zssm_explain: get_group_file_url failed: {e}")
        url = None

    # 元信息部分（即使无法下载，也可以使用）
    meta_lines: List[str] = [f"[群文件] {file_name}"]
    if summary:
        meta_lines.append(f"说明: {summary}")

    # 仅对配置允许的文本扩展名尝试内容预览
    if not url or aiohttp is None:
        return "\n".join(meta_lines)

    name_lower = str(file_name).lower()
    _, ext = os.path.splitext(name_lower)
    if ext not in text_exts:
        # 非文本类文件暂不尝试解析内容
        return "\n".join(meta_lines)

    max_bytes = 4096
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
                    if cl and cl.isdigit():
                        sz = int(cl)
                        if sz >= 0:
                            if sz < 1024:
                                size_hint = f"{sz} B"
                            elif sz < 1024 * 1024:
                                size_hint = f"{sz / 1024:.1f} KB"
                            else:
                                size_hint = f"{sz / 1024 / 1024:.2f} MB"
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

