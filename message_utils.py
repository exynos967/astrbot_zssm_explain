from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict
import json

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp


def extract_text_and_images_from_chain(chain: List[object]) -> Tuple[str, List[str]]:
    """从一段消息链中提取纯文本与图片地址/路径；支持合并转发节点的递归提取。"""
    texts: List[str] = []
    images: List[str] = []
    if not isinstance(chain, list):
        return ("", images)
    for seg in chain:
        try:
            if isinstance(seg, Comp.Plain):
                txt = getattr(seg, "text", None)
                texts.append(txt if isinstance(txt, str) else str(seg))
            elif isinstance(seg, Comp.Image):
                f = getattr(seg, "file", None)
                u = getattr(seg, "url", None)
                if isinstance(f, str) and f:
                    images.append(f)
                elif isinstance(u, str) and u:
                    images.append(u)
            elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                content = getattr(seg, "content", None)
                if isinstance(content, list):
                    t2, i2 = extract_text_and_images_from_chain(content)
                    if t2:
                        texts.append(t2)
                    images.extend(i2)
            elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            t2, i2 = extract_text_and_images_from_chain(c)
                            if t2:
                                texts.append(t2)
                            images.extend(i2)
            elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            t2, i2 = extract_text_and_images_from_chain(c)
                            if t2:
                                texts.append(t2)
                            images.extend(i2)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"zssm_explain: parse chain segment failed: {e}")
    return ("\n".join([t for t in texts if t]).strip(), images)


def chain_has_forward(chain: List[object]) -> bool:
    """检测一段消息链中是否包含合并转发相关组件（Forward/Node/Nodes）。"""
    if not isinstance(chain, list):
        return False
    for seg in chain:
        try:
            if hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                return True
            if hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                return True
            if hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                return True
        except Exception:
            continue
    return False


def _strip_bracket_prefix(text: str) -> str:
    """去掉类似 [QQ小程序] / 【聊天记录】 这样的前缀标签。"""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    if s.startswith("["):
        end = s.find("]")
        if end != -1:
            return s[end + 1 :].strip()
    if s.startswith("【"):
        end = s.find("】")
        if end != -1:
            return s[end + 1 :].strip()
    return s


def _format_json_share(data: Dict[str, Any]) -> str:
    """
    解析 Napcat/OneBot json 消息 (data.data 对应的结构化 JSON)，提取关键信息。

    支持：
    - 聊天记录: app = com.tencent.multimsg
    - QQ 小程序分享（含 B 站小程序）: app = com.tencent.miniapp_01
    - 普通图文/小程序卡片: app = com.tencent.tuwen.lua
    """
    if not isinstance(data, dict):
        return ""

    app = data.get("app") or ""

    # 3) 合并转发聊天记录
    if app == "com.tencent.multimsg":
        prompt = data.get("prompt") or data.get("desc") or "[聊天记录]"
        detail = data.get("meta", {}).get("detail", {}) or {}
        summary = detail.get("summary") or ""
        source = detail.get("source") or ""
        lines = [str(prompt).strip() or "[聊天记录]"]
        if source:
            lines.append(f"来源: {source}")
        if summary:
            lines.append(f"摘要: {summary}")
        return "\n".join(lines)

    # 2) B 站等 QQ 小程序分享
    if app == "com.tencent.miniapp_01":
        detail = data.get("meta", {}).get("detail_1", {}) or {}
        raw_prompt = str(data.get("prompt") or "").strip()
        title = _strip_bracket_prefix(raw_prompt) or detail.get("desc") or "无标题"
        desc = detail.get("desc") or ""
        url = detail.get("qqdocurl") or detail.get("url") or ""
        preview = detail.get("preview") or ""
        app_title = detail.get("title") or "小程序"

        lines = [f"[小程序分享 - {app_title}]", f"标题: {title}"]
        if desc:
            lines.append(f"简介: {desc}")
        if url:
            lines.append(f"跳转链接: {url}")
        if preview:
            lines.append(f"封面图: {preview}")
        return "\n".join(lines)

    # 1) 通用图文/小程序卡片（如小红书、微信图文）
    if app == "com.tencent.tuwen.lua":
        news = data.get("meta", {}).get("news", {}) or {}
        title = news.get("title") or "无标题"
        desc = news.get("desc") or ""
        url = news.get("jumpUrl") or ""
        preview = news.get("preview") or ""
        tag = news.get("tag") or "图文消息"
        lines = [f"[图文/小程序分享 - {tag}]", f"标题: {title}"]
        if desc:
            lines.append(f"简介: {desc}")
        if url:
            lines.append(f"跳转链接: {url}")
        if preview:
            lines.append(f"封面图: {preview}")
        return "\n".join(lines)

    # 兜底：未知 JSON 类型，使用 prompt/desc 作为简要说明
    prompt = data.get("prompt") or data.get("desc") or ""
    return str(prompt) if prompt else ""


def try_extract_from_reply_component(reply_comp: object) -> Tuple[Optional[str], List[str], bool]:
    """尽量从 Reply 组件中得到被引用消息的文本与图片。

    返回值第三项标记是否检测到其中包含合并转发节点（Forward/Node/Nodes）。"""
    for attr in ("message", "origin", "content"):
        payload = getattr(reply_comp, attr, None)
        if isinstance(payload, list):
            text, images = extract_text_and_images_from_chain(payload)
            has_forward = chain_has_forward(payload)
            return (text, images, has_forward)
    return (None, [], False)


def get_reply_message_id(reply_comp: object) -> Optional[str]:
    """从 Reply 组件中尽力获取原消息的 message_id（OneBot/Napcat 常见为 id）。"""
    for key in ("id", "message_id", "reply_id", "messageId", "message_seq"):
        val = getattr(reply_comp, key, None)
        if isinstance(val, (str, int)) and str(val):
            return str(val)
    data = getattr(reply_comp, "data", None)
    if isinstance(data, dict):
        for key in ("id", "message_id", "reply", "messageId", "message_seq"):
            val = data.get(key)
            if isinstance(val, (str, int)) and str(val):
                return str(val)
    return None


def ob_data(obj: Any) -> Dict[str, Any]:
    """OneBot 风格响应可能包裹在 data 字段中，展开后返回字典。"""
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, dict):
            return data
        return obj
    return {}


def extract_from_onebot_message_payload(payload: Any) -> Tuple[str, List[str]]:
    """从 OneBot/Napcat get_msg 返回的 payload 中提取文本与图片；识别 forward/nodes 由上层处理。"""
    texts: List[str] = []
    images: List[str] = []
    data = ob_data(payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        msg = data.get("message") or data.get("messages")
        if isinstance(msg, list):
            for seg in msg:
                try:
                    if not isinstance(seg, dict):
                        continue
                    t = seg.get("type")
                    d = seg.get("data", {}) if isinstance(seg.get("data"), dict) else {}
                    if t in ("text", "plain"):
                        txt = d.get("text")
                        if isinstance(txt, str) and txt:
                            texts.append(txt)
                    elif t == "image":
                        url = d.get("url") or d.get("file")
                        if isinstance(url, str) and url:
                            images.append(url)
                    elif t == "json":
                        # Napcat JSON 消息：核心数据在 data.data 中，需要再次解析
                        raw = d.get("data")
                        if isinstance(raw, str) and raw.strip():
                            try:
                                inner = json.loads(raw)
                                summary = _format_json_share(inner)
                                if summary:
                                    texts.append(summary)
                            except Exception as e:
                                logger.warning(f"zssm_explain: parse json segment failed: {e}")
                    elif t == "file":
                        # Napcat 文件消息：data.file 为标识，data.name/summary 为展示信息
                        name = d.get("name") or d.get("file") or "未命名文件"
                        summary = d.get("summary") or ""
                        file_id = d.get("file") or ""
                        parts = [f"[群文件] {name}"]
                        if summary:
                            parts.append(f"说明: {summary}")
                        if file_id:
                            parts.append(f"文件标识: {file_id}")
                        texts.append("\n".join(parts))
                    # 对于 forward/nodes，不在此层解析，由上层触发 get_forward_msg 获取节点
                except Exception as e:
                    logger.warning(f"zssm_explain: parse onebot segment failed: {e}")
            return ("\n".join([t for t in texts if t]).strip(), images)
        elif isinstance(msg, str) and msg:
            texts.append(msg)
            return ("\n".join(texts).strip(), images)
        raw = data.get("raw_message")
        if isinstance(raw, str) and raw:
            texts.append(raw)
            return ("\n".join(texts).strip(), images)
    logger.warning("zssm_explain: failed to extract text from OneBot payload; fallback to empty text")
    return ("", images)


def _extract_forward_nodes_recursively(
    message_nodes: List[Any],
    texts: List[str],
    images: List[str],
    depth: int = 0,
) -> None:
    """递归解析 Napcat/OneBot get_forward_msg 返回的 messages 列表，支持嵌套合并转发。

    设计目标：
    - 复用 forward_reader 插件中的核心递归思路，但以纯函数方式实现，便于在工具函数中复用；
    - 只负责结构展开与文本/图片提取，不做任何网络调用。
    """
    if not isinstance(message_nodes, list):
        return

    indent = "  " * depth

    for message_node in message_nodes:
        try:
            if not isinstance(message_node, dict):
                continue

            sender = message_node.get("sender") or {}
            sender_name = (
                sender.get("nickname")
                or sender.get("card")
                or sender.get("user_id")
                or "未知用户"
            )

            raw_content = message_node.get("message") or message_node.get("content", [])

            content_chain: List[Any] = []
            if isinstance(raw_content, str):
                try:
                    parsed_content = json.loads(raw_content)
                    if isinstance(parsed_content, list):
                        content_chain = parsed_content
                except (json.JSONDecodeError, TypeError):
                    # 无法解析为 JSON 的字符串，当作纯文本处理
                    content_chain = [
                        {
                            "type": "text",
                            "data": {"text": raw_content},
                        }
                    ]
            elif isinstance(raw_content, list):
                content_chain = raw_content

            node_text_parts: List[str] = []
            has_only_forward = False

            if isinstance(content_chain, list):
                first_seg = (
                    content_chain[0]
                    if len(content_chain) == 1 and isinstance(content_chain[0], dict)
                    else None
                )
                if first_seg and first_seg.get("type") == "forward":
                    has_only_forward = True

                for seg in content_chain:
                    if not isinstance(seg, dict):
                        continue
                    seg_type = seg.get("type")
                    seg_data = (
                        seg.get("data", {})
                        if isinstance(seg.get("data"), dict)
                        else {}
                    )

                    if seg_type in ("text", "plain"):
                        text = seg_data.get("text", "")
                        if isinstance(text, str) and text:
                            node_text_parts.append(text)
                    elif seg_type == "image":
                        url = seg_data.get("url") or seg_data.get("file")
                        if isinstance(url, str) and url:
                            images.append(url)
                            node_text_parts.append("[图片]")
                    elif seg_type == "forward":
                        nested_content = seg_data.get("content")
                        if isinstance(nested_content, list):
                            _extract_forward_nodes_recursively(
                                nested_content, texts, images, depth + 1
                            )
                        else:
                            node_text_parts.append("[转发消息内容缺失或格式错误]")

            full_node_text = "".join(node_text_parts).strip()
            if full_node_text and not has_only_forward:
                texts.append(f"{indent}{sender_name}: {full_node_text}")
        except Exception as e:
            logger.warning(f"zssm_explain: parse forward node failed: {e}")


def extract_from_onebot_forward_payload(payload: Any) -> Tuple[str, List[str]]:
    """解析 OneBot get_forward_msg 返回的 messages/nodes 列表，汇总文本与图片。"""
    texts: List[str] = []
    images: List[str] = []
    data = ob_data(payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        msgs = (
            data.get("messages")
            or data.get("message")
            or data.get("nodes")
            or data.get("nodeList")
        )
        if isinstance(msgs, list):
            try:
                _extract_forward_nodes_recursively(msgs, texts, images, depth=0)
            except Exception as e:
                logger.warning(f"zssm_explain: parse forward payload failed: {e}")
    return ("\n".join([x for x in texts if x]).strip(), images)


async def extract_quoted_payload(event: AstrMessageEvent) -> Tuple[Optional[str], List[str], bool]:
    """从当前事件中获取被回复消息的文本与图片。
    优先：Reply 携带嵌入消息；回退：OneBot get_msg；失败：(None, [], False)。

    返回:
    - text: 提取到的文本（可为空字符串）
    - images: 提取到的图片列表
    - from_forward: 是否来源于“合并转发”结构（含 Forward/Node/Nodes 或 get_forward_msg）
    """
    try:
        chain = event.get_messages()
    except Exception:
        chain = getattr(event.message_obj, "message", []) or []

    reply_comp = None
    for seg in chain:
        try:
            if isinstance(seg, Comp.Reply):
                reply_comp = seg
                break
        except Exception:
            pass

    if not reply_comp:
        return (None, [], False)

    text, images, from_forward = try_extract_from_reply_component(reply_comp)
    if text or images:
        return (text, images, from_forward)

    reply_id = get_reply_message_id(reply_comp)
    platform_name = None
    try:
        platform_name = event.get_platform_name()
    except Exception:
        platform_name = None

    if reply_id and platform_name == "aiocqhttp" and hasattr(event, "bot"):
        try:
            ret: Dict[str, Any] = await event.bot.api.call_action("get_msg", message_id=reply_id)
            data = ob_data(ret)
            t2, imgs2 = extract_from_onebot_message_payload(data)
            agg_texts: List[str] = [t2] if t2 else []
            agg_imgs: List[str] = list(imgs2)
            from_forward_ob = False
            try:
                msg_list = data.get("message") if isinstance(data, dict) else None
                if isinstance(msg_list, list):
                    for seg in msg_list:
                        if not isinstance(seg, dict):
                            continue
                        seg_type = seg.get("type")
                        if seg_type in ("forward", "forward_msg", "nodes"):
                            from_forward_ob = True
                            d = seg.get("data", {}) if isinstance(seg.get("data"), dict) else {}
                            fid = d.get("id")
                            if isinstance(fid, str) and fid:
                                try:
                                    fwd = await event.bot.api.call_action("get_forward_msg", id=fid)
                                    ft, fi = extract_from_onebot_forward_payload(fwd)
                                    if ft:
                                        agg_texts.append(ft)
                                    if fi:
                                        agg_imgs.extend(fi)
                                except Exception as fe:
                                    logger.warning(f"zssm_explain: get_forward_msg failed: {fe}")
                        elif seg_type == "json":
                            try:
                                d = (
                                    seg.get("data", {})
                                    if isinstance(seg.get("data"), dict)
                                    else {}
                                )
                                inner_data_str = d.get("data")
                                if isinstance(inner_data_str, str) and inner_data_str.strip():
                                    inner_data_str = inner_data_str.replace("&#44;", ",")
                                    inner_json = json.loads(inner_data_str)
                                    if inner_json.get("app") == "com.tencent.multimsg" and inner_json.get(
                                        "config", {}
                                    ).get("forward") == 1:
                                        detail = inner_json.get("meta", {}).get("detail", {}) or {}
                                        news_items = detail.get("news", []) or []
                                        for item in news_items:
                                            if not isinstance(item, dict):
                                                continue
                                            text_content = item.get("text")
                                            if isinstance(text_content, str):
                                                clean_text = (
                                                    text_content.strip().replace("[图片]", "").strip()
                                                )
                                                if clean_text:
                                                    agg_texts.append(clean_text)
                                        if news_items:
                                            from_forward_ob = True
                            except (json.JSONDecodeError, TypeError, KeyError) as je:
                                logger.debug(
                                    f"zssm_explain: parse multimsg json in get_msg failed: {je}"
                                )
            except Exception:
                pass
            if agg_texts or agg_imgs:
                logger.info("zssm_explain: fetched origin via get_msg")
                return ("\n".join([x for x in agg_texts if x]).strip(), agg_imgs, from_forward_ob)
        except Exception as e:
            logger.warning(f"zssm_explain: get_msg failed: {e}")

    logger.info("zssm_explain: reply component found but no embedded origin; consider platform API to fetch by id")
    return (None, [], False)
