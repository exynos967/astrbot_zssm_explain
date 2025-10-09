from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict
import os

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
import re

# === 可编辑的默认提示词（用户可直接在此处修改） ===
DEFAULT_SYSTEM_PROMPT = "你是一个中文助理，擅长从被引用的消息中提炼含义、意图和注意事项。"

DEFAULT_TEXT_USER_PROMPT = (
    "请解释这条被回复的消息的含义，输出简洁不超过100字。\n"
    "原始文本：\n{text}"
)

DEFAULT_IMAGE_USER_PROMPT = (
    "请解释这条被回复的消息/图片的含义，输出简洁不超过100字。\n"
    "{text_block}\n包含图片：若无法直接读取图片，请结合上下文或文件名描述。"
)


@register(
    "zssm_explain",
    "薄暝",
    "zssm，回复消息或关键词触发；支持“zssm + 内容”直接解释；引用+@ 场景；Napcat get_msg 回溯图片；按 Provider ID 选择文本/图片模型并带回退；未回复仅发 zssm 时提示；默认提示词可在 main.py 顶部修改",
    "0.2.1",
    "https://github.com/xiaoxi68/astrbot_zssm_explain",
)
class ZssmExplain(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = config or {}

    async def initialize(self):
        """可选：插件初始化。"""

    @staticmethod
    def _extract_text_and_images_from_chain(chain: List[object]) -> Tuple[str, List[str]]:
        """从一段消息链中提取纯文本与图片地址/路径。"""
        texts: List[str] = []
        images: List[str] = []
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
            except Exception as e:
                logger.warning(f"zssm_explain: parse chain segment failed: {e}")
        return ("\n".join([t for t in texts if t]).strip(), images)

    @staticmethod
    def _try_extract_from_reply_component(reply_comp: object) -> Tuple[Optional[str], List[str]]:
        """尽量从 Reply 组件中得到被引用消息的文本与图片。"""
        for attr in ("message", "origin", "content"):
            try:
                payload = getattr(reply_comp, attr, None)
                if isinstance(payload, list):
                    return ZssmExplain._extract_text_and_images_from_chain(payload)
            except Exception:
                pass
        return (None, [])

    @staticmethod
    def _get_reply_message_id(reply_comp: object) -> Optional[str]:
        """从 Reply 组件中尽力获取原消息的 message_id（OneBot/Napcat 常见为 id）。"""
        for key in ("id", "message_id", "reply_id", "messageId", "message_seq"):
            try:
                val = getattr(reply_comp, key, None)
                if isinstance(val, (str, int)) and str(val):
                    return str(val)
            except Exception:
                pass
        try:
            data = getattr(reply_comp, "data", None)
            if isinstance(data, dict):
                for key in ("id", "message_id", "reply", "messageId", "message_seq"):
                    val = data.get(key)
                    if isinstance(val, (str, int)) and str(val):
                        return str(val)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_from_onebot_message_payload(payload: Any) -> Tuple[str, List[str]]:
        """从 OneBot/Napcat get_msg 返回的 payload 中提取文本与图片。"""
        texts: List[str] = []
        images: List[str] = []
        if isinstance(payload, dict):
            msg = payload.get("message")
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
                    except Exception as e:
                        logger.warning(f"zssm_explain: parse onebot segment failed: {e}")
                return ("\n".join([t for t in texts if t]).strip(), images)
            elif isinstance(msg, str) and msg:
                texts.append(msg)
                return ("\n".join(texts).strip(), images)
            raw = payload.get("raw_message")
            if isinstance(raw, str) and raw:
                texts.append(raw)
                return ("\n".join(texts).strip(), images)
        try:
            return (str(payload), images)
        except Exception:
            return ("", images)

    @staticmethod
    def _filter_supported_images(images: List[str]) -> List[str]:
        """只保留看起来可被 LLM 读取的图片：以 http(s) 开头或本地存在的绝对路径。"""
        ok: List[str] = []
        for x in images:
            try:
                if isinstance(x, str) and x:
                    lx = x.lower()
                    if lx.startswith("http://") or lx.startswith("https://"):
                        ok.append(x)
                    elif os.path.isabs(x) and os.path.exists(x):
                        ok.append(x)
            except Exception:
                pass
        return ok

    def _provider_supports_image(self, provider: Any) -> bool:
        """尽力判断 Provider 是否支持图片/多模态。"""
        try:
            mods = getattr(provider, "modalities", None)
            if isinstance(mods, (list, tuple)):
                ml = [str(m).lower() for m in mods]
                if any(k in ml for k in ["image", "vision", "multimodal", "vl", "picture"]):
                    return True
        except Exception:
            pass
        # 一些 Provider 将信息挂在 config/model_config
        for attr in ("config", "model_config", "model"):
            try:
                val = getattr(provider, attr, None)
                if isinstance(val, dict):
                    text = str(val)
                else:
                    text = str(val)
                lt = text.lower()
                if any(k in lt for k in ["image", "vision", "multimodal", "vl", "gpt-4o", "gemini", "minicpm-v"]):
                    return True
            except Exception:
                pass
        return False

    def _select_caption_provider(self, event: AstrMessageEvent, prefer: Any) -> Any:
        """选择可用于图片转述的 Provider：
        1) 优先使用插件配置的 image_provider_id；
        2) 其次使用当前会话 Provider（若具备图片能力）；
        3) 再次在全部 Provider 中查找具备图片能力的第一个。
        """
        # 1) 配置优先
        try:
            cfg_img = self._get_config_provider("image_provider_id")
            if cfg_img is not None:
                return cfg_img
        except Exception:
            pass
        # 2) 当前会话 Provider
        try:
            if self._provider_supports_image(prefer):
                return prefer
        except Exception:
            pass
        # 3) 全部 Provider 中查找
        try:
            for p in self.context.get_all_providers():
                if p is prefer:
                    continue
                if self._provider_supports_image(p):
                    return p
        except Exception:
            pass
        return prefer

    async def _extract_quoted_payload(self, event: AstrMessageEvent) -> Tuple[Optional[str], List[str]]:
        """从当前事件中获取被回复消息的文本与图片。
        优先：Reply 携带嵌入消息；回退：OneBot get_msg；失败：(None, [])。
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
            return (None, [])

        text, images = self._try_extract_from_reply_component(reply_comp)
        if text or images:
            return (text, images)

        reply_id = self._get_reply_message_id(reply_comp)
        platform_name = None
        try:
            platform_name = event.get_platform_name()
        except Exception:
            platform_name = None

        if reply_id and platform_name == "aiocqhttp" and hasattr(event, "bot"):
            try:
                ret: Dict[str, Any] = await event.bot.api.call_action("get_msg", message_id=reply_id)
                t2, imgs2 = self._extract_from_onebot_message_payload(ret)
                if t2 or imgs2:
                    logger.info("zssm_explain: fetched origin via get_msg")
                    return (t2, imgs2)
            except Exception as e:
                logger.warning(f"zssm_explain: get_msg failed: {e}")

        logger.info("zssm_explain: reply component found but no embedded origin; consider platform API to fetch by id")
        return (None, [])

    def _build_user_prompt(self, text: Optional[str], images: List[str]) -> str:
        """仅使用文件顶部的默认常量构建用户提示词，不再读取配置。"""
        text_block = ("原始文本:\n" + text) if text else ""
        if images:
            tmpl = DEFAULT_IMAGE_USER_PROMPT
        else:
            tmpl = DEFAULT_TEXT_USER_PROMPT
        return tmpl.format(text=text or "", text_block=text_block)

    def _build_system_prompt(self) -> str:
        """仅使用文件顶部的默认系统提示词，不再读取配置。"""
        return DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _is_zssm_trigger(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.strip()
        # 忽略常见前缀：/ ! ！ . 。 、 ， - 等，匹配起始处 zssm
        if re.match(r"^[\s/!！。\.、，\-]*zssm(\s|$)", t, re.I):
            return True
        return False

    @staticmethod
    def _first_plain_head_text(chain: List[object]) -> str:
        """返回消息链中最靠前且非空的 Plain 文本。忽略 Reply、At 等非文本段。"""
        if not isinstance(chain, list):
            return ""
        for seg in chain:
            try:
                if isinstance(seg, Comp.Plain):
                    txt = getattr(seg, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt
            except Exception:
                continue
        return ""

    @staticmethod
    def _strip_trigger_and_get_content(text: str) -> str:
        """剥离前缀与 zssm 触发词，返回其后的内容；无内容则返回空串。"""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        m = re.match(r"^[\s/!！。\.、，\-]*zssm(?:\s+(.+))?$", t, re.I)
        if not m:
            return ""
        content = m.group(1) or ""
        return content.strip()

    def _get_inline_content(self, event: AstrMessageEvent) -> str:
        """从消息首个 Plain 文本或整体纯文本中提取 'zssm xxx' 的 xxx 内容。"""
        try:
            chain = event.get_messages()
        except Exception:
            chain = getattr(event.message_obj, "message", []) if hasattr(event, "message_obj") else []
        head = self._first_plain_head_text(chain)
        if head:
            c = self._strip_trigger_and_get_content(head)
            if c:
                return c
        try:
            s = event.get_message_str()
        except Exception:
            s = getattr(event, "message_str", "") or ""
        return self._strip_trigger_and_get_content(s)

    def _pick_llm_text(self, llm_resp: object) -> str:
        # 1) 优先解析 AstrBot 的结果链（MessageChain）
        try:
            rc = getattr(llm_resp, "result_chain", None)
            chain = getattr(rc, "chain", None)
            if isinstance(chain, list) and chain:
                parts: List[str] = []
                for seg in chain:
                    try:
                        txt = getattr(seg, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                    except Exception:
                        pass
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass

        # 2) 常见直接字段
        for attr in ("completion_text", "text", "content", "message"):
            try:
                val = getattr(llm_resp, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            except Exception:
                pass

        # 3) 原始补全（OpenAI 风格）
        try:
            rawc = getattr(llm_resp, "raw_completion", None)
            if rawc is not None:
                choices = getattr(rawc, "choices", None)
                if choices is None and isinstance(rawc, dict):
                    choices = rawc.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or {}
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                    else:
                        text = getattr(first, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text.strip()
        except Exception:
            pass

        # 4) 顶层 choices 兜底
        try:
            choices = getattr(llm_resp, "choices", None)
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                else:
                    text = getattr(first, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        except Exception:
            pass

        # 最终兜底：避免打印对象 repr
        return "（未解析到可读内容）"

    def _get_config_provider(self, key: str) -> Optional[Any]:
        """根据插件配置项（text_provider_id / image_provider_id）返回 Provider 实例。"""
        try:
            pid = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(pid, str):
                pid = pid.strip()
            if pid:
                try:
                    return self.context.get_provider_by_id(provider_id=pid)
                except Exception as e:
                    logger.warning(f"zssm_explain: provider id not found for {key}={pid}: {e}")
        except Exception:
            pass
        return None

    @filter.command("zssm", alias={"知识说明", "解释"})
    async def zssm(self, event: AstrMessageEvent):
        """解释被回复消息：/zssm 或关键词触发；若携带内容则直接解释该内容，否则按回复消息逻辑。"""
        inline = self._get_inline_content(event)
        if inline:
            text, images = inline, []
        else:
            text, images = await self._extract_quoted_payload(event)
            if not text and not images:
                # 未携带被回复内容时的提示
                yield event.plain_result("请输入要解释的内容。")
                return

        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None

        if not provider:
            yield event.plain_result("未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。")
            return

        user_prompt = self._build_user_prompt(text, images)
        system_prompt = self._build_system_prompt()
        image_urls = self._filter_supported_images(images)

        try:
            # 决定首选 Provider：配置优先
            call_provider = provider
            if image_urls:
                cfg_img = self._get_config_provider("image_provider_id")
                if cfg_img is not None:
                    call_provider = cfg_img
                else:
                    # 若未配置，优先使用具备 vision 能力的 Provider
                    call_provider = provider if self._provider_supports_image(provider) else self._select_caption_provider(event, provider)
            else:
                cfg_txt = self._get_config_provider("text_provider_id")
                if cfg_txt is not None:
                    call_provider = cfg_txt

            # 首次尝试
            try:
                llm_resp = await call_provider.text_chat(
                    prompt=user_prompt,
                    context=[],
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                )
            except Exception as e1:
                llm_resp = None
                # 回退路径：
                tried = {id(call_provider)}
                # 1) 若为图片场景，尝试其他 vision Provider
                if image_urls:
                    try:
                        # 优先尝试当前会话 Provider（若未作为首选）
                        if provider and id(provider) not in tried and self._provider_supports_image(provider):
                            tried.add(id(provider))
                            try:
                                llm_resp = await provider.text_chat(
                                    prompt=user_prompt,
                                    context=[],
                                    system_prompt=system_prompt,
                                    image_urls=image_urls,
                                )
                            except Exception:
                                pass
                        # 再尝试其余 vision Provider
                        if llm_resp is None:
                            for p in self.context.get_all_providers():
                                if id(p) in tried:
                                    continue
                                tried.add(id(p))
                                if not self._provider_supports_image(p):
                                    continue
                                try:
                                    llm_resp = await p.text_chat(
                                        prompt=user_prompt,
                                        context=[],
                                        system_prompt=system_prompt,
                                        image_urls=image_urls,
                                    )
                                    logger.info("zssm_explain: fallback vision provider succeeded")
                                    break
                                except Exception:
                                    continue
                    except Exception:
                        pass
                else:
                    # 文本场景：尝试当前会话 Provider（若非首选）
                    if provider and id(provider) not in tried:
                        tried.add(id(provider))
                        try:
                            llm_resp = await provider.text_chat(
                                prompt=user_prompt,
                                context=[],
                                system_prompt=system_prompt,
                                image_urls=[],
                            )
                        except Exception:
                            pass

                if llm_resp is None:
                    raise e1

            reply_text = self._pick_llm_text(llm_resp)
            yield event.plain_result(reply_text)
        except Exception as e:
            logger.error(f"zssm_explain: LLM 调用失败: {e}")
            # 文字场景或所有回退均失败
            yield event.plain_result("解释失败：LLM 或图片转述模型调用异常，请稍后再试或联系管理员。")

    async def terminate(self):
        return

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def keyword_zssm(self, event: AstrMessageEvent):
        """关键词触发：忽略常见前缀/Reply/At 等，检测首个 Plain 段的 zssm。
        避免与 /zssm 指令重复：若以 /zssm 开头则交由指令处理。
        """
        # 优先使用消息链首个 Plain 段判断
        try:
            chain = event.get_messages()
        except Exception:
            chain = getattr(event.message_obj, "message", []) if hasattr(event, "message_obj") else []
        head = self._first_plain_head_text(chain)
        # 指令冲突：首个 Plain 段是 /zssm 则交由指令处理
        if isinstance(head, str) and head.strip():
            if re.match(r"^\s*/\s*zssm(\s|$)", head.strip(), re.I):
                return
            if self._is_zssm_trigger(head):
                async for r in self.zssm(event):
                    yield r
                return
        # 回退到纯文本串
        try:
            text = event.get_message_str()
        except Exception:
            text = getattr(event, "message_str", "") or ""
        if isinstance(text, str) and text.strip():
            t = text.strip()
            if re.match(r"^\s*/\s*zssm(\s|$)", t, re.I):
                return
            if self._is_zssm_trigger(t):
                async for r in self.zssm(event):
                    yield r
