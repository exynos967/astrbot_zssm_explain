from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Set, Union
import os
import asyncio
import re
import shutil
import math
import time

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.core.star.star_handler import EventType
from astrbot.core.pipeline.context_utils import call_event_hook

from .url_utils import (
    fetch_html,
    extract_urls_from_text,
    prepare_url_prompt,
    build_url_failure_message,
    build_url_brief_for_forward,
)
from .message_utils import (
    extract_quoted_payload_with_videos,
    extract_text_images_videos_from_chain,
    call_get_msg,
    ob_data,
    extract_from_onebot_message_payload_with_videos,
)
from .video_utils import (
    extract_videos_from_chain,
    is_http_url,
    is_abs_file,
    napcat_resolve_file_url,
    extract_forward_video_keyframes,
    probe_duration_sec,
    resolve_ffmpeg,
    resolve_ffprobe,
    download_video_to_temp,
    sample_frames_equidistant,
    sample_frames_with_ffmpeg,
    extract_audio_wav,
    get_video_size_mb,
    is_safe_video_path,
    compress_video_to_target,
)
from .bilibili_utils import (
    is_bilibili_url,
    download_bilibili_video_to_temp,
    get_bilibili_url_type,
    resolve_bilibili_content,
    resolve_bilibili_short_url,
)
from .prompt_utils import (
    DEFAULT_URL_USER_PROMPT,
    DEFAULT_VIDEO_USER_PROMPT,
    DEFAULT_FRAME_CAPTION_PROMPT,
    build_user_prompt,
    build_system_prompt_for_event,
)
from .llm_client import LLMClient
from .file_preview_utils import (
    build_text_exts_from_config,
    extract_file_preview_from_reply,
)

"""
默认提示词已集中放在 prompt_utils.py 中：
- DEFAULT_* 常量用于不同流程的默认文案
- build_* 函数用于构造系统/用户提示词
"""

# URL 识别/抓取的默认参数（可通过插件配置覆盖）
URL_DETECT_ENABLE_KEY = "enable_url_detect"
URL_FETCH_TIMEOUT_KEY = "url_timeout_sec"
URL_MAX_CHARS_KEY = "url_max_chars"
KEYWORD_ZSSM_ENABLE_KEY = "enable_keyword_zssm"
HE_YI_WEI_ENABLE_KEY = "enable_heyiwei"
EMPTY_ZSSM_PROMPT_ENABLE_KEY = "enable_empty_zssm_prompt"
GROUP_LIST_MODE_KEY = "group_list_mode"
GROUP_LIST_KEY = "group_list"
VIDEO_PROVIDER_ID_KEY = "video_provider_id"
VIDEO_FRAME_INTERVAL_SEC_KEY = "video_frame_interval_sec"
VIDEO_ASR_ENABLE_KEY = "video_asr_enable"
VIDEO_MAX_DURATION_SEC_KEY = "video_max_duration_sec"
VIDEO_MAX_SIZE_MB_KEY = "video_max_size_mb"
FFMPEG_PATH_KEY = "ffmpeg_path"
ASR_PROVIDER_ID_KEY = "asr_provider_id"
CF_SCREENSHOT_ENABLE_KEY = "cf_screenshot_enable"
CF_SCREENSHOT_SIZE_KEY = "cf_screenshot_size"
KEEP_ORIGINAL_PERSONA_KEY = "keep_original_persona"
FILE_PREVIEW_EXTS_KEY = "file_preview_exts"
FILE_PREVIEW_MAX_SIZE_KB_KEY = "file_preview_max_size_kb"
FORWARD_VIDEO_KEYFRAME_ENABLE_KEY = "forward_video_keyframe_enable"
FORWARD_VIDEO_MAX_COUNT_KEY = "forward_video_max_count"

VIDEO_DIRECT_MODE_KEY = "video_direct_mode"
VIDEO_DIRECT_FPS_KEY = "video_direct_fps"
VIDEO_DIRECT_TARGET_SIZE_MB_KEY = "video_direct_target_size_mb"
VIDEO_DIRECT_TIMEOUT_SEC_KEY = "video_direct_timeout_sec"
BILIBILI_COOKIE_KEY = "bilibili_cookie"

DEFAULT_URL_DETECT_ENABLE = True
DEFAULT_URL_FETCH_TIMEOUT = 20
DEFAULT_URL_MAX_CHARS = 6000
DEFAULT_EMPTY_ZSSM_PROMPT_ENABLE = False
DEFAULT_VIDEO_FRAME_INTERVAL_SEC = 6
DEFAULT_VIDEO_ASR_ENABLE = False
DEFAULT_VIDEO_MAX_DURATION_SEC = 120
DEFAULT_VIDEO_MAX_SIZE_MB = 50
DEFAULT_FFMPEG_PATH = "ffmpeg"
DEFAULT_CF_SCREENSHOT_ENABLE = True
DEFAULT_CF_SCREENSHOT_SIZE = "1280x720"
DEFAULT_KEEP_ORIGINAL_PERSONA = True
DEFAULT_FILE_PREVIEW_EXTS = "txt,md,log,json,csv,ini,cfg,yml,yaml,py"
DEFAULT_FILE_PREVIEW_MAX_SIZE_KB = 100
DEFAULT_FORWARD_VIDEO_KEYFRAME_ENABLE = True
DEFAULT_FORWARD_VIDEO_MAX_COUNT = 2
TRIGGER_KEYWORDS_KEY = "trigger_keywords"
COMMAND_TRIGGER_KEYWORD = "zssm"
DEFAULT_EXTRA_TRIGGER_KEYWORDS = ("hyw", "何意味")
HE_YI_WEI_TRIGGER_KEYWORDS = frozenset(
    kw.lower() for kw in DEFAULT_EXTRA_TRIGGER_KEYWORDS
)


class ZssmExplain(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = config or {}
        self._last_fetch_info: Dict[str, Any] = {}
        self._llm = LLMClient(
            context=self.context,
            get_conf_int=self._get_conf_int,
            get_config_provider=self._get_config_provider,
            logger=logger,
        )

    async def initialize(self):
        """可选：插件初始化。"""
        self._migrate_legacy_trigger_keywords_config()

    def _reply_text_result(self, event: AstrMessageEvent, text: str):
        """构造一个显式“回复调用者”的文本消息结果。

        优先使用 Reply 组件引用当前事件的 message_id，无法获取或平台不支持时
        回退为普通纯文本结果，确保跨平台健壮性。
        """
        try:
            msg_id = None
            try:
                msg_id = getattr(event.message_obj, "message_id", None)
            except Exception:
                msg_id = None
            if msg_id:
                try:
                    chain = [
                        Comp.Reply(id=str(msg_id)),
                        Comp.Plain(str(text) if text is not None else ""),
                    ]
                    return event.chain_result(chain)
                except Exception:
                    # 某些平台或适配器不支持 Reply 组件
                    pass
            return event.plain_result(str(text) if text is not None else "")
        except Exception:
            return event.plain_result(str(text) if text is not None else "")

    def _format_llm_error(self, e: Exception, context: str = "") -> str:
        """将 LLM 调用异常转换为用户友好的错误提示。"""
        err_str = str(e)
        prefix = f"{context}失败：" if context else "LLM 调用失败："
        if "Connection error" in err_str or "ConnectionError" in err_str:
            return f"{prefix}LLM 服务连接失败，请检查网络或代理设置。"
        if "timeout" in err_str.lower() or "Timeout" in err_str:
            return f"{prefix}LLM 服务响应超时，请稍后重试。"
        if (
            "401" in err_str
            or "Unauthorized" in err_str
            or "invalid_api_key" in err_str
        ):
            return f"{prefix}LLM API Key 无效或已过期。"
        if "429" in err_str or "rate_limit" in err_str.lower():
            return f"{prefix}LLM 请求频率超限，请稍后重试。"
        if "all providers failed" in err_str:
            return f"{prefix}所有 LLM 服务均不可用，请检查配置。"
        return f"{prefix}LLM 调用出错，请查看日志。"

    # ===== 群聊权限控制 =====
    def _get_conf_str(self, key: str, default: str) -> str:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, str):
                return v.strip()
        except Exception:
            pass
        return default

    def _get_bilibili_cookie(self) -> Optional[str]:
        """从配置中获取 B 站 Cookie 字符串。

        配置格式为 object: {"SESSDATA": "xxx", "bili_jct": "xxx"}
        返回格式为 "SESSDATA=xxx; bili_jct=xxx"，若未配置则返回 None。
        """
        try:
            cookie_cfg = self.config.get(BILIBILI_COOKIE_KEY)
            if not isinstance(cookie_cfg, dict):
                return None
            sessdata = cookie_cfg.get("SESSDATA", "").strip()
            bili_jct = cookie_cfg.get("bili_jct", "").strip()
            if not sessdata:
                return None
            parts = [f"SESSDATA={sessdata}"]
            if bili_jct:
                parts.append(f"bili_jct={bili_jct}")
            return "; ".join(parts)
        except Exception:
            return None

    def _get_conf_list_str(self, key: str) -> List[str]:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, list):
                out: List[str] = []
                for it in v:
                    if isinstance(it, (str, int)):
                        s = str(it).strip()
                        if s:
                            out.append(s)
                return out
            if isinstance(v, str) and v.strip():
                # 兼容以逗号/空白/中文逗号/顿号分隔的字符串
                raw = [x.strip() for x in re.split(r"[\s,，、]+", v) if x.strip()]
                return raw
        except Exception:
            pass
        return []

    @staticmethod
    def _normalize_trigger_keyword(keyword: object) -> str:
        if not isinstance(keyword, (str, int)):
            return ""
        return str(keyword).strip().lower()

    def _get_configured_trigger_keywords(self) -> List[str]:
        # 仅当配置项缺失时回退默认；显式配置空列表表示禁用额外关键词。
        if isinstance(self.config, dict) and TRIGGER_KEYWORDS_KEY in self.config:
            raw_keywords = self._get_conf_list_str(TRIGGER_KEYWORDS_KEY)
        else:
            raw_keywords = list(DEFAULT_EXTRA_TRIGGER_KEYWORDS)

        keywords: List[str] = []
        seen: Set[str] = set()
        for raw_keyword in raw_keywords:
            keyword = self._normalize_trigger_keyword(raw_keyword)
            if (
                not keyword
                or keyword == COMMAND_TRIGGER_KEYWORD
                or keyword in seen
            ):
                continue
            keywords.append(keyword)
            seen.add(keyword)
        return keywords

    def _get_effective_trigger_keywords(self) -> List[str]:
        return [COMMAND_TRIGGER_KEYWORD, *self._get_configured_trigger_keywords()]

    @staticmethod
    def _build_trigger_keyword_pattern(keywords: List[str]) -> str:
        escaped = sorted((re.escape(keyword) for keyword in keywords), key=len, reverse=True)
        return "|".join(escaped)

    def _get_trigger_keyword_pattern(self) -> str:
        return self._build_trigger_keyword_pattern(self._get_effective_trigger_keywords())

    def _migrate_legacy_trigger_keywords_config(self) -> None:
        if not isinstance(self.config, dict):
            return
        if self._get_conf_bool(HE_YI_WEI_ENABLE_KEY, True):
            return

        configured_keywords = self._get_configured_trigger_keywords()
        default_keywords = [
            self._normalize_trigger_keyword(keyword)
            for keyword in DEFAULT_EXTRA_TRIGGER_KEYWORDS
        ]
        if configured_keywords != default_keywords:
            return

        self.config[TRIGGER_KEYWORDS_KEY] = []
        self.config[HE_YI_WEI_ENABLE_KEY] = True
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            try:
                save_config()
            except Exception as exc:
                logger.warning(
                    "zssm_explain: failed to persist legacy trigger keyword migration: %s",
                    exc,
                )
        logger.info(
            "zssm_explain: migrated legacy enable_heyiwei=false to trigger_keywords=[]"
        )

    def _is_group_allowed(self, event: AstrMessageEvent) -> bool:
        """根据配置的白/黑名单判断是否允许在该群聊中使用插件。
        模式：
        - whitelist：仅允许在列表内群使用
        - blacklist：拒绝列表内群使用
        - none：不限制
        当无法获取 group_id 时（如私聊），默认放行。

        兼容性增强：
        - 优先使用 event.get_group_id()
        - 回退：群聊场景尝试从 session_id/unified_msg_origin 解析
        - 再回退：尝试从原始 message_obj.group_id 获取
        """
        gid = None
        # 1) 官方获取
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        # 2) 从会话上下文回退解析（仅群聊）
        if not gid:
            try:
                mt = getattr(event, "get_message_type", None)
                mt = mt() if callable(mt) else None
                mt_str = str(mt).lower() if mt is not None else ""
                if "group" in mt_str or "guild" in mt_str:
                    sid = None
                    try:
                        sid = event.get_session_id()
                    except Exception:
                        sid = None
                    if isinstance(sid, str) and sid.strip():
                        gid = sid.strip()
            except Exception:
                pass
        # 3) 从原始消息对象回退
        if not gid:
            try:
                gid = getattr(getattr(event, "message_obj", None), "group_id", None)
            except Exception:
                gid = None
        if not gid:
            return True  # 非群聊或无法识别，放行

        mode = self._get_conf_str(GROUP_LIST_MODE_KEY, "none").lower()
        if mode not in ("whitelist", "blacklist", "none"):
            mode = "none"
        glist = self._get_conf_list_str(GROUP_LIST_KEY)

        if mode == "whitelist":
            return str(gid).strip() in glist if glist else False
        if mode == "blacklist":
            return str(gid).strip() not in glist if glist else True
        return True

    def _resolve_ffmpeg(self) -> Optional[str]:
        """根据插件配置解析 ffmpeg 路径，委托 video_utils.resolve_ffmpeg。"""
        cfg_path = self._get_conf_str(FFMPEG_PATH_KEY, DEFAULT_FFMPEG_PATH)
        return resolve_ffmpeg(cfg_path, DEFAULT_FFMPEG_PATH)

    def _resolve_ffprobe(self) -> Optional[str]:
        """根据已解析的 ffmpeg 路径解析 ffprobe，委托 video_utils.resolve_ffprobe。"""
        ff = self._resolve_ffmpeg()
        return resolve_ffprobe(ff)

    def _build_video_user_prompt(
        self, meta: Dict[str, Any], asr_text: Optional[str]
    ) -> str:
        # 始终使用代码常量模板，不从配置读取
        tmpl = DEFAULT_VIDEO_USER_PROMPT
        meta_items = []
        name = meta.get("name")
        if name:
            meta_items.append(f"视频: {name}")
        dur = meta.get("duration")
        if isinstance(dur, (int, float)):
            meta_items.append(f"时长: {int(dur)}s")
        fcnt = meta.get("frames")
        if isinstance(fcnt, int):
            meta_items.append(f"关键帧: {fcnt} 张")
        meta_block = "\n".join(meta_items)
        asr_block = (
            f"音频转写要点: \n{asr_text.strip()}"
            if isinstance(asr_text, str) and asr_text.strip()
            else ""
        )
        return tmpl.format(meta_block=meta_block, asr_block=asr_block)

    def _build_video_final_prompt(
        self, meta: Dict[str, Any], asr_text: Optional[str], captions: List[str]
    ) -> str:
        """构造最终汇总提示词：基于逐帧描述 + （可选）音频要点。"""
        meta_items = []
        name = meta.get("name")
        if name:
            meta_items.append(f"视频: {name}")
        dur = meta.get("duration")
        if isinstance(dur, (int, float)):
            meta_items.append(f"时长: {int(dur)}s")
        fcnt = meta.get("frames")
        if isinstance(fcnt, int):
            meta_items.append(f"关键帧: {fcnt} 张")
        meta_block = "\n".join(meta_items)
        caps_block = "\n".join(
            [f"- {c.strip()}" for c in captions if isinstance(c, str) and c.strip()]
        )
        asr_block = (
            f"音频转写要点: \n{asr_text.strip()}"
            if isinstance(asr_text, str) and asr_text.strip()
            else ""
        )
        final_prompt = (
            "请根据以下关键帧描述与最后一张关键帧图片，总结整段视频的主要内容（中文，不超过100字）。"
            "仅依据已给信息，信息不足请说明‘无法判断’，不要编造未出现的内容。\n"
            f"{meta_block}\n关键帧描述：\n{caps_block}\n{asr_block}"
        )
        return final_prompt

    def _choose_stt_provider(self, event: AstrMessageEvent) -> Optional[Any]:
        """根据配置选择 STT（ASR）提供商：优先 asr_provider_id；否则使用当前会话 STT。"""
        pid = None
        try:
            pid = (
                self.config.get(ASR_PROVIDER_ID_KEY)
                if isinstance(self.config, dict)
                else None
            )
            if isinstance(pid, str):
                pid = pid.strip()
        except Exception:
            pid = None
        # 优先根据 ID 在已注册 STT 中匹配
        if pid:
            try:
                stts = self.context.get_all_stt_providers()
            except Exception:
                stts = []
            pid_l = pid.lower()
            for p in stts or []:
                try:
                    candidates = []
                    for attr in ("id", "provider_id", "name"):
                        val = getattr(p, attr, None)
                        if isinstance(val, str) and val:
                            candidates.append(val)
                    cfg = getattr(p, "provider_config", None)
                    if isinstance(cfg, dict):
                        for k in ("id", "provider_id", "name"):
                            v = cfg.get(k)
                            if isinstance(v, str) and v:
                                candidates.append(v)
                    if any(str(c).lower() == pid_l for c in candidates):
                        return p
                except Exception:
                    continue
        # 回退为当前会话 STT
        try:
            return self.context.get_using_stt_provider(umo=event.unified_msg_origin)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # B站图文/动态/直播解释（非视频内容）
    # -------------------------------------------------------------------------

    async def _explain_bilibili(
        self,
        event: AstrMessageEvent,
        bili_url: str,
        bili_type: str,
    ):
        """解释 B 站非视频内容（动态/直播/专栏/图文），将图片发给模型。"""
        logger.info(
            "zssm_explain: bilibili content type=%s url=%s",
            bili_type,
            bili_url[:100] if bili_url else "",
        )

        type_name_map = {
            "dynamic": "动态",
            "live": "直播",
            "read": "专栏",
            "opus": "图文",
        }
        type_name = type_name_map.get(bili_type, "内容")

        # 获取 B 站 Cookie
        bili_cookie = self._get_bilibili_cookie()

        # 解析 B 站内容
        try:
            content = await resolve_bilibili_content(bili_url, cookie=bili_cookie)
        except Exception as e:
            logger.warning("zssm_explain: bilibili resolve failed: %s", e)
            content = None

        # API 解析失败时，回退到网页截图方式
        if not content:
            # 根据是否配置了 Cookie 给出不同提示
            if not bili_cookie:
                logger.info(
                    "zssm_explain: bilibili API failed (no cookie), fallback to screenshot"
                )
            else:
                logger.info(
                    "zssm_explain: bilibili API failed (with cookie), fallback to screenshot"
                )
            timeout_sec = self._get_conf_int(
                URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60
            )
            max_chars = self._get_conf_int(
                URL_MAX_CHARS_KEY, DEFAULT_URL_MAX_CHARS, min_v=1000, max_v=50000
            )
            cf_enable = self._get_conf_bool(
                CF_SCREENSHOT_ENABLE_KEY, DEFAULT_CF_SCREENSHOT_ENABLE
            )
            width, height = self._get_cf_screenshot_size()
            url_ctx = await prepare_url_prompt(
                bili_url,
                timeout_sec,
                self._last_fetch_info,
                max_chars=max_chars,
                cf_screenshot_enable=cf_enable,
                cf_screenshot_width=width,
                cf_screenshot_height=height,
                file_preview_max_bytes=self._get_file_preview_max_bytes(),
                user_prompt_template=f"请解释以下 B 站{type_name}的内容：\n{{text}}",
            )
            if url_ctx:
                user_prompt, _text, images = url_ctx
                # 获取 provider
                try:
                    provider = self.context.get_using_provider(
                        umo=event.unified_msg_origin
                    )
                except Exception as e:
                    logger.error(f"zssm_explain: get provider failed: {e}")
                    provider = None

                if not provider:
                    yield self._reply_text_result(
                        event,
                        "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。",
                    )
                    return

                image_urls = self._llm.filter_supported_images(images)
                system_prompt = await self._build_system_prompt(event)

                try:
                    start_ts = time.perf_counter()
                    call_provider = self._llm.select_primary_provider(
                        session_provider=provider, image_urls=image_urls
                    )
                    llm_resp = await self._llm.call_with_fallback(
                        primary=call_provider,
                        session_provider=provider,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        image_urls=image_urls,
                    )
                    reply_text = self._llm.pick_llm_text(llm_resp)
                    elapsed = time.perf_counter() - start_ts
                    reply_text = self._format_explain_output(
                        reply_text, elapsed_sec=elapsed
                    )
                    yield self._reply_text_result(event, reply_text)
                    return
                except Exception as e:
                    logger.error("zssm_explain: bilibili screenshot LLM failed: %s", e)

            # 根据是否配置了 Cookie 给出不同的错误提示
            if not bili_cookie:
                yield self._reply_text_result(
                    event,
                    f"无法解析该 B 站{type_name}链接（API 风控需要 Cookie，请在插件配置中填写 B站 Cookie）。",
                )
            else:
                yield self._reply_text_result(
                    event,
                    f"无法解析该 B 站{type_name}链接（API 解析失败，截图也失败了）。",
                )
            return

        # 构建提示词
        title = content.get("title") or ""
        text = content.get("text") or ""
        author = content.get("author") or ""
        images = content.get("images") or []

        prompt_parts = [f"请解释以下 B 站{type_name}的内容："]
        if title:
            prompt_parts.append(f"标题：{title}")
        if author:
            prompt_parts.append(f"作者：{author}")
        if text:
            prompt_parts.append(f"正文：\n{text[:2000]}")
        if images:
            prompt_parts.append(f"（附带 {len(images)} 张图片）")

        user_prompt = "\n".join(prompt_parts)

        # 获取 provider
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None

        if not provider:
            yield self._reply_text_result(
                event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。"
            )
            return

        # 过滤支持的图片
        image_urls = self._llm.filter_supported_images(images)
        system_prompt = await self._build_system_prompt(event)

        try:
            start_ts = time.perf_counter()
            call_provider = self._llm.select_primary_provider(
                session_provider=provider, image_urls=image_urls
            )
            llm_resp = await self._llm.call_with_fallback(
                primary=call_provider,
                session_provider=provider,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image_urls=image_urls,
            )

            reply_text = self._llm.pick_llm_text(llm_resp)
            elapsed = time.perf_counter() - start_ts
            reply_text = self._format_explain_output(reply_text, elapsed_sec=elapsed)
            yield self._reply_text_result(event, reply_text)

        except Exception as e:
            logger.error("zssm_explain: bilibili LLM call failed: %s", e)
            yield self._reply_text_result(
                event, self._format_llm_error(e, f"解释 B 站{type_name}")
            )

    async def _explain_video(
        self,
        event: AstrMessageEvent,
        video_src: str,
        *,
        video_meta: Optional[dict] = None,
    ):
        # 配置检查：未配置 video_provider_id 时视为未启用视频解释
        cfg_video_provider = self._get_config_provider(VIDEO_PROVIDER_ID_KEY)
        if cfg_video_provider is None:
            yield self._reply_text_result(
                event,
                "视频解释未配置可用的模型提供商，请在插件配置中选择 video_provider_id 后再试。",
            )
            return

        direct_mode = self._get_conf_str(VIDEO_DIRECT_MODE_KEY, "off").lower()
        if direct_mode not in ("off", "base64", "dashscope"):
            direct_mode = "off"

        # 非直传模式下，ffmpeg 是必须的；直传模式下 ffmpeg 仅用于回退
        ffmpeg_path = self._resolve_ffmpeg()
        if direct_mode == "off" and not ffmpeg_path:
            yield self._reply_text_result(
                event,
                "未检测到 ffmpeg，请安装系统 ffmpeg 或 Python 包 imageio-ffmpeg，并在插件配置中设置 ffmpeg_path。",
            )
            return

        logger.info(
            "zssm_explain: video start src=%s ffmpeg=%s direct_mode=%s",
            (str(video_src)[:128] if video_src else ""),
            ffmpeg_path,
            direct_mode,
        )

        # 统一获取本地文件路径（支持 http/https 下载，Napcat 直链解析）
        max_mb = self._get_conf_int(
            VIDEO_MAX_SIZE_MB_KEY, DEFAULT_VIDEO_MAX_SIZE_MB, 1, 512
        )
        local_path = None
        src = video_src
        # 优先特判 B 站视频链接：通过 video_utils 解析并下载到临时文件
        if isinstance(src, str) and is_http_url(src) and is_bilibili_url(src):
            try:
                bili_local = await download_bilibili_video_to_temp(src, max_mb)
            except ValueError as ve:
                # 大小超限，给出明确提示
                yield self._reply_text_result(event, str(ve))
                return
            except Exception as e:
                logger.warning("zssm_explain: bilibili download failed: %s", e)
                bili_local = None
            if bili_local:
                local_path = bili_local
            else:
                # B 站链接解析失败时直接给出友好提示，避免误将网页当作视频文件处理
                yield self._reply_text_result(
                    event,
                    "暂时无法解析该 B 站视频链接，请确认视频为公开可访问状态，或改为发送视频文件/截图后再试。",
                )
                return

        # 若尚未得到本地路径，再按通用逻辑处理 Napcat/file_id 与普通 http 链接/本地路径
        if local_path is None:
            # 若有 fileUuid，优先尝试 get_group_file_url 拿公网直链
            file_uuid = (video_meta or {}).get("fileUuid")
            if file_uuid and not is_http_url(src) and not is_abs_file(src):
                try:
                    gid = event.get_group_id()
                    if gid:
                        gid_param = (
                            int(gid) if isinstance(gid, str) and gid.isdigit() else gid
                        )
                        ret = await event.bot.api.call_action(
                            "get_group_file_url",
                            group_id=gid_param,
                            file_id=file_uuid,
                        )
                        d = ret.get("data") if isinstance(ret, dict) else None
                        if not isinstance(d, dict):
                            d = ret if isinstance(ret, dict) else {}
                        url = d.get("url")
                        if isinstance(url, str) and url.strip():
                            logger.info(
                                "zssm_explain: get_group_file_url ok => %s",
                                url[:80],
                            )
                            src = url.strip()
                except Exception as e:
                    logger.debug("zssm_explain: get_group_file_url failed: %s", e)
            # 若 src 已经是绝对路径且存在（从 records filePath 拿到的），也优先尝试
            # get_group_file_url 拿公网直链（dashscope 可直接用 http URL）
            elif file_uuid and is_abs_file(src) and direct_mode == "dashscope":
                try:
                    gid = event.get_group_id()
                    if gid:
                        gid_param = (
                            int(gid) if isinstance(gid, str) and gid.isdigit() else gid
                        )
                        ret = await event.bot.api.call_action(
                            "get_group_file_url",
                            group_id=gid_param,
                            file_id=file_uuid,
                        )
                        d = ret.get("data") if isinstance(ret, dict) else None
                        if not isinstance(d, dict):
                            d = ret if isinstance(ret, dict) else {}
                        url = d.get("url")
                        if isinstance(url, str) and url.strip():
                            logger.info(
                                "zssm_explain: get_group_file_url ok (override local) => %s",
                                url[:80],
                            )
                            src = url.strip()
                except Exception as e:
                    logger.debug("zssm_explain: get_group_file_url failed: %s", e)

            # 若不是 URL/绝对路径，尝试通过 Napcat file_id 获取直链
            if (
                isinstance(src, str)
                and (not is_http_url(src))
                and (not is_abs_file(src))
            ):
                try:
                    resolved = await napcat_resolve_file_url(event, src)
                except Exception:
                    resolved = None
                if isinstance(resolved, str) and resolved:
                    src = resolved
            if isinstance(src, str) and is_http_url(src):
                local_path = await download_video_to_temp(src, max_mb)
                if not local_path:
                    yield self._reply_text_result(
                        event, f"视频下载失败或超过大小限制（>{max_mb}MB）。"
                    )
                    return
            else:
                # 假定为本地路径
                if not (
                    isinstance(src, str) and os.path.isabs(src) and os.path.exists(src)
                ):
                    yield self._reply_text_result(
                        event, "无法读取该视频源，请确认路径或链接有效。"
                    )
                    return
                # 大小检查（直传模式跳过，百炼支持大文件）
                try:
                    sz = os.path.getsize(src)
                    if sz > max_mb * 1024 * 1024 and direct_mode == "off":
                        yield self._reply_text_result(
                            event,
                            f"视频大小超过限制（>{max_mb}MB），请压缩或截取片段后重试。",
                        )
                        return
                except Exception:
                    pass
                local_path = src

        # 时长检查（可选，缺少 ffprobe 时跳过）
        max_sec = self._get_conf_int(
            VIDEO_MAX_DURATION_SEC_KEY, DEFAULT_VIDEO_MAX_DURATION_SEC, 10, 3600
        )
        dur = probe_duration_sec(self._resolve_ffprobe(), local_path)
        logger.info(
            "zssm_explain: probed duration=%s (max=%s)",
            dur if dur is not None else "unknown",
            max_sec,
        )
        if isinstance(dur, (int, float)) and dur > max_sec and direct_mode == "off":
            yield self._reply_text_result(
                event, f"视频时长超过限制（>{max_sec}s），请截取片段后重试。"
            )
            return

        # ===== 视频直传分支 =====
        direct_cleanup: List[str] = []
        if direct_mode != "off" and is_safe_video_path(local_path):
            try:
                direct_path = local_path
                size_mb = get_video_size_mb(direct_path)
                target_cfg = self._get_conf_int(
                    VIDEO_DIRECT_TARGET_SIZE_MB_KEY, -1, -1, 500
                )

                if direct_mode == "base64":
                    # base64 模式：限制 10MB
                    limit_mb = 10.0
                    if target_cfg == -1:
                        target_mb = limit_mb
                    elif target_cfg == 0:
                        target_mb = 0.0
                    else:
                        target_mb = min(float(target_cfg), limit_mb)

                    if size_mb > limit_mb and target_mb > 0 and ffmpeg_path:
                        compressed = await compress_video_to_target(
                            ffmpeg_path, direct_path, target_mb
                        )
                        if compressed and compressed != direct_path:
                            direct_cleanup.append(compressed)
                            direct_path = compressed
                            size_mb = get_video_size_mb(direct_path)

                    if size_mb <= limit_mb:
                        logger.info("zssm_explain: video direct base64 %.1fMB", size_mb)
                        system_prompt = await self._build_system_prompt(event)
                        start_ts = time.perf_counter()
                        result = await self._llm.call_video_direct(
                            video_path=direct_path,
                            mode="base64",
                            provider=cfg_video_provider,
                            prompt=self._build_video_user_prompt(
                                {"name": os.path.basename(local_path), "duration": dur},
                                None,
                            ),
                            timeout=self._get_conf_int(
                                VIDEO_DIRECT_TIMEOUT_SEC_KEY, 300, 60, 600
                            ),
                        )
                        elapsed = time.perf_counter() - start_ts
                        result = self._format_explain_output(
                            result, elapsed_sec=elapsed
                        )
                        yield self._reply_text_result(event, result)
                        try:
                            event.stop_event()
                        except Exception:
                            pass
                    else:
                        raise ValueError(
                            f"视频 {size_mb:.1f}MB 超过 base64 限制 {limit_mb:.0f}MB"
                        )

                elif direct_mode == "dashscope":
                    # dashscope 模式：≤100MB 用 file://，>100MB 上传 OSS
                    limit_mb = 100.0
                    if target_cfg == -1:
                        target_mb = limit_mb
                    elif target_cfg == 0:
                        target_mb = 0.0
                    else:
                        target_mb = min(float(target_cfg), limit_mb)

                    if size_mb > limit_mb and target_mb > 0 and ffmpeg_path:
                        compressed = await compress_video_to_target(
                            ffmpeg_path, direct_path, target_mb
                        )
                        if compressed and compressed != direct_path:
                            direct_cleanup.append(compressed)
                            direct_path = compressed
                            size_mb = get_video_size_mb(direct_path)

                    fps = self._get_conf_int(VIDEO_DIRECT_FPS_KEY, 2, 1, 30)
                    logger.info(
                        "zssm_explain: video direct dashscope %.1fMB fps=%d",
                        size_mb,
                        fps,
                    )
                    system_prompt = await self._build_system_prompt(event)
                    start_ts = time.perf_counter()
                    result = await self._llm.call_video_direct(
                        video_path=direct_path,
                        mode="dashscope",
                        provider=cfg_video_provider,
                        prompt=self._build_video_user_prompt(
                            {"name": os.path.basename(local_path), "duration": dur},
                            None,
                        ),
                        fps=fps,
                        timeout=self._get_conf_int(
                            VIDEO_DIRECT_TIMEOUT_SEC_KEY, 300, 60, 600
                        ),
                    )
                    elapsed = time.perf_counter() - start_ts
                    result = self._format_explain_output(result, elapsed_sec=elapsed)
                    yield self._reply_text_result(event, result)
                    try:
                        event.stop_event()
                    except Exception:
                        pass

            except Exception as e:
                logger.warning("zssm_explain: 视频直传失败: %s", e)
                # 使用统一的错误美化，避免泄露内部信息
                yield self._reply_text_result(
                    event, self._format_llm_error(e, "视频直传")
                )
            finally:
                for p in direct_cleanup:
                    try:
                        if isinstance(p, str) and p and os.path.isfile(p):
                            os.remove(p)
                    except Exception:
                        pass

            # 直传模式下不回退抽帧，直接返回
            return

        # 回退抽帧需要 ffmpeg
        if not ffmpeg_path:
            yield self._reply_text_result(
                event,
                "未检测到 ffmpeg，视频直传也未成功，请安装 ffmpeg 或检查直传配置。",
            )
            return

        # 抽帧
        is_gif = False
        try:
            if isinstance(local_path, str) and local_path.lower().endswith(".gif"):
                is_gif = True
        except Exception:
            is_gif = False

        interval = self._get_conf_int(
            VIDEO_FRAME_INTERVAL_SEC_KEY, DEFAULT_VIDEO_FRAME_INTERVAL_SEC, 1, 120
        )
        try:
            if isinstance(dur, (int, float)) and dur > 0:
                # 依据时长与间隔估算目标帧数；GIF 固定抽 1 帧
                if is_gif:
                    n_frames = 1
                else:
                    n_frames = max(1, int(math.ceil(float(dur) / max(1, interval))))
                logger.info(
                    "zssm_explain: sampling plan: duration=%.2fs interval=%ss => target_frames=%s",
                    float(dur),
                    interval,
                    n_frames,
                )
                frames = await sample_frames_equidistant(
                    ffmpeg_path, local_path, float(dur), n_frames
                )
            else:
                # 未获知时长时，以最大允许时长作为上界推导帧数；GIF 固定抽 1 帧
                if is_gif:
                    n_frames = 1
                else:
                    n_frames = max(1, int(math.ceil(float(max_sec) / max(1, interval))))
                logger.info(
                    "zssm_explain: sampling plan: unknown duration, use max_sec=%ss interval=%ss => target_frames=%s",
                    max_sec,
                    interval,
                    n_frames,
                )
                frames = await sample_frames_with_ffmpeg(
                    ffmpeg_path, local_path, interval, n_frames
                )
        except Exception as e:
            logger.error("zssm_explain: frame sampling failed: %s", e)
            yield self._reply_text_result(
                event, "视频抽帧失败，请检查 ffmpeg 配置或更换视频后重试。"
            )
            return
        logger.info("zssm_explain: sampled %d frames", len(frames))
        image_urls = self._llm.filter_supported_images(frames)
        if not image_urls:
            yield self._reply_text_result(
                event, "未能生成可用关键帧，请检查 ffmpeg 或更换视频后重试。"
            )
            return

        # 可选 ASR：由 video_asr_enable + asr_provider_id 控制
        asr_text = None
        asr_enabled = False
        try:
            asr_enabled = self._get_conf_bool(
                VIDEO_ASR_ENABLE_KEY, DEFAULT_VIDEO_ASR_ENABLE
            )
        except Exception:
            asr_enabled = DEFAULT_VIDEO_ASR_ENABLE
        if asr_enabled:
            try:
                wav = await extract_audio_wav(ffmpeg_path, local_path)
                if wav and os.path.exists(wav):
                    stt = self._choose_stt_provider(event)
                    try:
                        sid = (
                            getattr(stt, "id", None)
                            or getattr(stt, "provider_id", None)
                            or stt.__class__.__name__
                        )
                    except Exception:
                        sid = None
                    logger.info("zssm_explain: stt provider=%s", sid or "unknown")
                    if stt is not None:
                        try:
                            asr_text = await stt.get_text(wav)
                            logger.info(
                                "zssm_explain: asr text length=%s",
                                len(asr_text) if isinstance(asr_text, str) else 0,
                            )
                        except Exception:
                            asr_text = None
                    try:
                        os.remove(wav)
                    except Exception:
                        pass
            except Exception:
                asr_text = None

        # 组装并调用 LLM（多次调用：逐帧→最终汇总）
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None
        if not provider:
            yield self._reply_text_result(
                event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。"
            )
            return
        system_prompt = await self._build_system_prompt(event)
        meta = {
            "name": os.path.basename(local_path),
            "duration": dur if isinstance(dur, (int, float)) else None,
            "frames": len(image_urls),
        }
        try:
            start_ts = time.perf_counter()
            call_provider = self._llm.select_vision_provider(
                session_provider=provider,
                preferred_provider=cfg_video_provider,
            )
            try:
                pid = (
                    getattr(call_provider, "id", None)
                    or getattr(call_provider, "provider_id", None)
                    or call_provider.__class__.__name__
                )
            except Exception:
                pid = None
            logger.info(
                "zssm_explain: llm provider=%s, frames=%d (multi-call)",
                pid or "unknown",
                len(image_urls),
            )

            # 1) 逐帧描述（前 n-1 帧）
            captions: List[str] = []
            if len(image_urls) > 1:
                for idx, img in enumerate(image_urls[:-1], start=1):
                    prompt_cap = DEFAULT_FRAME_CAPTION_PROMPT
                    try:
                        logger.info(
                            "zssm_explain: caption frame %d/%d",
                            idx,
                            len(image_urls) - 1,
                        )
                        resp = await self._llm.call_with_fallback(
                            primary=call_provider,
                            session_provider=provider,
                            user_prompt=prompt_cap,
                            system_prompt=system_prompt,
                            image_urls=[img],
                        )
                        cap = self._llm.pick_llm_text(resp)
                        if not cap:
                            cap = "未识别"
                        captions.append(cap)
                        logger.info("zssm_explain: caption len=%d", len(cap))
                    except Exception as e:
                        logger.warning(
                            "zssm_explain: caption failed on frame %d: %s", idx, e
                        )
                        captions.append("未识别")

            # 2) 最后一帧 + 汇总提示（总调用次数 = 帧数）
            final_prompt = self._build_video_final_prompt(meta, asr_text, captions)
            try:
                resp_final = await self._llm.call_with_fallback(
                    primary=call_provider,
                    session_provider=provider,
                    user_prompt=final_prompt,
                    system_prompt=system_prompt,
                    image_urls=[image_urls[-1]],
                )
            except Exception as e:
                logger.error("zssm_explain: final summary call failed: %s", e)
                yield self._reply_text_result(
                    event, self._format_llm_error(e, "视频解释")
                )
                return

            # 只对最终结果触发 OnLLMResponseEvent（避免中间过程外泄）
            try:
                await call_event_hook(event, EventType.OnLLMResponseEvent, resp_final)
            except Exception:
                pass

            reply_text = self._llm.pick_llm_text(resp_final)

            elapsed = None
            try:
                elapsed = time.perf_counter() - start_ts
            except Exception:
                elapsed = None
            reply_text = self._format_explain_output(reply_text, elapsed_sec=elapsed)
            yield self._reply_text_result(event, reply_text)
            try:
                event.stop_event()
            except Exception:
                pass
        finally:
            # 清理帧文件
            try:
                frame_dir = os.path.dirname(image_urls[0]) if image_urls else None
                if frame_dir and os.path.isdir(frame_dir):
                    shutil.rmtree(frame_dir, ignore_errors=True)
            except Exception:
                pass

    # 文本与图片解析等通用工具已迁移至 message_utils / llm_client / video_utils 模块

    def _is_zssm_trigger(self, text: str, is_command: bool = False) -> bool:
        """统一触发检测逻辑。
        is_command=True 表示带有前缀或 @Bot 的显式调用。
        """
        if not isinstance(text, str):
            return False
        t = text.strip()
        kw_enabled = self._get_conf_bool(KEYWORD_ZSSM_ENABLE_KEY, True)

        # 1. 关键词自动触发逻辑判定
        # 检查是否有显式的指令前缀符号 (如 / ! . 等)
        prefix_match = re.match(r"^\s*([/!！。\.、，\-]+)", t)
        has_prefix = bool(prefix_match)

        if not kw_enabled:
            # 如果关闭了关键词触发，则必须有显式的前缀符号才允许通过
            if not has_prefix:
                logger.debug(
                    f"zssm_explain: blocked keyword trigger (kw_off, no_prefix): {t}"
                )
                return False

        # 2. 正则匹配触发词
        keyword = self._match_trigger_keyword(t)
        if not keyword:
            logger.debug(f"zssm_explain: no trigger match: {t}")
            return False

        logger.debug(
            f"zssm_explain: trigger found: {keyword} (prefix={has_prefix}, is_command={is_command})"
        )
        # 仅允许 zssm 作为带前缀的命令触发；@Bot + 额外关键词仍按关键词触发处理。
        if has_prefix and keyword != COMMAND_TRIGGER_KEYWORD:
            return False

        return True

    def _match_trigger_keyword(self, text: str) -> Optional[str]:
        """从文本中提取命中的触发词，未命中则返回 None。"""
        if not isinstance(text, str):
            return None
        pattern = self._get_trigger_keyword_pattern()
        if not pattern:
            return None
        matched = re.match(
            rf"^[\s/!！。\.、，\-]*({pattern})(\s|$)",
            text.strip(),
            re.I,
        )
        if not matched:
            return None
        return self._normalize_trigger_keyword(matched.group(1))

    @staticmethod
    def _first_plain_head_text(chain: list[object]) -> str:
        """返回消息链中最靠前且非空的 Plain 文本。支持 Comp 对象和 raw dict。忽略 Reply、At 等非文本段。"""
        if not isinstance(chain, list):
            return ""
        for seg in chain:
            try:
                # 支持 Comp 对象
                if isinstance(seg, Comp.Plain):
                    txt = getattr(seg, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt
                # 支持 raw dict (OneBot/aiocqhttp)
                elif isinstance(seg, dict):
                    if seg.get("type") in ("text", "plain"):
                        txt = seg.get("data", {}).get("text")
                        if isinstance(txt, str) and txt.strip():
                            return txt
            except (AttributeError, TypeError):
                continue
        return ""

    @staticmethod
    def _chain_has_at_me(chain: list[object], self_id: str) -> bool:
        """检测消息链是否 @ 了当前 Bot。支持 Comp 对象和 raw dict。"""
        if not isinstance(chain, list):
            return False
        for seg in chain:
            try:
                # 支持 Comp 对象
                if isinstance(seg, Comp.At):
                    qq = getattr(seg, "qq", None)
                    if qq is not None and str(qq) == str(self_id):
                        return True
                # 支持 raw dict
                elif isinstance(seg, dict) and seg.get("type") == "at":
                    qq = seg.get("data", {}).get("qq")
                    if qq is not None and str(qq) == str(self_id):
                        return True
            except (AttributeError, TypeError):
                continue
        return False

    def _already_handled(self, event: AstrMessageEvent) -> bool:
        """同一事件只处理一次，避免指令与关键词双触发产生重复回复。"""
        try:
            extras = event.get_extra()
            if isinstance(extras, dict) and extras.get("zssm_handled"):
                return True
        except Exception:
            pass
        try:
            event.set_extra("zssm_handled", True)
        except Exception:
            pass
        return False

    def _strip_trigger_and_get_content(self, text: str) -> str:
        """剥离前缀与 zssm 触发词，返回其后的内容；无内容则返回空串。"""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        pattern = self._get_trigger_keyword_pattern()
        if not pattern:
            return ""
        m = re.match(
            rf"^[\s/!！。\.、，\-]*(?:{pattern})(?:\s+(.+))?$",
            t,
            re.I,
        )
        if not m:
            return ""
        content = (m.group(1) or "").strip()

        # AstrBot 的 message_outline/日志可能用 “[图片]” 等占位符表示非文本段；这里将其从 inline 内容里剔除，
        # 避免把占位符当成用户真实输入（图片本身会从 message chain 单独提取）。
        try:
            content = re.sub(
                r"[\[【](图片|image|img|视频|video|语音|record|文件|file)[\]】]",
                " ",
                content,
                flags=re.I,
            )
        except Exception:
            pass
        try:
            content = re.sub(r"\s{2,}", " ", content).strip()
        except Exception:
            content = content.strip()
        return content

    def _get_inline_content(self, event: AstrMessageEvent) -> str:
        """从消息首个 Plain 文本或整体纯文本中提取 'zssm xxx' 的 xxx 内容。"""
        try:
            chain = event.get_messages()
        except Exception:
            chain = (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )
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

    @staticmethod
    def _safe_get_chain(event: AstrMessageEvent) -> List[object]:
        try:
            return event.get_messages()
        except Exception:
            return (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )

    def _extract_images_from_event(self, event: AstrMessageEvent) -> List[str]:
        """从当前事件消息链中提取图片（用于“zssm + 图片”场景）。"""
        chain = self._safe_get_chain(event)
        try:
            _t, images, _v = extract_text_images_videos_from_chain(chain)
        except Exception:
            images = []
        return [x for x in images if isinstance(x, str) and x]

    @staticmethod
    def _chain_has_reply(chain: List[object]) -> bool:
        """检测当前消息链是否显式引用了其他消息。"""
        if not isinstance(chain, list):
            return False
        for seg in chain:
            try:
                if isinstance(seg, Comp.Reply):
                    return True
                if isinstance(seg, dict) and seg.get("type") in ("reply", "quote"):
                    return True
            except (AttributeError, TypeError):
                continue
        return False

    def _has_direct_explainable_payload(
        self, event: AstrMessageEvent, chain: Optional[List[object]] = None
    ) -> bool:
        """判断当前消息是否直接携带了可解释的媒体内容。"""
        chain = chain if isinstance(chain, list) else self._safe_get_chain(event)
        try:
            if self._extract_images_from_event(event):
                return True
        except Exception:
            pass
        try:
            return bool(extract_videos_from_chain(chain))
        except Exception:
            return False

    def _should_defer_empty_heyiwei_trigger(
        self,
        event: AstrMessageEvent,
        *,
        trigger_keyword: Optional[str],
        inline: str,
        chain: Optional[List[object]] = None,
    ) -> bool:
        """空的 `hyw/何意味` 触发词不默认占用消息，让普通聊天继续处理。"""
        if trigger_keyword not in HE_YI_WEI_TRIGGER_KEYWORDS:
            return False
        if inline:
            return False
        chain = chain if isinstance(chain, list) else self._safe_get_chain(event)
        if self._chain_has_reply(chain):
            return False
        return not self._has_direct_explainable_payload(event, chain)

    async def _resolve_images_for_llm(
        self, event: AstrMessageEvent, images: List[str]
    ) -> List[str]:
        """将图片引用尽量解析成 LLM 可用的形式，并去重保持顺序。

        支持：
        - http(s) URL
        - base64://... / data:image/...;base64,...
        - file://...（转换为本地路径）
        - 本地路径（绝对/相对，存在则通过）
        - Napcat/OneBot 场景下的 file_id（尝试 get_image/get_file/get_msg 回查）
        """

        def _norm(x: object) -> Optional[str]:
            if not isinstance(x, str) or not x:
                return None
            s = x.strip()
            if not s:
                return None
            ls = s.lower()
            if ls.startswith(("http://", "https://")):
                return s
            if ls.startswith("base64://") or ls.startswith("data:image/"):
                return s
            if ls.startswith("file://"):
                try:
                    fp = s[7:]
                    # Windows: file:///C:/xxx
                    if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                        fp = fp[1:]
                    if fp and os.path.exists(fp):
                        return os.path.abspath(fp)
                except Exception:
                    return None
                return None
            try:
                if os.path.exists(s):
                    return os.path.abspath(s)
            except Exception:
                return None
            return None

        resolved: List[str] = []
        seen = set()

        def _add(cand: str) -> None:
            if cand and cand not in seen:
                seen.add(cand)
                resolved.append(cand)

        resolve_candidates: List[str] = []
        for img in images:
            if not isinstance(img, str) or not img:
                continue
            direct = _norm(img)
            if direct:
                _add(direct)
            else:
                resolve_candidates.append(img)

        unresolved: List[str] = []
        if resolve_candidates:
            sem = asyncio.Semaphore(6)

            async def _resolve_one(fid: str) -> Optional[str]:
                async with sem:
                    try:
                        return await napcat_resolve_file_url(event, fid)
                    except Exception:
                        return None

            tasks = [_resolve_one(fid) for fid in resolve_candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for fid, res in zip(resolve_candidates, results):
                if isinstance(res, Exception) or not isinstance(res, str) or not res:
                    unresolved.append(fid)
                    continue
                rr = _norm(res)
                if rr:
                    _add(rr)
                else:
                    unresolved.append(fid)

        # 兜底：部分 OneBot 实现 event.get_messages() 不带 url，尝试 get_msg 回查当前消息拿到 url
        if unresolved and hasattr(event, "message_obj"):
            try:
                mid = getattr(event.message_obj, "message_id", None)
                mid = str(mid) if mid is not None else ""
            except Exception:
                mid = ""
            if mid:
                try:
                    ret = await call_get_msg(event, mid)
                    data = ob_data(ret or {})
                    _t, imgs2, _v, _vm = (
                        extract_from_onebot_message_payload_with_videos(data)
                    )
                    for x in imgs2:
                        nx = _norm(x)
                        if nx:
                            _add(nx)
                except Exception as e:
                    logger.debug(
                        "zssm_explain: get_msg fallback for current images failed: %s",
                        e,
                    )

        if not resolved and images:
            # 打印可读的排查信息（避免把整段 base64 打到日志）
            try:
                plat = event.get_platform_name()
            except Exception:
                plat = None
            try:
                mid = (
                    getattr(event.message_obj, "message_id", None)
                    if hasattr(event, "message_obj")
                    else None
                )
            except Exception:
                mid = None
            brief = []
            for it in images[:5]:
                if not isinstance(it, str):
                    continue
                s = it.strip()
                if not s:
                    continue
                ls = s.lower()
                if ls.startswith("base64://") or ls.startswith("data:image/"):
                    brief.append(f"{ls[:16]}...(len={len(s)})")
                else:
                    brief.append(s[:120])
            logger.warning(
                "zssm_explain: image resolve failed (platform=%s msg_id=%s count=%d samples=%s)",
                plat,
                str(mid)[:64] if mid is not None else None,
                len(images),
                brief,
            )

        return resolved

    # ===== URL 相关：检测、抓取、提取与组装提示词 =====
    def _get_conf_bool(self, key: str, default: bool) -> bool:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                lv = v.strip().lower()
                if lv in ("1", "true", "yes", "on"):  # 兼容字符串布尔
                    return True
                if lv in ("0", "false", "no", "off"):
                    return False
        except Exception:
            pass
        return default

    def _get_conf_int(
        self, key: str, default: int, min_v: int = 1, max_v: int = 120000
    ) -> int:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, int):
                return max(min(v, max_v), min_v)
            if isinstance(v, str) and v.strip().isdigit():
                return max(min(int(v.strip()), max_v), min_v)
        except Exception:
            pass
        return default

    def _get_file_preview_exts(self) -> Set[str]:
        """从配置构造文本文件预览的扩展名集合（含点）。"""
        raw = self._get_conf_str(FILE_PREVIEW_EXTS_KEY, DEFAULT_FILE_PREVIEW_EXTS)
        base_default = [
            ext.strip() for ext in DEFAULT_FILE_PREVIEW_EXTS.split(",") if ext.strip()
        ]
        return build_text_exts_from_config(raw, base_default)

    def _get_file_preview_max_bytes(self) -> Optional[int]:
        """获取允许尝试内容预览的群文件最大体积（字节）。"""
        try:
            kb = self._get_conf_int(
                FILE_PREVIEW_MAX_SIZE_KB_KEY,
                DEFAULT_FILE_PREVIEW_MAX_SIZE_KB,
                1,
                1024 * 1024,
            )
        except Exception:
            kb = DEFAULT_FILE_PREVIEW_MAX_SIZE_KB
        try:
            return int(kb) * 1024
        except Exception:
            return None

    def _get_cf_screenshot_size(self) -> Tuple[int, int]:
        """从配置解析 Cloudflare 截图尺寸（宽、高，带边界校验）。"""
        raw = self._get_conf_str(CF_SCREENSHOT_SIZE_KEY, DEFAULT_CF_SCREENSHOT_SIZE)
        w, h = 1280, 720
        if isinstance(raw, str) and "x" in raw.lower():
            try:
                parts = raw.lower().split("x")
                if len(parts) == 2:
                    w = int(parts[0].strip())
                    h = int(parts[1].strip())
            except Exception:
                w, h = 1280, 720
        # 边界与兜底
        try:
            w = max(320, min(int(w), 4096))
        except Exception:
            w = 1280
        try:
            h = max(240, min(int(h), 4096))
        except Exception:
            h = 720
        return w, h

    async def _build_system_prompt(self, event: AstrMessageEvent) -> str:
        keep = self._get_conf_bool(
            KEEP_ORIGINAL_PERSONA_KEY, DEFAULT_KEEP_ORIGINAL_PERSONA
        )
        return await build_system_prompt_for_event(
            self.context,
            event.unified_msg_origin,
            keep_original_persona=keep,
        )

    # 解释输出格式化仍保留在 main.py（与事件/输出策略紧耦合）

    def _format_explain_output(
        self,
        content: str,
        elapsed_sec: Optional[float] = None,
    ) -> str:
        """统一格式化解释结果，仅追加耗时信息。

        “关键词：...” 行以及“**详细阐述：**”等结构由 LLM 自行生成。
        """
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        body = content.strip()
        if not body:
            return ""

        parts: List[str] = [body]
        if isinstance(elapsed_sec, (int, float)) and elapsed_sec > 0:
            parts.append("")
            parts.append(f"cost: {elapsed_sec:.3f}s")

        return "\n".join(parts)

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
                    logger.warning(
                        f"zssm_explain: provider id not found for {key}={pid}: {e}"
                    )
        except Exception:
            pass
        return None

    @dataclass
    class _LLMPlan:
        user_prompt: str
        images: List[str] = field(default_factory=list)
        cleanup_paths: List[str] = field(default_factory=list)

    @dataclass
    class _VideoPlan:
        video_src: str
        cleanup_paths: List[str] = field(default_factory=list)
        video_meta: dict = field(default_factory=dict)

    @dataclass
    class _BilibiliPlan:
        """B站图文/动态/直播等非视频内容的解释计划。"""

        bili_url: str
        bili_type: str  # dynamic/live/read/opus
        cleanup_paths: List[str] = field(default_factory=list)

    @dataclass
    class _ReplyPlan:
        message: str
        stop_event: bool = True
        cleanup_paths: List[str] = field(default_factory=list)

    _ExplainPlan = Union[_LLMPlan, _VideoPlan, _BilibiliPlan, _ReplyPlan]

    def _build_empty_input_reply_plan(
        self, cleanup_paths: Optional[List[str]] = None
    ) -> Optional[_ReplyPlan]:
        if not self._get_conf_bool(
            EMPTY_ZSSM_PROMPT_ENABLE_KEY, DEFAULT_EMPTY_ZSSM_PROMPT_ENABLE
        ):
            return None
        return self._ReplyPlan(
            message="请输入要解释的内容。",
            stop_event=True,
            cleanup_paths=list(cleanup_paths or []),
        )

    async def _build_explain_plan(
        self,
        event: AstrMessageEvent,
        *,
        inline: str,
        enable_url: bool,
    ) -> Optional[_ExplainPlan]:
        """将输入解析/拼装为一个可执行的解释计划（builder 阶段）。"""
        cleanup_paths: List[str] = []

        if inline:
            urls = extract_urls_from_text(inline) if enable_url else []
            if urls:
                target_url = urls[0]
                if is_bilibili_url(target_url):
                    # 短链先展开，再判断真实类型
                    bili_type = get_bilibili_url_type(target_url)
                    if bili_type == "short":
                        target_url = await resolve_bilibili_short_url(target_url)
                        bili_type = get_bilibili_url_type(target_url)
                    # 视频类型走 VideoPlan
                    if bili_type == "video" or bili_type is None:
                        return self._VideoPlan(
                            video_src=target_url, cleanup_paths=cleanup_paths
                        )
                    # 动态/直播/专栏/图文走 BilibiliPlan（图文发给模型）
                    return self._BilibiliPlan(
                        bili_url=target_url,
                        bili_type=bili_type,
                        cleanup_paths=cleanup_paths,
                    )

                timeout_sec = self._get_conf_int(
                    URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60
                )
                max_chars = self._get_conf_int(
                    URL_MAX_CHARS_KEY,
                    DEFAULT_URL_MAX_CHARS,
                    min_v=1000,
                    max_v=50000,
                )
                cf_enable = self._get_conf_bool(
                    CF_SCREENSHOT_ENABLE_KEY, DEFAULT_CF_SCREENSHOT_ENABLE
                )
                width, height = self._get_cf_screenshot_size()
                url_ctx = await prepare_url_prompt(
                    target_url,
                    timeout_sec,
                    self._last_fetch_info,
                    max_chars=max_chars,
                    cf_screenshot_enable=cf_enable,
                    cf_screenshot_width=width,
                    cf_screenshot_height=height,
                    file_preview_max_bytes=self._get_file_preview_max_bytes(),
                    user_prompt_template=DEFAULT_URL_USER_PROMPT,
                )
                if not url_ctx:
                    return self._ReplyPlan(
                        message=build_url_failure_message(
                            self._last_fetch_info, cf_enable
                        ),
                        stop_event=True,
                        cleanup_paths=cleanup_paths,
                    )
                user_prompt, _text, images = url_ctx
                return self._LLMPlan(
                    user_prompt=user_prompt, images=images, cleanup_paths=cleanup_paths
                )

            inline_images_raw = self._extract_images_from_event(event)
            inline_images = (
                await self._resolve_images_for_llm(event, inline_images_raw)
                if inline_images_raw
                else []
            )
            if inline_images_raw and not inline_images:
                # 部分平台会把图片段转成占位文本（如 "[图片]"），此时如果图片解析失败就不要继续调用 LLM。
                placeholder = str(inline or "").strip().lower()
                if placeholder in ("[图片]", "[image]", "[img]") or not placeholder:
                    return self._ReplyPlan(
                        message="未能获取到图片（未拿到可访问的链接/本地路径/base64）。请尝试重新发送图片，或查看日志 `zssm_explain: image resolve failed` / `zssm_explain: napcat resolve file/url failed` 获取详细信息。",
                        cleanup_paths=cleanup_paths,
                    )
            user_prompt = build_user_prompt(inline, inline_images)
            return self._LLMPlan(
                user_prompt=user_prompt,
                images=inline_images,
                cleanup_paths=cleanup_paths,
            )

        (
            text,
            images,
            vids,
            from_forward,
            video_meta,
        ) = await extract_quoted_payload_with_videos(event)
        # 同时支持“zssm + 图片”（图片在当前消息里，而非被回复消息中）
        try:
            images.extend(self._extract_images_from_event(event))
        except Exception:
            pass
        # 去重图片列表
        if isinstance(images, list):
            images = list(dict.fromkeys(images))

        if vids and not from_forward:
            vm = video_meta.get(0, {}) if video_meta else {}
            return self._VideoPlan(
                video_src=vids[0], cleanup_paths=cleanup_paths, video_meta=vm
            )

        if from_forward and vids:
            try:
                enabled = self._get_conf_bool(
                    FORWARD_VIDEO_KEYFRAME_ENABLE_KEY,
                    DEFAULT_FORWARD_VIDEO_KEYFRAME_ENABLE,
                )
                ffmpeg_path = self._resolve_ffmpeg()
                ffprobe_path = self._resolve_ffprobe()
                max_count = self._get_conf_int(
                    FORWARD_VIDEO_MAX_COUNT_KEY,
                    DEFAULT_FORWARD_VIDEO_MAX_COUNT,
                    0,
                    10,
                )
                max_mb = self._get_conf_int(
                    VIDEO_MAX_SIZE_MB_KEY, DEFAULT_VIDEO_MAX_SIZE_MB, 1, 512
                )
                max_sec = self._get_conf_int(
                    VIDEO_MAX_DURATION_SEC_KEY,
                    DEFAULT_VIDEO_MAX_DURATION_SEC,
                    10,
                    3600,
                )
                timeout_sec = self._get_conf_int(
                    URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60
                )
                f_frames, f_cleanup = await extract_forward_video_keyframes(
                    event,
                    vids,
                    enabled=enabled,
                    max_count=max_count,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    max_mb=max_mb,
                    max_sec=max_sec,
                    timeout_sec=timeout_sec,
                    video_meta=video_meta,
                )
                if f_frames:
                    images.extend(f_frames)
                    cleanup_paths.extend(f_cleanup)
                    note = (
                        f"（聊天记录包含 {len(vids)} 个视频，已抽取部分关键帧辅助解释）"
                    )
                    if isinstance(text, str) and text.strip():
                        text = f"{text}\n\n{note}"
                    else:
                        text = note
            except Exception as e:
                logger.warning("zssm_explain: forward video keyframe failed: %s", e)

        if (not vids) and (not from_forward) and (not text) and (not images):
            try:
                chain_now = event.get_messages()
            except Exception:
                chain_now = getattr(event.message_obj, "message", []) or []
            try:
                vids_now = extract_videos_from_chain(chain_now)
            except Exception:
                vids_now = []
            if vids_now:
                return self._VideoPlan(
                    video_src=vids_now[0], cleanup_paths=cleanup_paths
                )

        try:
            file_preview = await extract_file_preview_from_reply(
                event,
                text_exts=self._get_file_preview_exts(),
                max_size_bytes=self._get_file_preview_max_bytes(),
            )
        except Exception:
            file_preview = None
        if file_preview:
            if text:
                text = f"{file_preview}\n\n{text}"
            else:
                text = file_preview

        raw_images = list(images) if isinstance(images, list) else []
        try:
            images = await self._resolve_images_for_llm(event, images)
        except Exception:
            images = []
        # Deduplicate resolved images
        if isinstance(images, list):
            images = list(dict.fromkeys(images))

        if not text and not images:
            if raw_images:
                return self._ReplyPlan(
                    message="未能获取到图片（未拿到可访问的链接/本地路径/base64），请尝试重新发送图片，或查看日志 `zssm_explain: image resolve failed` / `zssm_explain: napcat resolve file/url failed` 排查 OneBot/Napcat 是否能返回图片 URL（可检查 get_msg/get_image/get_file）。",
                    stop_event=True,
                    cleanup_paths=cleanup_paths,
                )
            return self._build_empty_input_reply_plan(cleanup_paths)

        urls = extract_urls_from_text(text) if (enable_url and text) else []
        if urls and not from_forward:
            target_url = urls[0]
            if is_bilibili_url(target_url):
                # 短链先展开，再判断真实类型
                bili_type = get_bilibili_url_type(target_url)
                if bili_type == "short":
                    target_url = await resolve_bilibili_short_url(target_url)
                    bili_type = get_bilibili_url_type(target_url)
                # 视频类型走 VideoPlan
                if bili_type == "video" or bili_type is None:
                    return self._VideoPlan(
                        video_src=target_url, cleanup_paths=cleanup_paths
                    )
                return self._BilibiliPlan(
                    bili_url=target_url,
                    bili_type=bili_type,
                    cleanup_paths=cleanup_paths,
                )

            timeout_sec = self._get_conf_int(
                URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60
            )
            max_chars = self._get_conf_int(
                URL_MAX_CHARS_KEY,
                DEFAULT_URL_MAX_CHARS,
                min_v=1000,
                max_v=50000,
            )
            cf_enable = self._get_conf_bool(
                CF_SCREENSHOT_ENABLE_KEY, DEFAULT_CF_SCREENSHOT_ENABLE
            )
            width, height = self._get_cf_screenshot_size()
            url_ctx = await prepare_url_prompt(
                target_url,
                timeout_sec,
                self._last_fetch_info,
                max_chars=max_chars,
                cf_screenshot_enable=cf_enable,
                cf_screenshot_width=width,
                cf_screenshot_height=height,
                file_preview_max_bytes=self._get_file_preview_max_bytes(),
                user_prompt_template=DEFAULT_URL_USER_PROMPT,
            )
            if not url_ctx:
                return self._ReplyPlan(
                    message=build_url_failure_message(self._last_fetch_info, cf_enable),
                    stop_event=True,
                    cleanup_paths=cleanup_paths,
                )
            user_prompt, _text, images = url_ctx
            return self._LLMPlan(
                user_prompt=user_prompt, images=images, cleanup_paths=cleanup_paths
            )

        if urls and from_forward:
            base_prompt = build_user_prompt(text, images)
            target_url = urls[0]
            timeout_sec = self._get_conf_int(
                URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60
            )
            extra_block = ""
            try:
                html = await fetch_html(target_url, timeout_sec, self._last_fetch_info)
            except Exception:
                html = None
            if isinstance(html, str) and html.strip():
                max_chars = self._get_conf_int(
                    URL_MAX_CHARS_KEY,
                    DEFAULT_URL_MAX_CHARS,
                    min_v=1000,
                    max_v=50000,
                )
                title, desc, snippet = build_url_brief_for_forward(html, max_chars)
                extra_block = (
                    "\n\n此外，这段聊天记录中包含一个网页链接，请结合下面的网页关键信息一起解释整段对话：\n"
                    f"网址: {target_url}\n"
                    f"标题: {title or '(未获取)'}\n"
                    f"描述: {desc or '(未获取)'}\n"
                    "正文片段:\n"
                    f"{snippet}"
                )
            user_prompt = base_prompt + extra_block
            return self._LLMPlan(
                user_prompt=user_prompt, images=images, cleanup_paths=cleanup_paths
            )

        user_prompt = build_user_prompt(text, images)
        return self._LLMPlan(
            user_prompt=user_prompt, images=images, cleanup_paths=cleanup_paths
        )

    async def _execute_explain_plan(self, event: AstrMessageEvent, plan: _ExplainPlan):
        """执行解释计划（executor 阶段）。"""
        if isinstance(plan, self._VideoPlan):
            async for r in self._explain_video(
                event, plan.video_src, video_meta=plan.video_meta
            ):
                yield r
            return

        if isinstance(plan, self._BilibiliPlan):
            async for r in self._explain_bilibili(event, plan.bili_url, plan.bili_type):
                yield r
            return

        if isinstance(plan, self._ReplyPlan):
            if isinstance(plan.message, str) and plan.message.strip():
                yield self._reply_text_result(event, plan.message)
            if plan.stop_event:
                try:
                    event.stop_event()
                except Exception:
                    pass
            return

        user_prompt = plan.user_prompt
        images = plan.images

        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None

        if not provider:
            yield self._reply_text_result(
                event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。"
            )
            return

        system_prompt = await self._build_system_prompt(event)
        image_urls = self._llm.filter_supported_images(images)

        try:
            start_ts = time.perf_counter()
            call_provider = self._llm.select_primary_provider(
                session_provider=provider, image_urls=image_urls
            )
            llm_resp = await self._llm.call_with_fallback(
                primary=call_provider,
                session_provider=provider,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image_urls=image_urls,
            )

            try:
                await call_event_hook(event, EventType.OnLLMResponseEvent, llm_resp)
            except Exception:
                pass

            reply_text = None
            try:
                ct = getattr(llm_resp, "completion_text", None)
                if isinstance(ct, str) and ct.strip():
                    reply_text = ct.strip()
            except Exception:
                reply_text = None
            if not reply_text:
                reply_text = self._llm.pick_llm_text(llm_resp)

            elapsed = None
            try:
                elapsed = time.perf_counter() - start_ts
            except Exception:
                elapsed = None
            reply_text = self._format_explain_output(reply_text, elapsed_sec=elapsed)
            yield self._reply_text_result(event, reply_text)
            try:
                event.stop_event()
            except Exception:
                pass
        except asyncio.TimeoutError:
            yield self._reply_text_result(
                event, "解释超时，请稍后重试或换一个模型提供商。"
            )
            try:
                event.stop_event()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"zssm_explain: LLM call failed: {e}")
            yield self._reply_text_result(event, self._format_llm_error(e, "解释"))
            try:
                event.stop_event()
            except Exception:
                pass

    @filter.command("zssm")
    async def zssm(self, event: AstrMessageEvent):
        """解释被回复消息：/zssm 或关键词触发；若携带内容则直接解释该内容，否则按回复消息逻辑。"""
        cleanup_paths: List[str] = []
        try:
            try:
                if not self._is_group_allowed(event):
                    return
            except Exception:
                pass

            # 触发校验：确保开启了对应开关，且在关闭关键词触发时必须带有指令前缀
            chain = self._safe_get_chain(event)
            head_text = self._first_plain_head_text(chain)
            # 如果首个文本没搜到，回退到整体字符串
            check_text = head_text or (getattr(event, "message_str", "") or "")
            trigger_keyword = self._match_trigger_keyword(check_text)
            actual_command_like = False
            try:
                text_for_check = check_text.strip()
                has_prefix = bool(
                    re.match(r"^\s*([/!！。\.、，\-]+)", text_for_check)
                )
                at_me = False
                try:
                    self_id = event.get_self_id()
                    at_me = self._chain_has_at_me(chain, self_id)
                except Exception:
                    at_me = False
                actual_command_like = has_prefix or (
                    at_me and trigger_keyword == COMMAND_TRIGGER_KEYWORD
                )
            except Exception:
                actual_command_like = False

            if not self._is_zssm_trigger(
                check_text, is_command=actual_command_like
            ):
                logger.debug(
                    f"zssm_explain: zssm command filter blocked execution. check_text: {check_text}"
                )
                return

            inline = self._get_inline_content(event)
            if self._should_defer_empty_heyiwei_trigger(
                event,
                trigger_keyword=trigger_keyword,
                inline=inline,
                chain=chain,
            ):
                reply_plan = self._build_empty_input_reply_plan()
                if reply_plan is None:
                    logger.debug(
                        "zssm_explain: ignore empty heyiwei trigger without reply/payload"
                    )
                    return
                event.should_call_llm(True)
                if self._already_handled(event):
                    return
                async for r in self._execute_explain_plan(event, reply_plan):
                    yield r
                return

            # 真正进入解释流程前再阻止默认 LLM 接管，避免空的 hyw/何意味 误拦截普通聊天
            event.should_call_llm(True)
            if self._already_handled(event):
                return
            enable_url = self._get_conf_bool(
                URL_DETECT_ENABLE_KEY, DEFAULT_URL_DETECT_ENABLE
            )

            plan = await self._build_explain_plan(
                event, inline=inline, enable_url=enable_url
            )
            if plan is None:
                return
            try:
                cleanup_paths = list(getattr(plan, "cleanup_paths", []) or [])
            except Exception:
                cleanup_paths = []

            async for r in self._execute_explain_plan(event, plan):
                yield r
        except Exception as e:
            logger.error("zssm_explain: handler crashed: %s", e)
            yield self._reply_text_result(
                event, "解释失败：插件内部异常，请稍后再试或联系管理员。"
            )
            try:
                event.stop_event()
            except Exception:
                pass
        finally:
            try:
                for p in cleanup_paths:
                    try:
                        if isinstance(p, str) and p:
                            if os.path.isdir(p):
                                shutil.rmtree(p, ignore_errors=True)
                            elif os.path.isfile(p):
                                os.remove(p)
                    except Exception:
                        continue
            except Exception:
                pass

    async def terminate(self):
        return

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def keyword_zssm(self, event: AstrMessageEvent):
        """关键词触发：忽略常见前缀/Reply/At 等，检测首个 Plain 段的配置关键词。
        避免与 /zssm 指令重复：若以 /zssm 开头则交由指令处理。
        """
        # 群聊权限控制：不满足条件则直接忽略
        try:
            if not self._is_group_allowed(event):
                return
        except Exception:
            pass
        # 优先使用消息链首个 Plain 段判断
        try:
            chain = event.get_messages()
        except Exception:
            chain = (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )
        head = self._first_plain_head_text(chain)
        # 如果 @ 了 Bot 并且首个 Plain 文本是 zssm，则交由指令处理以避免重复
        at_me = False
        try:
            self_id = event.get_self_id()
            at_me = self._chain_has_at_me(chain, self_id)
        except Exception:
            at_me = False
        if isinstance(head, str) and head.strip():
            hs = head.strip()
            if re.match(
                rf"^\s*/\s*({re.escape(COMMAND_TRIGGER_KEYWORD)})(\s|$)",
                hs,
                re.I,
            ):
                return
            if at_me and re.match(
                rf"^({re.escape(COMMAND_TRIGGER_KEYWORD)})(\s|$)", hs, re.I
            ):
                return
            if self._is_zssm_trigger(hs, is_command=at_me):
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
            if re.match(
                rf"^\s*/\s*({re.escape(COMMAND_TRIGGER_KEYWORD)})(\s|$)",
                t,
                re.I,
            ):
                return
            if at_me and re.match(
                rf"^({re.escape(COMMAND_TRIGGER_KEYWORD)})(\s|$)", t, re.I
            ):
                return
            if self._is_zssm_trigger(t, is_command=at_me):
                async for r in self.zssm(event):
                    yield r
