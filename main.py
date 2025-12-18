from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict, Set
import os
import asyncio
import re
import shutil
import math
import time

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
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
from .message_utils import extract_quoted_payload_with_videos
from .video_utils import (
    extract_videos_from_chain,
    is_http_url,
    is_abs_file,
    napcat_resolve_file_url,
    probe_duration_sec,
    resolve_ffmpeg,
    resolve_ffprobe,
    download_video_to_temp,
    sample_frames_equidistant,
    sample_frames_with_ffmpeg,
    extract_audio_wav,
)
from .bilibili_utils import is_bilibili_url, download_bilibili_video_to_temp
from .prompt_utils import (
    DEFAULT_URL_USER_PROMPT,
    DEFAULT_VIDEO_USER_PROMPT,
    DEFAULT_FRAME_CAPTION_PROMPT,
    build_user_prompt,
    build_system_prompt,
)
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
LLM_TIMEOUT_SEC_KEY = "llm_timeout_sec"
KEYWORD_ZSSM_ENABLE_KEY = "enable_keyword_zssm"
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

DEFAULT_URL_DETECT_ENABLE = True
DEFAULT_URL_FETCH_TIMEOUT = 8
DEFAULT_URL_MAX_CHARS = 6000
DEFAULT_LLM_TIMEOUT_SEC = 90
DEFAULT_VIDEO_FRAME_INTERVAL_SEC = 6
DEFAULT_VIDEO_ASR_ENABLE = False
DEFAULT_VIDEO_MAX_DURATION_SEC = 120
DEFAULT_VIDEO_MAX_SIZE_MB = 50
DEFAULT_FFMPEG_PATH = "ffmpeg"
DEFAULT_CF_SCREENSHOT_ENABLE = True
DEFAULT_CF_SCREENSHOT_SIZE = "1280x720"
DEFAULT_KEEP_ORIGINAL_PERSONA = False
DEFAULT_FILE_PREVIEW_EXTS = "txt,md,log,json,csv,ini,cfg,yml,yaml,py"
DEFAULT_FILE_PREVIEW_MAX_SIZE_KB = 8192
DEFAULT_FORWARD_VIDEO_KEYFRAME_ENABLE = True
DEFAULT_FORWARD_VIDEO_MAX_COUNT = 2




@register(
    "astrbot_zssm_explain",
    "薄暝",
    "zssm，支持关键词“zssm”（忽略前缀）与“zssm + 内容”直接解释；引用消息（含@）正常处理；支持 QQ 合并转发；未回复仅发 zssm 时提示；默认提示词可在 main.py 顶部修改。",
    "v3.9.8",
    "https://github.com/xiaoxi68/astrbot_zssm_explain",
)
class ZssmExplain(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = config or {}
        self._last_fetch_info: Dict[str, Any] = {}

    async def initialize(self):
        """可选：插件初始化。"""

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

    # ===== 群聊权限控制 =====
    def _get_conf_str(self, key: str, default: str) -> str:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, str):
                return v.strip()
        except Exception:
            pass
        return default

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

    # ===== 视频相关工具 =====
    @staticmethod
    def _extract_videos_from_chain(chain: List[object]) -> List[str]:
        """兼容旧接口，委托 video_utils.extract_videos_from_chain。"""
        return extract_videos_from_chain(chain)

    # ===== Napcat (aiocqhttp) 文件直链解析 =====
    @staticmethod
    def _is_http_url(s: Optional[str]) -> bool:
        return is_http_url(s)

    @staticmethod
    def _is_abs_file(s: Optional[str]) -> bool:
        return is_abs_file(s)

    def _is_napcat(self, event: AstrMessageEvent) -> bool:
        try:
            return event.get_platform_name() == "aiocqhttp" and hasattr(event, "bot") and hasattr(event.bot, "api")
        except Exception:
            return False

    async def _napcat_resolve_file_url(self, event: AstrMessageEvent, file_id: str) -> Optional[str]:
        """兼容旧接口，委托 video_utils.napcat_resolve_file_url。"""
        return await napcat_resolve_file_url(event, file_id)

    def _resolve_ffmpeg(self) -> Optional[str]:
        """根据插件配置解析 ffmpeg 路径，委托 video_utils.resolve_ffmpeg。"""
        cfg_path = self._get_conf_str(FFMPEG_PATH_KEY, DEFAULT_FFMPEG_PATH)
        return resolve_ffmpeg(cfg_path, DEFAULT_FFMPEG_PATH)

    def _resolve_ffprobe(self) -> Optional[str]:
        """根据已解析的 ffmpeg 路径解析 ffprobe，委托 video_utils.resolve_ffprobe。"""
        ff = self._resolve_ffmpeg()
        return resolve_ffprobe(ff)

    def _probe_duration_sec(self, ffprobe_path: Optional[str], video_path: str) -> Optional[float]:
        """兼容旧接口，委托 video_utils.probe_duration_sec。"""
        return probe_duration_sec(ffprobe_path, video_path)

    def _build_video_user_prompt(self, meta: Dict[str, Any], asr_text: Optional[str]) -> str:
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
        asr_block = f"音频转写要点: \n{asr_text.strip()}" if isinstance(asr_text, str) and asr_text.strip() else ""
        return tmpl.format(meta_block=meta_block, asr_block=asr_block)

    def _build_video_final_prompt(self, meta: Dict[str, Any], asr_text: Optional[str], captions: List[str]) -> str:
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
        caps_block = "\n".join([f"- {c.strip()}" for c in captions if isinstance(c, str) and c.strip()])
        asr_block = f"音频转写要点: \n{asr_text.strip()}" if isinstance(asr_text, str) and asr_text.strip() else ""
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
            pid = self.config.get(ASR_PROVIDER_ID_KEY) if isinstance(self.config, dict) else None
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

    def _select_video_provider(self, session_provider: Any, image_urls: List[str]) -> Any:
        """用于视频解释的 Provider 选择：
        1) 优先使用配置的 video_provider_id；
        2) 否则使用当前会话 Provider（需支持图片能力）；
        3) 否则在所有 Provider 中选择首个支持图片的；
        4) 否则回退为当前会话 Provider。
        """
        cfg_vid = self._get_config_provider(VIDEO_PROVIDER_ID_KEY)
        if cfg_vid is not None:
            return cfg_vid
        if session_provider and self._provider_supports_image(session_provider):
            return session_provider
        for p in self.context.get_all_providers():
            if p is session_provider:
                continue
            if self._provider_supports_image(p):
                return p
        return session_provider

    async def _explain_video(self, event: AstrMessageEvent, video_src: str):
        # 配置检查：未配置 video_provider_id 时视为未启用视频解释
        cfg_video_provider = self._get_config_provider(VIDEO_PROVIDER_ID_KEY)
        if cfg_video_provider is None:
            yield self._reply_text_result(
                event,
                "视频解释未配置可用的模型提供商，请在插件配置中选择 video_provider_id 后再试。",
            )
            return
        ffmpeg_path = self._resolve_ffmpeg()
        if not ffmpeg_path:
            yield self._reply_text_result(
                event,
                "未检测到 ffmpeg，请安装系统 ffmpeg 或 Python 包 imageio-ffmpeg，并在插件配置中设置 ffmpeg_path。",
            )
            return
        logger.info("zssm_explain: video start src=%s ffmpeg=%s", (str(video_src)[:128] if video_src else ""), ffmpeg_path)

        # 统一获取本地文件路径（支持 http/https 下载，Napcat 直链解析）
        max_mb = self._get_conf_int(VIDEO_MAX_SIZE_MB_KEY, DEFAULT_VIDEO_MAX_SIZE_MB, 1, 512)
        local_path = None
        src = video_src
        # 优先特判 B 站视频链接：通过 video_utils 解析并下载到临时文件
        if isinstance(src, str) and self._is_http_url(src) and is_bilibili_url(src):
            try:
                bili_local = await download_bilibili_video_to_temp(src, max_mb)
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
            # 若不是 URL/绝对路径，尝试通过 Napcat file_id 获取直链
            if isinstance(src, str) and (not self._is_http_url(src)) and (not self._is_abs_file(src)):
                try:
                    resolved = await self._napcat_resolve_file_url(event, src)
                except Exception:
                    resolved = None
                if isinstance(resolved, str) and resolved:
                    src = resolved
            if isinstance(src, str) and self._is_http_url(src):
                local_path = await download_video_to_temp(src, max_mb)
                if not local_path:
                    yield self._reply_text_result(event, f"视频下载失败或超过大小限制（>{max_mb}MB）。")
                    return
            else:
                # 假定为本地路径
                if not (isinstance(src, str) and os.path.isabs(src) and os.path.exists(src)):
                    yield self._reply_text_result(event, "无法读取该视频源，请确认路径或链接有效。")
                    return
                # 大小检查
                try:
                    sz = os.path.getsize(src)
                    if sz > max_mb * 1024 * 1024:
                        yield self._reply_text_result(event, f"视频大小超过限制（>{max_mb}MB），请压缩或截取片段后重试。")
                        return
                except Exception:
                    pass
                local_path = src

        # 时长检查（可选，缺少 ffprobe 时跳过）
        max_sec = self._get_conf_int(VIDEO_MAX_DURATION_SEC_KEY, DEFAULT_VIDEO_MAX_DURATION_SEC, 10, 3600)
        dur = self._probe_duration_sec(resolve_ffprobe(ffmpeg_path), local_path)
        logger.info("zssm_explain: probed duration=%s (max=%s)", dur if dur is not None else "unknown", max_sec)
        if isinstance(dur, (int, float)) and dur > max_sec:
            yield self._reply_text_result(event, f"视频时长超过限制（>{max_sec}s），请截取片段后重试。")
            return

        # 抽帧
        is_gif = False
        try:
            if isinstance(local_path, str) and local_path.lower().endswith(".gif"):
                is_gif = True
        except Exception:
            is_gif = False

        interval = self._get_conf_int(VIDEO_FRAME_INTERVAL_SEC_KEY, DEFAULT_VIDEO_FRAME_INTERVAL_SEC, 1, 120)
        try:
            if isinstance(dur, (int, float)) and dur > 0:
                # 依据时长与间隔估算目标帧数；GIF 固定抽 1 帧
                if is_gif:
                    n_frames = 1
                else:
                    n_frames = max(1, int(math.ceil(float(dur) / max(1, interval))))
                logger.info(
                    "zssm_explain: sampling plan: duration=%.2fs interval=%ss => target_frames=%s",
                    float(dur), interval, n_frames,
                )
                frames = await sample_frames_equidistant(ffmpeg_path, local_path, float(dur), n_frames)
            else:
                # 未获知时长时，以最大允许时长作为上界推导帧数；GIF 固定抽 1 帧
                if is_gif:
                    n_frames = 1
                else:
                    n_frames = max(1, int(math.ceil(float(max_sec) / max(1, interval))))
                logger.info(
                    "zssm_explain: sampling plan: unknown duration, use max_sec=%ss interval=%ss => target_frames=%s",
                    max_sec, interval, n_frames,
                )
                frames = await sample_frames_with_ffmpeg(ffmpeg_path, local_path, interval, n_frames)
        except Exception as e:
            yield self._reply_text_result(event, f"抽帧失败：{e}")
            return
        logger.info("zssm_explain: sampled %d frames", len(frames))
        image_urls = self._filter_supported_images(frames)
        if not image_urls:
            yield self._reply_text_result(event, "未能生成可用关键帧，请检查 ffmpeg 或更换视频后重试。")
            return

        # 可选 ASR：由 video_asr_enable + asr_provider_id 控制
        asr_text = None
        asr_enabled = False
        try:
            asr_enabled = self._get_conf_bool(VIDEO_ASR_ENABLE_KEY, DEFAULT_VIDEO_ASR_ENABLE)
        except Exception:
            asr_enabled = DEFAULT_VIDEO_ASR_ENABLE
        if asr_enabled:
            try:
                wav = await extract_audio_wav(ffmpeg_path, local_path)
                if wav and os.path.exists(wav):
                    stt = self._choose_stt_provider(event)
                    try:
                        sid = getattr(stt, "id", None) or getattr(stt, "provider_id", None) or stt.__class__.__name__
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
            yield self._reply_text_result(event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。")
            return
        system_prompt = await self._build_system_prompt(event)
        meta = {
            "name": os.path.basename(local_path),
            "duration": dur if isinstance(dur, (int, float)) else None,
            "frames": len(image_urls),
        }
        try:
            start_ts = time.perf_counter()
            call_provider = self._select_video_provider(provider, image_urls)
            try:
                pid = getattr(call_provider, "id", None) or getattr(call_provider, "provider_id", None) or call_provider.__class__.__name__
            except Exception:
                pid = None
            logger.info("zssm_explain: llm provider=%s, frames=%d (multi-call)", pid or "unknown", len(image_urls))

            # 1) 逐帧描述（前 n-1 帧）
            captions: List[str] = []
            if len(image_urls) > 1:
                for idx, img in enumerate(image_urls[:-1], start=1):
                    prompt_cap = DEFAULT_FRAME_CAPTION_PROMPT
                    try:
                        logger.info("zssm_explain: caption frame %d/%d", idx, len(image_urls)-1)
                        resp = await self._call_llm_with_fallback(
                            primary=call_provider,
                            session_provider=provider,
                            user_prompt=prompt_cap,
                            system_prompt=system_prompt,
                            image_urls=[img],
                        )
                        cap = self._pick_llm_text(resp)
                        cap = self._sanitize_model_output(cap)
                        if not cap:
                            cap = "未识别"
                        captions.append(cap)
                        logger.info("zssm_explain: caption len=%d", len(cap))
                    except Exception as e:
                        logger.warning("zssm_explain: caption failed on frame %d: %s", idx, e)
                        captions.append("未识别")

            # 2) 最后一帧 + 汇总提示（总调用次数 = 帧数）
            final_prompt = self._build_video_final_prompt(meta, asr_text, captions)
            try:
                resp_final = await self._call_llm_with_fallback(
                    primary=call_provider,
                    session_provider=provider,
                    user_prompt=final_prompt,
                    system_prompt=system_prompt,
                    image_urls=[image_urls[-1]],
                )
            except Exception as e:
                logger.error("zssm_explain: final summary call failed: %s", e)
                raise

            # 只对最终结果触发 OnLLMResponseEvent（避免中间过程外泄）
            try:
                await call_event_hook(event, EventType.OnLLMResponseEvent, resp_final)
            except Exception:
                pass

            reply_text = self._pick_llm_text(resp_final)
            show_reasoning = False
            try:
                cfg = self.context.get_config(umo=event.unified_msg_origin) or {}
                ps = cfg.get("provider_settings", {})
                show_reasoning = bool(ps.get("display_reasoning_text", False))
            except Exception:
                show_reasoning = False
            if not show_reasoning:
                reply_text = self._sanitize_model_output(reply_text)

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

    async def _extract_forward_video_keyframes(
        self,
        event: AstrMessageEvent,
        video_sources: List[str],
    ) -> Tuple[List[str], List[str]]:
        """将合并转发中的视频源转换为少量关键帧图片，用于“聊天记录解释”场景。

        返回:
        - frames: 关键帧图片路径/URL（当前仅返回本地抽帧图片路径）
        - cleanup_paths: 需要清理的临时路径（文件或目录）
        """
        if not video_sources:
            return ([], [])
        if not self._get_conf_bool(FORWARD_VIDEO_KEYFRAME_ENABLE_KEY, DEFAULT_FORWARD_VIDEO_KEYFRAME_ENABLE):
            return ([], [])

        ffmpeg_path = self._resolve_ffmpeg()
        if not ffmpeg_path:
            return ([], [])

        max_count = self._get_conf_int(FORWARD_VIDEO_MAX_COUNT_KEY, DEFAULT_FORWARD_VIDEO_MAX_COUNT, 0, 10)
        if max_count <= 0:
            return ([], [])

        # 去重保持顺序
        uniq_sources: List[str] = []
        seen = set()
        for s in video_sources:
            if isinstance(s, str) and s and s not in seen:
                seen.add(s)
                uniq_sources.append(s)

        max_mb = self._get_conf_int(VIDEO_MAX_SIZE_MB_KEY, DEFAULT_VIDEO_MAX_SIZE_MB, 1, 512)
        max_sec = self._get_conf_int(VIDEO_MAX_DURATION_SEC_KEY, DEFAULT_VIDEO_MAX_DURATION_SEC, 10, 3600)
        ffprobe_path = self._resolve_ffprobe()

        frames: List[str] = []
        cleanup: List[str] = []

        timeout_sec = self._get_conf_int(URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60)

        for src in uniq_sources[:max_count]:
            local_path = None
            downloaded_tmp = False
            try:
                resolved_src = src
                # Napcat file_id -> 直链
                if isinstance(resolved_src, str) and (not self._is_http_url(resolved_src)) and (not self._is_abs_file(resolved_src)):
                    try:
                        resolved = await self._napcat_resolve_file_url(event, resolved_src)
                    except Exception:
                        resolved = None
                    if isinstance(resolved, str) and resolved:
                        resolved_src = resolved

                if isinstance(resolved_src, str) and self._is_http_url(resolved_src):
                    try:
                        local_path = await asyncio.wait_for(
                            download_video_to_temp(resolved_src, max_mb),
                            timeout=max(2, int(timeout_sec)),
                        )
                    except Exception as e:
                        logger.warning("zssm_explain: forward video download timeout/failed: %s", e)
                    if local_path:
                        downloaded_tmp = True
                elif isinstance(resolved_src, str) and self._is_abs_file(resolved_src) and os.path.exists(resolved_src):
                    local_path = resolved_src

                if not local_path:
                    continue

                # 时长限制（缺少 ffprobe 时跳过）
                dur = self._probe_duration_sec(ffprobe_path, local_path) if ffprobe_path else None
                if isinstance(dur, (int, float)) and dur > max_sec:
                    continue

                # 仅抽 1 张关键帧：优先等距抽帧（有时长时取中间帧），否则从开头抽 1 帧
                try:
                    if isinstance(dur, (int, float)) and dur > 0:
                        sampled = await sample_frames_equidistant(ffmpeg_path, local_path, float(dur), 1)
                    else:
                        sampled = await sample_frames_with_ffmpeg(ffmpeg_path, local_path, max(1, max_sec), 1)
                except Exception:
                    sampled = []

                if sampled:
                    frames.append(sampled[0])
                    # sample_* 返回的都是同一目录内的图片，清理目录即可
                    try:
                        cleanup.append(os.path.dirname(sampled[0]))
                    except Exception:
                        pass
            finally:
                # 下载到临时目录的视频文件在此处优先清理（抽帧完成后不再需要）
                if downloaded_tmp and isinstance(local_path, str) and local_path:
                    cleanup.append(local_path)
                    try:
                        if os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass

        # 去重清理列表，避免重复删除同一路径
        uniq_cleanup: List[str] = []
        seen2 = set()
        for p in cleanup:
            if isinstance(p, str) and p and p not in seen2:
                seen2.add(p)
                uniq_cleanup.append(p)

        return (frames, uniq_cleanup)

    # 文本与图片解析等通用工具已迁移至 message_utils 模块

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
        except (AttributeError, TypeError):
            pass
        # 一些 Provider 将信息挂在 config/model_config
        for attr in ("config", "model_config", "model"):
            try:
                val = getattr(provider, attr, None)
                text = str(val)
                lt = text.lower()
                if any(k in lt for k in ["image", "vision", "multimodal", "vl", "gpt-4o", "gemini", "minicpm-v"]):
                    return True
            except (AttributeError, TypeError, ValueError):
                pass
        return False

    def _select_primary_provider(self, session_provider: Any, image_urls: List[str]) -> Any:
        """根据是否包含图片选择首选 Provider。
        - 图片：优先配置 image_provider_id；否则首选会话 Provider（需具备图片能力）；否则从全部 Provider 中挑首个具备图片能力的；否则回退会话 Provider。
        - 文本：优先配置 text_provider_id；否则采用会话 Provider。
        """
        images_present = bool(image_urls)
        if images_present:
            cfg_img = self._get_config_provider("image_provider_id")
            if cfg_img is not None:
                return cfg_img
            if session_provider and self._provider_supports_image(session_provider):
                return session_provider
            for p in self.context.get_all_providers():
                if p is session_provider:
                    continue
                if self._provider_supports_image(p):
                    return p
            return session_provider
        else:
            cfg_txt = self._get_config_provider("text_provider_id")
            if cfg_txt is not None:
                return cfg_txt
            return session_provider

    async def _call_llm_with_fallback(
        self,
        primary: Any,
        session_provider: Any,
        user_prompt: str,
        system_prompt: str,
        image_urls: List[str],
    ) -> Any:
        """执行 LLM 调用与统一回退：
        - 先 primary，再 session_provider（若不同），然后遍历全部 Provider。
        - 图片场景仅尝试具备图片能力的 Provider；文本场景尝试所有 Provider。
        """
        tried = set()
        images_present = bool(image_urls)
        timeout_sec = self._get_conf_int(LLM_TIMEOUT_SEC_KEY, DEFAULT_LLM_TIMEOUT_SEC, 5, 600)

        async def _try_call(p: Any) -> Optional[Any]:
            return await asyncio.wait_for(
                p.text_chat(
                    prompt=user_prompt,
                    context=[],
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                ),
                timeout=max(5, int(timeout_sec)),
            )

        # 1) primary
        if primary is not None:
            tried.add(id(primary))
            try:
                return await _try_call(primary)
            except Exception:
                pass

        # 2) session provider
        if session_provider is not None and id(session_provider) not in tried:
            tried.add(id(session_provider))
            try:
                # 图片时校验能力
                if not images_present or self._provider_supports_image(session_provider):
                    return await _try_call(session_provider)
            except Exception:
                pass

        # 3) enumerate others
        for p in self.context.get_all_providers():
            if id(p) in tried:
                continue
            if images_present and not self._provider_supports_image(p):
                continue
            tried.add(id(p))
            try:
                resp = await _try_call(p)
                logger.info(
                    "zssm_explain: fallback %s provider succeeded",
                    "vision" if images_present else "text",
                )
                return resp
            except Exception:
                continue

        raise RuntimeError("all providers failed for current request")

    async def _extract_quoted_payload_with_videos(
        self, event: AstrMessageEvent
    ) -> Tuple[Optional[str], List[str], List[str], bool]:
        """扩展接口：提取被回复的文本/图片/视频，并标记是否为合并转发。"""
        return await extract_quoted_payload_with_videos(event)

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
            except (AttributeError, TypeError):
                continue
        return ""

    @staticmethod
    def _chain_has_at_me(chain: List[object], self_id: str) -> bool:
        """检测消息链是否 @ 了当前 Bot。"""
        if not isinstance(chain, list):
            return False
        for seg in chain:
            try:
                if isinstance(seg, Comp.At):
                    qq = getattr(seg, "qq", None)
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

    def _get_conf_int(self, key: str, default: int, min_v: int = 1, max_v: int = 120000) -> int:
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
        base_default = [ext.strip() for ext in DEFAULT_FILE_PREVIEW_EXTS.split(",") if ext.strip()]
        return build_text_exts_from_config(raw, base_default)

    def _get_file_preview_max_bytes(self) -> Optional[int]:
        """获取允许尝试内容预览的群文件最大体积（字节）。"""
        try:
            kb = self._get_conf_int(FILE_PREVIEW_MAX_SIZE_KB_KEY, DEFAULT_FILE_PREVIEW_MAX_SIZE_KB, 1, 1024 * 1024)
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
        """根据配置与当前会话人格构造系统提示词。

        当 keep_original_persona 启用时：
        - 从 AstrBot persona_manager 获取当前会话应使用的人格（v3 格式）；
        - 使用人格的 prompt 替换 DEFAULT_SYSTEM_PROMPT 的首行描述部分，
          保留原有的 Markdown 结构与输出约束说明。
        未启用或获取人格失败时，回退为默认系统提示词。
        """
        sp = build_system_prompt()

        # 未启用开关时，直接使用默认提示词
        try:
            if not self._get_conf_bool(KEEP_ORIGINAL_PERSONA_KEY, DEFAULT_KEEP_ORIGINAL_PERSONA):
                return sp
        except Exception:
            return sp

        # 仅在存在 persona_manager 时尝试获取当前会话人格
        try:
            persona_mgr = getattr(self.context, "persona_manager", None)
        except Exception:
            persona_mgr = None
        if persona_mgr is None:
            return sp

        persona_prompt: Optional[str] = None
        try:
            # get_default_persona_v3 会综合会话配置与 persona 绑定，返回 Personality（含 prompt）
            personality = await persona_mgr.get_default_persona_v3(event.unified_msg_origin)
            if isinstance(personality, dict):
                persona_prompt = personality.get("prompt")
            else:
                persona_prompt = getattr(personality, "prompt", None)
        except Exception as e:
            logger.warning(f"zssm_explain: get_default_persona_v3 failed: {e}")
            persona_prompt = None

        if not isinstance(persona_prompt, str) or not persona_prompt.strip():
            return sp

        # 使用人格 prompt 替换默认系统提示词的首行，保留其余结构化输出要求
        base_lines = sp.splitlines()
        rest_lines = base_lines[1:] if len(base_lines) > 1 else []
        merged_lines: List[str] = [persona_prompt.strip()]
        merged_lines.extend(rest_lines)
        return "\n".join(merged_lines)


    @staticmethod
    def _extract_urls_from_text(text: Optional[str]) -> List[str]:
        """兼容旧接口，委托 url_utils.extract_urls_from_text。"""
        return extract_urls_from_text(text)

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

    # ===== 输出清洗：去除思考/推理内容，仅保留结论性文本 =====
    @staticmethod
    def _sanitize_model_output(text: str) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if not s:
            return s
        # 1) 去除常见 CoT 包裹：<think>…</think>、```think``` 块
        s = re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", s)
        s = re.sub(r"(?is)```\s*(think|thinking|reasoning|cot|chain[-_ ]?of[-_ ]?thought)[\s\S]*?```", "", s)
        # 2) 若存在“答案/结论/回答/总结/Result/Final Answer”等标记，优先保留其后的内容
        markers = [
            r"答案[:：]", r"结论[:：]", r"回答[:：]", r"总结[:：]",
            r"最终答案[:：]?", r"Final Answer[:：]?", r"Result[:：]?",
        ]
        for mk in markers:
            m = re.search(rf"(?is){mk}\s*(.+)$", s)
            if m:
                s = m.group(1).strip()
                break
        # 3) 去除常见前缀段落：以“思考/推理/分析/计划/步骤/原因/链式推理/思维/思路/推导/内心独白/Reasoning/Thinking/Analysis/Plan/Steps/Rationale/Chain of Thought”开头
        s = re.sub(r"(?im)^(思考|推理|分析|计划|步骤|原因|链式推理|思维|思路|推导|内心独白)[:：].*(\n\s*\n|$)", "", s)
        s = re.sub(r"(?im)^(Reasoning|Thinking|Analysis|Plan|Steps|Rationale|Chain[-_ ]?of[-_ ]?Thought)[:：].*(\n\s*\n|$)", "", s)
        # 同时移除中文括注头如【思考】/【分析】等
        s = re.sub(r"(?im)^【(思考|分析|推理|思维|计划|步骤)】.*(\n\s*\n|$)", "", s)
        # 4) 去除开头冗余标记符与多余空白
        s = re.sub(r"^[#>*\-\s]+", "", s).strip()
        # 5) 若清洗后为空，则回退原文（避免误删全部内容）
        return s or text.strip()

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
                    logger.warning(f"zssm_explain: provider id not found for {key}={pid}: {e}")
        except Exception:
            pass
        return None

    @filter.command("zssm", alias={"知识说明", "解释"})
    async def zssm(self, event: AstrMessageEvent):
        """解释被回复消息：/zssm 或关键词触发；若携带内容则直接解释该内容，否则按回复消息逻辑。"""
        cleanup_paths: List[str] = []
        try:
            # 群聊权限控制：不满足条件则直接忽略
            try:
                if not self._is_group_allowed(event):
                    return
            except Exception:
                pass
            if self._already_handled(event):
                return
            inline = self._get_inline_content(event)
            enable_url = self._get_conf_bool(URL_DETECT_ENABLE_KEY, DEFAULT_URL_DETECT_ENABLE)

            # 1) 先解析内联内容
            if inline:
                # 若内联包含 URL，优先检测是否为 B 站视频链接，其次才走“网页摘要”流程
                urls = self._extract_urls_from_text(inline) if enable_url else []
                if urls:
                    target_url = urls[0]
                    # B 站视频链接优先走视频解释流程
                    if is_bilibili_url(target_url):
                        try:
                            async for r in self._explain_video(event, target_url):
                                yield r
                            return
                        except Exception as e:
                            logger.warning("zssm_explain: inline bilibili handle failed: %s", e)

                    timeout_sec = self._get_conf_int(
                        URL_FETCH_TIMEOUT_KEY,
                        DEFAULT_URL_FETCH_TIMEOUT,
                        2,
                        60,
                    )
                    max_chars = self._get_conf_int(
                        URL_MAX_CHARS_KEY,
                        DEFAULT_URL_MAX_CHARS,
                        min_v=1000,
                        max_v=50000,
                    )
                    cf_enable = self._get_conf_bool(
                        CF_SCREENSHOT_ENABLE_KEY,
                        DEFAULT_CF_SCREENSHOT_ENABLE,
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
                        yield self._reply_text_result(
                            event,
                            build_url_failure_message(self._last_fetch_info, cf_enable),
                        )
                        event.stop_event()
                        return
                    user_prompt, text, images = url_ctx
                else:
                    text, images = inline, []
                    user_prompt = build_user_prompt(text, images)
            else:
                # 2) 无内联时，先提取被回复消息的文本/图片/视频
                text, images, vids, from_forward = await self._extract_quoted_payload_with_videos(event)

                # 2.1) 非合并转发：视频优先走视频解释（保持原有行为）
                if vids and not from_forward:
                    async for r in self._explain_video(event, vids[0]):
                        yield r
                    return

                # 2.3) 合并转发：将其中的视频以“关键帧图片”形式纳入同一次聊天记录解释（避免直接跳过）
                if from_forward and vids:
                    try:
                        f_frames, f_cleanup = await self._extract_forward_video_keyframes(event, vids)
                        if f_frames:
                            images.extend(f_frames)
                            cleanup_paths.extend(f_cleanup)
                            note = f"（聊天记录包含 {len(vids)} 个视频，已抽取部分关键帧辅助解释）"
                            if isinstance(text, str) and text.strip():
                                text = f"{text}\n\n{note}"
                            else:
                                text = note
                    except Exception as e:
                        logger.warning("zssm_explain: forward video keyframe failed: %s", e)

                # 2.4) 无 Reply 但当前消息携带视频：补充识别（保持旧行为）
                if (not vids) and (not from_forward) and (not text) and (not images):
                    try:
                        chain_now = event.get_messages()
                    except Exception:
                        chain_now = getattr(event.message_obj, "message", []) or []
                    try:
                        vids_now = self._extract_videos_from_chain(chain_now)
                    except Exception:
                        vids_now = []
                    if vids_now:
                        async for r in self._explain_video(event, vids_now[0]):
                            yield r
                        return
                # 若仍未提取到文本/图片，尝试从群文件构造内容预览
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
                if not text and not images:
                    yield self._reply_text_result(event, "请输入要解释的内容。")
                    event.stop_event()
                    return
                # 对于来源于“合并转发”的聊天记录：始终以“整段聊天记录”为主，
                # 若其中包含网址，则在聊天记录解释的基础上，尝试附加网页摘要信息。
                urls = self._extract_urls_from_text(text) if (enable_url and text) else []
                if urls and not from_forward:
                    # 非合并转发场景，沿用原有“网页摘要优先”的行为
                    target_url = urls[0]
                    # 若为 B 站视频链接，则优先走视频解释流程
                    if is_bilibili_url(target_url):
                        try:
                            async for r in self._explain_video(event, target_url):
                                yield r
                            return
                        except Exception as e:
                            logger.warning("zssm_explain: reply bilibili handle failed: %s", e)

                    timeout_sec = self._get_conf_int(
                        URL_FETCH_TIMEOUT_KEY,
                        DEFAULT_URL_FETCH_TIMEOUT,
                        2,
                        60,
                    )
                    max_chars = self._get_conf_int(
                        URL_MAX_CHARS_KEY,
                        DEFAULT_URL_MAX_CHARS,
                        min_v=1000,
                        max_v=50000,
                    )
                    cf_enable = self._get_conf_bool(
                        CF_SCREENSHOT_ENABLE_KEY,
                        DEFAULT_CF_SCREENSHOT_ENABLE,
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
                        yield self._reply_text_result(
                            event,
                            build_url_failure_message(self._last_fetch_info, cf_enable),
                        )
                        event.stop_event()
                        return
                    user_prompt, text, images = url_ctx
                elif urls and from_forward:
                    # 合并转发 + 含网址：以聊天记录解释为主，并在可能的情况下附加网页摘要信息
                    base_prompt = build_user_prompt(text, images)
                    target_url = urls[0]
                    timeout_sec = self._get_conf_int(
                        URL_FETCH_TIMEOUT_KEY,
                        DEFAULT_URL_FETCH_TIMEOUT,
                        2,
                        60,
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
                    # 若网页抓取失败，则仅使用聊天记录解释（base_prompt）即可
                    user_prompt = base_prompt + extra_block
                else:
                    user_prompt = build_user_prompt(text, images)

            try:
                provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            except Exception as e:
                logger.error(f"zssm_explain: get provider failed: {e}")
                provider = None

            if not provider:
                yield self._reply_text_result(event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。")
                return

            system_prompt = await self._build_system_prompt(event)
            image_urls = self._filter_supported_images(images)

            try:
                # 统一选择与回退
                start_ts = time.perf_counter()
                call_provider = self._select_primary_provider(provider, image_urls)
                llm_resp = await self._call_llm_with_fallback(
                    primary=call_provider,
                    session_provider=provider,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                )

                # 走标准事件钩子：触发 OnLLMResponseEvent，让 thinking_filter 根据配置统一过滤/展示思考
                try:
                    await call_event_hook(event, EventType.OnLLMResponseEvent, llm_resp)
                except Exception:
                    pass

                # 优先使用经钩子可能更新过的 completion_text；否则回退原解析
                reply_text = None
                try:
                    ct = getattr(llm_resp, "completion_text", None)
                    if isinstance(ct, str) and ct.strip():
                        reply_text = ct.strip()
                except Exception:
                    reply_text = None
                if not reply_text:
                    reply_text = self._pick_llm_text(llm_resp)

                # 根据 AstrBot 配置是否展示思考，决定是否再做插件侧清洗
                show_reasoning = False
                try:
                    cfg = self.context.get_config(umo=event.unified_msg_origin) or {}
                    ps = cfg.get("provider_settings", {})
                    show_reasoning = bool(ps.get("display_reasoning_text", False))
                except Exception:
                    show_reasoning = False
                if not show_reasoning:
                    reply_text = self._sanitize_model_output(reply_text)

                elapsed = None
                try:
                    elapsed = time.perf_counter() - start_ts
                except Exception:
                    elapsed = None
                reply_text = self._format_explain_output(reply_text, elapsed_sec=elapsed)
                yield self._reply_text_result(event, reply_text)
                # 防止后续流程重复处理当前事件
                try:
                    event.stop_event()
                except Exception:
                    pass
            except asyncio.TimeoutError:
                yield self._reply_text_result(event, "解释超时，请稍后重试或换一个模型提供商。")
                try:
                    event.stop_event()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"zssm_explain: LLM 调用失败: {e}")
                yield self._reply_text_result(event, "解释失败：LLM 或图片转述模型调用异常，请稍后再试或联系管理员。")
                try:
                    event.stop_event()
                except Exception:
                    pass
        except Exception as e:
            logger.error("zssm_explain: handler crashed: %s", e)
            yield self._reply_text_result(event, "解释失败：插件内部异常，请稍后再试或联系管理员。")
            try:
                event.stop_event()
            except Exception:
                pass
        finally:
            # 清理合并转发视频抽帧的临时文件/目录（不影响普通图片 URL）
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
        """关键词触发：忽略常见前缀/Reply/At 等，检测首个 Plain 段的 zssm。
        避免与 /zssm 指令重复：若以 /zssm 开头则交由指令处理。
        """
        # 配置开关：允许用户关闭正则关键词触发，仅保留 /zssm 指令。
        if not self._get_conf_bool(KEYWORD_ZSSM_ENABLE_KEY, True):
            return
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
            chain = getattr(event.message_obj, "message", []) if hasattr(event, "message_obj") else []
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
            if re.match(r"^\s*/\s*zssm(\s|$)", hs, re.I):
                return
            if at_me and re.match(r"^zssm(\s|$)", hs, re.I):
                return
            if self._is_zssm_trigger(hs):
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
            if at_me and re.match(r"^zssm(\s|$)", t, re.I):
                return
            if self._is_zssm_trigger(t):
                async for r in self.zssm(event):
                    yield r
