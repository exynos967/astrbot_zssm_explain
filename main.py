from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict
import os
import asyncio
import re
import json
import shutil
import tempfile
import subprocess
from urllib.parse import urlparse, unquote
from html import unescape

try:
    import aiohttp  # 优先使用异步 HTTP 客户端
except Exception:  # pragma: no cover
    aiohttp = None  # 运行环境若无 aiohttp，将回退到线程内 urllib

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.core.star.star_handler import EventType
from astrbot.core.pipeline.context_utils import call_event_hook

# === 可编辑的默认提示词（用户可直接在此处修改） ===
DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文助理，擅长从被引用的消息中提炼含义、意图和注意事项。"
)

DEFAULT_TEXT_USER_PROMPT = (
    "请解释这条被回复的消息的含义，输出简洁不超过100字。\n"
    "原始文本：\n{text}"
)

DEFAULT_IMAGE_USER_PROMPT = (
    "请解释这条被回复的消息/图片的含义，输出简洁不超过100字。\n"
    "{text_block}\n包含图片：若无法直接读取图片，请结合上下文或文件名描述。"
)

DEFAULT_URL_USER_PROMPT = (
    "你将看到一个网页的关键信息，请输出简版摘要（2-8句，中文）。"
    "避免口水话，保留事实与结论，适当含链接上下文。\n"
    "网址: {url}\n"
    "标题: {title}\n"
    "描述: {desc}\n"
    "正文片段: \n{snippet}"
)

DEFAULT_VIDEO_USER_PROMPT = (
    "请解释这段视频的主要内容，输出简洁不超过100字。仅依据提供的关键帧与音频转写（如有）作答；若信息不足请明确说明‘无法判断’，不要编造未出现的内容。\n"
    "{meta_block}\n{asr_block}"
)

# URL 识别/抓取的默认参数（可通过插件配置覆盖）
URL_DETECT_ENABLE_KEY = "enable_url_detect"
URL_FETCH_TIMEOUT_KEY = "url_timeout_sec"
URL_MAX_CHARS_KEY = "url_max_chars"
GROUP_LIST_MODE_KEY = "group_list_mode"
GROUP_LIST_KEY = "group_list"
VIDEO_PROVIDER_ID_KEY = "video_provider_id"
ENABLE_VIDEO_EXPLAIN_KEY = "enable_video_explain"
VIDEO_FRAME_INTERVAL_SEC_KEY = "video_frame_interval_sec"
VIDEO_FRAME_COUNT_LIMIT_KEY = "video_frame_count_limit"
VIDEO_ASR_ENABLE_KEY = "video_asr_enable"
VIDEO_MAX_DURATION_SEC_KEY = "video_max_duration_sec"
VIDEO_MAX_SIZE_MB_KEY = "video_max_size_mb"
FFMPEG_PATH_KEY = "ffmpeg_path"
ASR_PROVIDER_ID_KEY = "asr_provider_id"

DEFAULT_URL_DETECT_ENABLE = True
DEFAULT_URL_FETCH_TIMEOUT = 8
DEFAULT_URL_MAX_CHARS = 6000
DEFAULT_ENABLE_VIDEO_EXPLAIN = True
DEFAULT_VIDEO_FRAME_INTERVAL_SEC = 6
DEFAULT_VIDEO_FRAME_COUNT_LIMIT = 6
DEFAULT_VIDEO_ASR_ENABLE = False
DEFAULT_VIDEO_MAX_DURATION_SEC = 120
DEFAULT_VIDEO_MAX_SIZE_MB = 50
DEFAULT_FFMPEG_PATH = "ffmpeg"




@register(
    "zssm_explain",
    "薄暝",
    "zssm，支持关键词“zssm”（忽略前缀）与“zssm + 内容”直接解释；引用消息（含@）正常处理；支持 QQ 合并转发；未回复仅发 zssm 时提示；默认提示词可在 main.py 顶部修改。",
    "0.8.0",
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
                # 兼容以逗号/空白分隔的字符串
                raw = [x.strip() for x in re.split(r"[\s,]+", v) if x.strip()]
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
        """
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        if not gid:
            return True  # 非群聊或无法识别，放行

        mode = self._get_conf_str(GROUP_LIST_MODE_KEY, "none").lower()
        if mode not in ("whitelist", "blacklist", "none"):
            mode = "none"
        glist = self._get_conf_list_str(GROUP_LIST_KEY)

        if mode == "whitelist":
            return str(gid) in glist if glist else False
        if mode == "blacklist":
            return str(gid) not in glist if glist else True
        return True

    # ===== 视频相关工具 =====
    @staticmethod
    def _extract_videos_from_chain(chain: List[object]) -> List[str]:
        videos: List[str] = []
        if not isinstance(chain, list):
            return videos
        def _looks_like_video(name_or_url: str) -> bool:
            if not isinstance(name_or_url, str) or not name_or_url:
                return False
            s = name_or_url.lower()
            return any(
                s.endswith(ext)
                for ext in (
                    ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".ts", ".mpeg", ".mpg", ".3gp"
                )
            )
        for seg in chain:
            try:
                if hasattr(Comp, "Video") and isinstance(seg, getattr(Comp, "Video")):
                    f = getattr(seg, "file", None)
                    u = getattr(seg, "url", None)
                    if isinstance(f, str) and f:
                        videos.append(f)
                    elif isinstance(u, str) and u:
                        videos.append(u)
                elif hasattr(Comp, "File") and isinstance(seg, getattr(Comp, "File")):
                    # 大视频可能以 File 形式承载
                    u = getattr(seg, "url", None)
                    f = getattr(seg, "file", None)
                    n = getattr(seg, "name", None)
                    cand = None
                    if isinstance(u, str) and u and _looks_like_video(u):
                        cand = u
                    elif isinstance(f, str) and f and (_looks_like_video(f) or os.path.isabs(f)):
                        cand = f
                    elif isinstance(n, str) and n and _looks_like_video(n) and isinstance(f, str) and f:
                        cand = f
                    if isinstance(cand, str) and cand:
                        videos.append(cand)
                elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                    content = getattr(seg, "content", None)
                    if isinstance(content, list):
                        videos.extend(ZssmExplain._extract_videos_from_chain(content))
                elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                    nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            c = getattr(node, "content", None)
                            if isinstance(c, list):
                                videos.extend(ZssmExplain._extract_videos_from_chain(c))
                elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                    nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            c = getattr(node, "content", None)
                            if isinstance(c, list):
                                videos.extend(ZssmExplain._extract_videos_from_chain(c))
            except Exception:
                continue
        return videos

    async def _extract_videos_from_event(self, event: AstrMessageEvent) -> List[str]:
        # 先找被回复消息中的视频
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
        if reply_comp:
            for attr in ("message", "origin", "content"):
                payload = getattr(reply_comp, attr, None)
                if isinstance(payload, list):
                    vids = self._extract_videos_from_chain(payload)
                    if vids:
                        return vids
            # 无内嵌内容时，尝试通过平台能力（OneBot/Napcat）用 message_id 拉取原消息
            reply_id = self._get_reply_message_id(reply_comp)
            platform_name = None
            try:
                platform_name = event.get_platform_name()
            except Exception:
                platform_name = None
            if reply_id and platform_name == "aiocqhttp" and hasattr(event, "bot"):
                try:
                    data = await event.bot.get_msg(message_id=int(reply_id))
                    vids = self._extract_videos_from_onebot_message_payload(data)
                    if vids:
                        return vids
                except Exception:
                    pass
        # 没有 Reply 或 Reply 无视频，则直接从当前消息链找视频
        return self._extract_videos_from_chain(chain)

    @staticmethod
    def _extract_videos_from_onebot_message_payload(payload: Any) -> List[str]:
        """从 OneBot/Napcat get_msg/get_forward_msg 返回的 payload 中提取视频 URL/路径。"""
        videos: List[str] = []
        data = ZssmExplain._ob_data(payload) if isinstance(payload, dict) else {}
        if isinstance(data, dict):
            # 常见字段 message/messages/nodes
            candidates = data.get("message") or data.get("messages") or data.get("nodes") or data.get("nodeList")
            if isinstance(candidates, list):
                for seg in candidates:
                    try:
                        if isinstance(seg, dict):
                            # 可能是消息段，或转发节点
                            if "type" in seg and "data" in seg:
                                t = seg.get("type")
                                d = seg.get("data") or {}
                                if isinstance(d, dict):
                                    if t == "video":
                                        url = d.get("url") or d.get("file")
                                        if isinstance(url, str) and url:
                                            videos.append(url)
                                    elif t == "file":
                                        # 文件类型里也可能是视频
                                        url = d.get("url") or d.get("file")
                                        name = d.get("name") or d.get("filename")
                                        def _looks_like_video(name_or_url: str) -> bool:
                                            if not isinstance(name_or_url, str) or not name_or_url:
                                                return False
                                            s = name_or_url.lower()
                                            return any(s.endswith(ext) for ext in (
                                                ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".ts", ".mpeg", ".mpg", ".3gp"
                                            ))
                                        if isinstance(url, str) and url and _looks_like_video(url):
                                            videos.append(url)
                                        elif isinstance(name, str) and _looks_like_video(name) and isinstance(url, str) and url:
                                            videos.append(url)
                            else:
                                # 转发节点内的 message/content 列表
                                content = seg.get("content") or seg.get("message")
                                if isinstance(content, list):
                                    inner = ZssmExplain._extract_videos_from_onebot_message_payload({"message": content})
                                    videos.extend(inner)
                    except Exception:
                        continue
        return videos

    def _resolve_ffmpeg(self) -> Optional[str]:
        # 配置优先
        path = self._get_conf_str(FFMPEG_PATH_KEY, DEFAULT_FFMPEG_PATH)
        if path and shutil.which(path):
            return shutil.which(path)
        # 尝试系统 ffmpeg
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        # 尝试 imageio-ffmpeg
        try:
            import imageio_ffmpeg
            p = imageio_ffmpeg.get_ffmpeg_exe()
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
        return None

    def _resolve_ffprobe(self) -> Optional[str]:
        # 同目录的 ffprobe（若 imageio-ffmpeg 提供）或系统 ffprobe
        sys_ffprobe = shutil.which("ffprobe")
        if sys_ffprobe:
            return sys_ffprobe
        # 粗略尝试：若 ffmpeg 同目录存在 ffprobe
        ff = self._resolve_ffmpeg()
        if ff:
            cand = os.path.join(os.path.dirname(ff), "ffprobe")
            if os.path.exists(cand):
                return cand
        return None

    async def _download_to_temp(self, url: str, size_mb_limit: int) -> Optional[str]:
        # 为 URL 提取安全的短扩展名，避免把查询串当后缀导致路径过长
        def _safe_ext_from_url(u: str) -> str:
            try:
                path = urlparse(u).path
                base = os.path.basename(unquote(path))
                ext = os.path.splitext(base)[1]
                # 限制扩展名长度并校验字符
                if isinstance(ext, str):
                    ext = ext[:8]
                if not ext or not re.match(r"^\.[A-Za-z0-9]{1,6}$", ext):
                    # 尝试常见视频后缀匹配
                    lower = base.lower()
                    for cand in (".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".flv", ".wmv"):
                        if lower.endswith(cand):
                            return cand
                    return ".bin"
                return ext
            except Exception:
                return ".bin"

        ext = _safe_ext_from_url(url)
        tmp = tempfile.NamedTemporaryFile(prefix="zssm_video_", suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()
        max_bytes = size_mb_limit * 1024 * 1024
        if aiohttp is not None:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=20) as resp:
                        if resp.status != 200:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                            return None
                        # 内容长度预判
                        cl = resp.headers.get("Content-Length")
                        if cl and cl.isdigit() and int(cl) > max_bytes:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                            return None
                        total = 0
                        with open(tmp_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                if not chunk:
                                    break
                                total += len(chunk)
                                if total > max_bytes:
                                    try:
                                        f.close()
                                    except Exception:
                                        pass
                                    try:
                                        os.remove(tmp_path)
                                    except Exception:
                                        pass
                                    return None
                                f.write(chunk)
                return tmp_path if os.path.exists(tmp_path) else None
            except Exception:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return None
        # urllib 回退
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=20) as r, open(tmp_path, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        try:
                            f.close()
                        except Exception:
                            pass
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        return None
                    f.write(chunk)
            return tmp_path if os.path.exists(tmp_path) else None
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None

    def _probe_duration_sec(self, ffprobe_path: Optional[str], video_path: str) -> Optional[float]:
        if not ffprobe_path:
            return None
        try:
            res = subprocess.run(
                [ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if res.returncode != 0:
                return None
            data = json.loads(res.stdout.decode("utf-8", errors="ignore") or "{}")
            dur = None
            if isinstance(data, dict):
                fmt = data.get("format")
                if isinstance(fmt, dict):
                    d = fmt.get("duration")
                    try:
                        dur = float(d)
                    except Exception:
                        dur = None
            return dur
        except Exception:
            return None

    async def _sample_frames_with_ffmpeg(self, ffmpeg_path: str, video_path: str, interval_sec: int, count_limit: int) -> List[str]:
        out_dir = tempfile.mkdtemp(prefix="zssm_frames_")
        out_tpl = os.path.join(out_dir, "frame_%03d.jpg")
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vf", f"fps=1/{max(1, interval_sec)}",
            "-frames:v", str(max(1, count_limit)),
            "-qscale:v", "2",
            out_tpl,
        ]
        loop = asyncio.get_running_loop()
        def _run():
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        res = await loop.run_in_executor(None, _run)
        if res.returncode != 0:
            # 失败时删除目录
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            logger.error("zssm_explain: ffmpeg fps-sampler failed (code=%s)", res.returncode)
            raise RuntimeError("ffmpeg sample frames failed")
        frames = []
        try:
            for name in sorted(os.listdir(out_dir)):
                if name.lower().endswith('.jpg'):
                    frames.append(os.path.join(out_dir, name))
        except Exception:
            pass
        if not frames:
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            raise RuntimeError("no frames generated")
        return frames

    async def _sample_frames_equidistant(self, ffmpeg_path: str, video_path: str, duration_sec: float, count_limit: int) -> List[str]:
        """按等距时间点抽帧，覆盖全片。选择 N 个时间点：t_i = (i/(N+1))*duration。"""
        N = max(1, int(count_limit))
        out_dir = tempfile.mkdtemp(prefix="zssm_frames_")
        loop = asyncio.get_running_loop()
        frames: List[str] = []
        times: List[float] = []
        try:
            total = max(0.0, float(duration_sec))
            for i in range(1, N + 1):
                t = (i / (N + 1.0)) * total
                times.append(t)
            logger.info("zssm_explain: equidistant times=%s", [round(x, 2) for x in times])
            for idx, t in enumerate(times, start=1):
                out_path = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
                cmd = [
                    ffmpeg_path, "-y",
                    "-ss", f"{max(0.0, t):.3f}",
                    "-i", video_path,
                    "-frames:v", "1",
                    "-qscale:v", "2",
                    out_path,
                ]
                def _run_one():
                    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                res = await loop.run_in_executor(None, _run_one)
                if res.returncode == 0 and os.path.exists(out_path):
                    frames.append(out_path)
                else:
                    logger.warning("zssm_explain: ffmpeg sample at %.3fs failed (code=%s)", t, res.returncode)
        except Exception as e:
            logger.error("zssm_explain: equidistant sampler error: %s", e)
        if not frames:
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            raise RuntimeError("no frames generated by equidistant sampler")
        return frames

    async def _extract_audio_wav(self, ffmpeg_path: str, video_path: str) -> Optional[str]:
        out_fd, out_path = tempfile.mkstemp(prefix="zssm_audio_", suffix=".wav")
        os.close(out_fd)
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            out_path,
        ]
        loop = asyncio.get_running_loop()
        def _run():
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        res = await loop.run_in_executor(None, _run)
        if res.returncode != 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None
        return out_path if os.path.exists(out_path) else None

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
        # 配置检查
        if not self._get_conf_bool(ENABLE_VIDEO_EXPLAIN_KEY, DEFAULT_ENABLE_VIDEO_EXPLAIN):
            yield self._reply_text_result(event, "视频解释功能未启用。")
            return
        ffmpeg_path = self._resolve_ffmpeg()
        if not ffmpeg_path:
            yield self._reply_text_result(event, "未检测到 ffmpeg，请安装系统 ffmpeg 或 Python 包 imageio-ffmpeg，并在插件配置中设置 ffmpeg_path。")
            return
        logger.info("zssm_explain: video start src=%s ffmpeg=%s", (str(video_src)[:128] if video_src else ""), ffmpeg_path)

        # 统一获取本地文件路径（支持 http/https 下载）
        max_mb = self._get_conf_int(VIDEO_MAX_SIZE_MB_KEY, DEFAULT_VIDEO_MAX_SIZE_MB, 1, 512)
        local_path = None
        if isinstance(video_src, str) and video_src.lower().startswith(("http://", "https://")):
            local_path = await self._download_to_temp(video_src, max_mb)
            if not local_path:
                yield self._reply_text_result(event, f"视频下载失败或超过大小限制（>{max_mb}MB）。")
                return
        else:
            # 假定为本地路径
            if not (isinstance(video_src, str) and os.path.isabs(video_src) and os.path.exists(video_src)):
                yield self._reply_text_result(event, "无法读取该视频源，请确认路径或链接有效。")
                return
            # 大小检查
            try:
                sz = os.path.getsize(video_src)
                if sz > max_mb * 1024 * 1024:
                    yield self._reply_text_result(event, f"视频大小超过限制（>{max_mb}MB），请压缩或截取片段后重试。")
                    return
            except Exception:
                pass
            local_path = video_src

        # 时长检查（可选，缺少 ffprobe 时跳过）
        max_sec = self._get_conf_int(VIDEO_MAX_DURATION_SEC_KEY, DEFAULT_VIDEO_MAX_DURATION_SEC, 10, 3600)
        dur = self._probe_duration_sec(self._resolve_ffprobe(), local_path)
        logger.info("zssm_explain: probed duration=%s (max=%s)", dur if dur is not None else "unknown", max_sec)
        if isinstance(dur, (int, float)) and dur > max_sec:
            yield self._reply_text_result(event, f"视频时长超过限制（>{max_sec}s），请截取片段后重试。")
            return

        # 抽帧
        interval = self._get_conf_int(VIDEO_FRAME_INTERVAL_SEC_KEY, DEFAULT_VIDEO_FRAME_INTERVAL_SEC, 1, 120)
        limit = self._get_conf_int(VIDEO_FRAME_COUNT_LIMIT_KEY, DEFAULT_VIDEO_FRAME_COUNT_LIMIT, 1, 16)
        try:
            if isinstance(dur, (int, float)) and dur > 0:
                frames = await self._sample_frames_equidistant(ffmpeg_path, local_path, float(dur), limit)
            else:
                frames = await self._sample_frames_with_ffmpeg(ffmpeg_path, local_path, interval, limit)
        except Exception as e:
            yield self._reply_text_result(event, f"抽帧失败：{e}")
            return
        logger.info("zssm_explain: sampled %d frames", len(frames))
        image_urls = self._filter_supported_images(frames)
        if not image_urls:
            yield self._reply_text_result(event, "未能生成可用关键帧，请检查 ffmpeg 或更换视频后重试。")
            return

        # 可选 ASR
        asr_text = None
        if self._get_conf_bool(VIDEO_ASR_ENABLE_KEY, DEFAULT_VIDEO_ASR_ENABLE):
            try:
                wav = await self._extract_audio_wav(ffmpeg_path, local_path)
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
                            logger.info("zssm_explain: asr text length=%s", len(asr_text) if isinstance(asr_text, str) else 0)
                        except Exception:
                            asr_text = None
                    try:
                        os.remove(wav)
                    except Exception:
                        pass
            except Exception:
                asr_text = None

        # 组装并调用 LLM
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None
        if not provider:
            yield self._reply_text_result(event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。")
            return
        system_prompt = self._build_system_prompt()
        meta = {
            "name": os.path.basename(local_path),
            "duration": dur if isinstance(dur, (int, float)) else None,
            "frames": len(image_urls),
        }
        user_prompt = self._build_video_user_prompt(meta, asr_text)

        try:
            call_provider = self._select_video_provider(provider, image_urls)
            try:
                pid = getattr(call_provider, "id", None) or getattr(call_provider, "provider_id", None) or call_provider.__class__.__name__
            except Exception:
                pid = None
            logger.info("zssm_explain: llm provider=%s, images=%d", pid or "unknown", len(image_urls))
            llm_resp = await self._call_llm_with_fallback(
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
                reply_text = self._pick_llm_text(llm_resp)
            show_reasoning = False
            try:
                cfg = self.context.get_config(umo=event.unified_msg_origin) or {}
                ps = cfg.get("provider_settings", {})
                show_reasoning = bool(ps.get("display_reasoning_text", False))
            except Exception:
                show_reasoning = False
            if not show_reasoning:
                reply_text = self._sanitize_model_output(reply_text)
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

    @staticmethod
    def _extract_text_and_images_from_chain(chain: List[object]) -> Tuple[str, List[str]]:
        """从一段消息链中提取纯文本与图片地址/路径；支持合并转发节点的递归提取。"""
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
                elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                    content = getattr(seg, "content", None)
                    if isinstance(content, list):
                        t2, i2 = ZssmExplain._extract_text_and_images_from_chain(content)
                        if t2:
                            texts.append(t2)
                        images.extend(i2)
                elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                    nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            c = getattr(node, "content", None)
                            if isinstance(c, list):
                                t2, i2 = ZssmExplain._extract_text_and_images_from_chain(c)
                                if t2:
                                    texts.append(t2)
                                images.extend(i2)
                elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                    nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            c = getattr(node, "content", None)
                            if isinstance(c, list):
                                t2, i2 = ZssmExplain._extract_text_and_images_from_chain(c)
                                if t2:
                                    texts.append(t2)
                                images.extend(i2)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"zssm_explain: parse chain segment failed: {e}")
        return ("\n".join([t for t in texts if t]).strip(), images)

    @staticmethod
    def _try_extract_from_reply_component(reply_comp: object) -> Tuple[Optional[str], List[str]]:
        """尽量从 Reply 组件中得到被引用消息的文本与图片。"""
        for attr in ("message", "origin", "content"):
            payload = getattr(reply_comp, attr, None)
            if isinstance(payload, list):
                return ZssmExplain._extract_text_and_images_from_chain(payload)
        return (None, [])

    @staticmethod
    def _get_reply_message_id(reply_comp: object) -> Optional[str]:
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

    @staticmethod
    def _ob_data(obj: Any) -> Dict[str, Any]:
        """OneBot 风格响应可能包裹在 data 字段中，展开后返回字典。"""
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, dict):
                return data
            return obj
        return {}

    @staticmethod
    def _extract_from_onebot_message_payload(payload: Any) -> Tuple[str, List[str]]:
        """从 OneBot/Napcat get_msg 返回的 payload 中提取文本与图片；识别 forward/nodes 由上层处理。"""
        texts: List[str] = []
        images: List[str] = []
        data = ZssmExplain._ob_data(payload) if isinstance(payload, dict) else {}
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
        # 无法解析出有意义的文本，返回空字符串而非对象字符串，避免误导 LLM
        logger.warning("zssm_explain: failed to extract text from OneBot payload; fallback to empty text")
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

        async def _try_call(p: Any) -> Optional[Any]:
            return await p.text_chat(
                prompt=user_prompt,
                context=[],
                system_prompt=system_prompt,
                image_urls=image_urls,
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
                data = ZssmExplain._ob_data(ret)
                # 先解析普通文本/图片
                t2, imgs2 = self._extract_from_onebot_message_payload(data)
                agg_texts: List[str] = [t2] if t2 else []
                agg_imgs: List[str] = list(imgs2)
                # 检测是否包含合并转发段，尝试调用 get_forward_msg 拉取节点
                try:
                    msg_list = data.get("message") if isinstance(data, dict) else None
                    if isinstance(msg_list, list):
                        for seg in msg_list:
                            if not isinstance(seg, dict):
                                continue
                            if seg.get("type") in ("forward", "forward_msg", "nodes"):
                                d = seg.get("data", {}) if isinstance(seg.get("data"), dict) else {}
                                fid = d.get("id")
                                if isinstance(fid, str) and fid:
                                    try:
                                        fwd = await event.bot.api.call_action("get_forward_msg", id=fid)
                                        ft, fi = self._extract_from_onebot_forward_payload(fwd)
                                        if ft:
                                            agg_texts.append(ft)
                                        if fi:
                                            agg_imgs.extend(fi)
                                    except Exception as fe:
                                        logger.warning(f"zssm_explain: get_forward_msg failed: {fe}")
                except Exception:
                    pass
                if agg_texts or agg_imgs:
                    logger.info("zssm_explain: fetched origin via get_msg")
                    return ("\n".join([x for x in agg_texts if x]).strip(), agg_imgs)
            except Exception as e:
                logger.warning(f"zssm_explain: get_msg failed: {e}")

        logger.info("zssm_explain: reply component found but no embedded origin; consider platform API to fetch by id")
        return (None, [])

    @staticmethod
    def _extract_from_onebot_forward_payload(payload: Any) -> Tuple[str, List[str]]:
        """解析 OneBot get_forward_msg 返回的 messages/nodes 列表，汇总文本与图片。"""
        texts: List[str] = []
        images: List[str] = []
        data = ZssmExplain._ob_data(payload) if isinstance(payload, dict) else {}
        if isinstance(data, dict):
            msgs = (
                data.get("messages")
                or data.get("message")
                or data.get("nodes")
                or data.get("nodeList")
            )
            if isinstance(msgs, list):
                for node in msgs:
                    try:
                        content = None
                        if isinstance(node, dict):
                            content = node.get("content") or node.get("message")
                        if isinstance(content, list):
                            t, i = ZssmExplain._extract_from_onebot_message_payload({"message": content})
                            if t:
                                texts.append(t)
                            images.extend(i)
                    except Exception:
                        continue
        return ("\n".join([x for x in texts if x]).strip(), images)

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

    @staticmethod
    def _extract_urls_from_text(text: Optional[str]) -> List[str]:
        if not isinstance(text, str) or not text:
            return []
        # 基本 URL 正则：匹配 http/https 及常见顶级域名
        url_pattern = re.compile(r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)", re.IGNORECASE)
        urls = [m.group(1) for m in url_pattern.finditer(text)]
        # 去重并保持顺序
        seen = set()
        uniq = []
        for u in urls:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        return uniq

    async def _fetch_html(self, url: str, timeout_sec: int) -> Optional[str]:
        """获取网页 HTML 文本：优先 aiohttp；回退 urllib 在线程池中执行，避免阻塞事件循环。"""
        def _mark(status: Optional[int] = None, headers: Optional[Dict[str, str]] = None, text_hint: Optional[str] = None, via: str = "", error: Optional[str] = None):
            headers = headers or {}
            # 简易 Cloudflare 识别：server=cloudflare 或存在 cf-* 响应头；或文本提示
            server = str(headers.get("server", "")).lower()
            cf_header = any(h.lower().startswith("cf-") for h in headers.keys()) if headers else False
            text_has_cf = False
            if isinstance(text_hint, str):
                tl = text_hint.lower()
                if "cloudflare" in tl or "attention required" in tl or "enable javascript and cookies" in tl:
                    text_has_cf = True
            is_cf = ("cloudflare" in server) or cf_header or text_has_cf
            self._last_fetch_info = {
                "url": url,
                "status": status,
                "cloudflare": is_cf,
                "via": via,
                "error": error,
            }

        async def _aiohttp_fetch() -> Optional[str]:
            if aiohttp is None:
                return None
            try:
                async with aiohttp.ClientSession(headers={
                    "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)"
                }) as session:
                    async with session.get(url, timeout=timeout_sec, allow_redirects=True) as resp:
                        status = int(resp.status)
                        hdrs = {k: v for k, v in resp.headers.items()}
                        if 200 <= status < 400:
                            text = await resp.text()
                            _mark(status=status, headers=hdrs, text_hint=text[:512], via="aiohttp")
                            return text
                        # 非 2xx/3xx，记录并返回 None
                        _mark(status=status, headers=hdrs, text_hint=None, via="aiohttp")
                        return None
            except Exception as e:
                logger.warning(f"zssm_explain: aiohttp fetch failed: {e}")
                _mark(status=None, headers=None, text_hint=None, via="aiohttp", error=str(e))
                return None

        async def _urllib_fetch() -> Optional[str]:
            import urllib.request
            import urllib.error
            def _do() -> Optional[str]:
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "AstrBot-zssm/1.0 (+https://github.com/xiaoxi68/astrbot_zssm_explain)"
                    })
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        data = resp.read()
                        # 尝试从头部/内容推断编码；回退 utf-8
                        enc = resp.headers.get_content_charset() or "utf-8"
                        try:
                            text = data.decode(enc, errors="replace")
                            _mark(status=getattr(resp, "status", 200), headers=dict(resp.headers), text_hint=text[:512], via="urllib")
                            return text
                        except Exception:
                            text = data.decode("utf-8", errors="replace")
                            _mark(status=getattr(resp, "status", 200), headers=dict(resp.headers), text_hint=text[:512], via="urllib")
                            return text
                except urllib.error.HTTPError as e:  # 带状态码与响应头
                    try:
                        body = e.read() or b""
                        hint = body.decode("utf-8", errors="ignore")[:512]
                    except Exception:
                        hint = None
                    hdrs = dict(getattr(e, "headers", {}) or {})
                    _mark(status=getattr(e, "code", None), headers=hdrs, text_hint=hint, via="urllib", error=str(e))
                    logger.warning(f"zssm_explain: urllib fetch failed: {e}")
                    return None
                except Exception as e:
                    _mark(status=None, headers=None, text_hint=None, via="urllib", error=str(e))
                    logger.warning(f"zssm_explain: urllib fetch failed: {e}")
                    return None
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _do)

        html = await _aiohttp_fetch()
        if html is not None:
            return html
        return await _urllib_fetch()

    @staticmethod
    def _strip_html(html: str) -> str:
        # 去掉 script/style
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        # 基础去标签
        text = re.sub(r"<[^>]+>", " ", html)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return unescape(re.sub(r"\s+", " ", m.group(1)).strip())
        return ""

    @staticmethod
    def _extract_meta_desc(html: str) -> str:
        # 常见 meta 描述字段
        for name in [
            r"name=\"description\"",
            r"property=\"og:description\"",
            r"name=\"twitter:description\"",
        ]:
            m = re.search(rf"<meta[^>]+{name}[^>]+content=\"(.*?)\"[^>]*>", html, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return unescape(re.sub(r"\s+", " ", m.group(1)).strip())
        return ""

    def _build_url_user_prompt(self, url: str, html: str) -> Tuple[str, str]:
        title = self._extract_title(html)
        desc = self._extract_meta_desc(html)
        plain = self._strip_html(html)
        max_chars = self._get_conf_int(URL_MAX_CHARS_KEY, DEFAULT_URL_MAX_CHARS, min_v=1000, max_v=50000)
        snippet = plain[:max_chars]
        user_prompt = DEFAULT_URL_USER_PROMPT.format(url=url, title=title or "(无)", desc=desc or "(无)", snippet=snippet)
        return user_prompt, title or ""

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
            # 若内联包含 URL，优先走“网页摘要”流程
            urls = self._extract_urls_from_text(inline) if enable_url else []
            if urls:
                target_url = urls[0]
                timeout_sec = self._get_conf_int(URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60)
                html = await self._fetch_html(target_url, timeout_sec)
                if not html:
                    info = getattr(self, "_last_fetch_info", {}) or {}
                    if info.get("cloudflare"):
                        logger.warning(
                            "zssm_explain: Cloudflare protection detected for URL: %s (status=%s, via=%s)",
                            target_url, info.get("status"), info.get("via")
                        )
                        yield self._reply_text_result(event, "目标站点启用 Cloudflare 防护，暂无法抓取网页内容。请稍后重试，或发送页面截图/复制关键段落。")
                    else:
                        yield self._reply_text_result(event, "网页获取失败或不支持，请稍后再试或检查链接可访问性。")
                    event.stop_event()
                    return
                user_prompt, _page_title = self._build_url_user_prompt(target_url, html)
                text, images = None, []  # 网页模式不直接传原文文本
            else:
                text, images = inline, []
                user_prompt = self._build_user_prompt(text, images)
        else:
            # 2) 无内联时，优先检测是否存在视频
            try:
                vids = await self._extract_videos_from_event(event)
            except Exception:
                vids = []
            if vids:
                async for r in self._explain_video(event, vids[0]):
                    yield r
                return
            # 其次，尝试被回复消息中的文本/图片
            text, images = await self._extract_quoted_payload(event)
            if not text and not images:
                yield self._reply_text_result(event, "请输入要解释的内容。")
                event.stop_event()
                return
            urls = self._extract_urls_from_text(text) if (enable_url and text) else []
            if urls:
                target_url = urls[0]
                timeout_sec = self._get_conf_int(URL_FETCH_TIMEOUT_KEY, DEFAULT_URL_FETCH_TIMEOUT, 2, 60)
                html = await self._fetch_html(target_url, timeout_sec)
                if not html:
                    info = getattr(self, "_last_fetch_info", {}) or {}
                    if info.get("cloudflare"):
                        logger.warning(
                            "zssm_explain: Cloudflare protection detected for URL: %s (status=%s, via=%s)",
                            target_url, info.get("status"), info.get("via")
                        )
                        yield self._reply_text_result(event, "目标站点启用 Cloudflare 防护，暂无法抓取网页内容。请稍后重试，或发送页面截图/复制关键段落。")
                    else:
                        yield self._reply_text_result(event, "网页获取失败或不支持，请稍后再试或检查链接可访问性。")
                    event.stop_event()
                    return
                user_prompt, _page_title = self._build_url_user_prompt(target_url, html)
                text, images = None, []
            else:
                user_prompt = self._build_user_prompt(text, images)

        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None

        if not provider:
            yield self._reply_text_result(event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。")
            return

        system_prompt = self._build_system_prompt()
        image_urls = self._filter_supported_images(images)

        try:
            # 统一选择与回退
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
            yield self._reply_text_result(event, reply_text)
            # 防止后续流程重复处理当前事件
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

    async def terminate(self):
        return

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def keyword_zssm(self, event: AstrMessageEvent):
        """关键词触发：忽略常见前缀/Reply/At 等，检测首个 Plain 段的 zssm。
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
