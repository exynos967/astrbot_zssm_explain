from __future__ import annotations

from typing import List, Optional, Any, Dict

import json
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse, unquote

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

from .message_utils import get_reply_message_id, ob_data


def extract_videos_from_chain(chain: List[object]) -> List[str]:
    """从消息链中递归提取视频相关 URL / 路径。"""
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
                ".mp4",
                ".mov",
                ".m4v",
                ".avi",
                ".webm",
                ".mkv",
                ".flv",
                ".wmv",
                ".ts",
                ".mpeg",
                ".mpg",
                ".3gp",
            )
        )

    for seg in chain:
        try:
            if hasattr(Comp, "Video") and isinstance(seg, getattr(Comp, "Video")):
                f = getattr(seg, "file", None)
                u = getattr(seg, "url", None)
                # 对于视频组件，优先使用 URL，其次才回退到 file/path
                if isinstance(u, str) and u:
                    videos.append(u)
                elif isinstance(f, str) and f:
                    videos.append(f)
            elif hasattr(Comp, "File") and isinstance(seg, getattr(Comp, "File")):
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
                    videos.extend(extract_videos_from_chain(content))
            elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
            elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
        except Exception:
            continue
    return videos


async def extract_videos_from_event(event: AstrMessageEvent) -> List[str]:
    """从当前事件中提取视频 URL/路径，优先从 Reply 引用中获取。"""
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
                vids = extract_videos_from_chain(payload)
                if vids:
                    return vids
        # 无内嵌内容时，尝试通过平台能力（OneBot/Napcat）用 message_id 拉取原消息
        reply_id = get_reply_message_id(reply_comp)
        platform_name = None
        try:
            platform_name = event.get_platform_name()
        except Exception:
            platform_name = None
        if reply_id and platform_name == "aiocqhttp" and hasattr(event, "bot"):
            try:
                vids: List[str] = []
                # Napcat/OneBot 分支：优先使用底层 api.call_action 以获得完整 payload
                if is_napcat(event) and hasattr(event.bot, "api"):
                    ret: Dict[str, Any] = await event.bot.api.call_action("get_msg", message_id=reply_id)
                    data = ob_data(ret)
                    # 1) 直接从原消息中提取视频
                    vids.extend(extract_videos_from_onebot_message_payload(data))
                    # 2) 检测其中是否包含 forward/nodes，并拉取合并转发节点中的视频
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
                                            from_forward = extract_videos_from_onebot_forward_payload(fwd)
                                            if from_forward:
                                                vids.extend(from_forward)
                                        except Exception as fe:
                                            logger.warning("zssm_explain: get_forward_msg for video failed: %s", fe)
                    except Exception:
                        pass
                else:
                    # 其他 OneBot 实现，沿用通用 get_msg 接口
                    data = await event.bot.get_msg(message_id=int(reply_id))
                    vids.extend(extract_videos_from_onebot_message_payload(data))

                if vids:
                    # 去重保持顺序
                    uniq: List[str] = []
                    seen = set()
                    for v in vids:
                        if isinstance(v, str) and v and v not in seen:
                            seen.add(v)
                            uniq.append(v)
                    if uniq:
                        return uniq
            except Exception:
                pass
    return extract_videos_from_chain(chain)


def is_http_url(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def is_abs_file(s: Optional[str]) -> bool:
    return isinstance(s, str) and os.path.isabs(s)


def is_napcat(event: AstrMessageEvent) -> bool:
    try:
        return event.get_platform_name() == "aiocqhttp" and hasattr(event, "bot") and hasattr(event.bot, "api")
    except Exception:
        return False


async def napcat_resolve_file_url(event: AstrMessageEvent, file_id: str) -> Optional[str]:
    """使用 Napcat 接口将文件/视频的 file_id 解析为可下载 URL。"""
    if not (isinstance(file_id, str) and file_id):
        return None
    if not is_napcat(event):
        return None
    # 优先根据上下文决定调用顺序：群聊先尝试群文件接口，再尝试私聊文件接口；
    # 私聊则只调用 get_private_file_url。
    try:
        gid = event.get_group_id()
    except Exception:
        gid = None

    actions: List[Dict[str, Any]] = []
    if gid:
        actions.append({"action": "get_group_file_url", "params": {"group_id": gid, "file_id": file_id}})
        actions.append({"action": "get_private_file_url", "params": {"file_id": file_id}})
    else:
        actions.append({"action": "get_private_file_url", "params": {"file_id": file_id}})

    for item in actions:
        action = item["action"]
        params = item["params"]
        try:
            ret = await event.bot.api.call_action(action, **params)
            data = ret.get("data") if isinstance(ret, dict) else None
            url = data.get("url") if isinstance(data, dict) else None
            if isinstance(url, str) and url:
                logger.info("zssm_explain: napcat %s ok, url=%s", action, url[:80])
                return url
            logger.warning("zssm_explain: napcat %s returned no url", action)
        except Exception as e:
            logger.warning("zssm_explain: napcat %s failed: %s", action, e)
    return None


def extract_videos_from_onebot_message_payload(payload: Any, prefer_file_id: bool = False) -> List[str]:
    """从 OneBot/Napcat get_msg/get_forward_msg 返回的 payload 中提取视频 URL/路径。

    - 默认行为：优先使用 url 字段，回退 file 字段（兼容通用 OneBot 实现）。
    - 当 prefer_file_id=True 且存在 file 字段时，优先返回 file（用于 Napcat，结合 get_*_file_url
      接口将 file_id 解析为下载 URL，避免直接使用可能不稳定的 url 字段）。
    """
    videos: List[str] = []
    data = ob_data(payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        candidates = data.get("message") or data.get("messages") or data.get("nodes") or data.get("nodeList")
        if isinstance(candidates, list):
            for seg in candidates:
                try:
                    if isinstance(seg, dict):
                        if "type" in seg and "data" in seg:
                            t = seg.get("type")
                            d = seg.get("data") or {}
                            if isinstance(d, dict):
                                if t == "video":
                                    # 对于 OneBot/Napcat 视频段，优先使用 url 字段，
                                    # file 字段通常为内部标识，不直接作为下载链接。
                                    url = d.get("url") or d.get("file")
                                    if isinstance(url, str) and url:
                                        videos.append(url)
                                elif t == "file":
                                    url = d.get("url") or d.get("file")
                                    name = d.get("name") or d.get("filename")

                                    def _looks_like_video(name_or_url: str) -> bool:
                                        if not isinstance(name_or_url, str) or not name_or_url:
                                            return False
                                        s = name_or_url.lower()
                                        return any(
                                            s.endswith(ext)
                                            for ext in (
                                                ".mp4",
                                                ".mov",
                                                ".m4v",
                                                ".avi",
                                                ".webm",
                                                ".mkv",
                                                ".flv",
                                                ".wmv",
                                                ".ts",
                                                ".mpeg",
                                                ".mpg",
                                                ".3gp",
                                            )
                                        )

                                    if isinstance(url, str) and url and _looks_like_video(url):
                                        videos.append(url)
                                    elif isinstance(name, str) and _looks_like_video(name) and isinstance(url, str) and url:
                                        videos.append(url)
                        else:
                            content = seg.get("content") or seg.get("message")
                            if isinstance(content, list):
                                inner = extract_videos_from_onebot_message_payload({"message": content}, prefer_file_id=prefer_file_id)
                                videos.extend(inner)
                except Exception:
                    continue
    return videos


def extract_videos_from_onebot_forward_payload(payload: Any) -> List[str]:
    """解析 OneBot get_forward_msg 返回的 messages/nodes/nodeList，汇总其中的视频 URL/路径。"""
    videos: List[str] = []
    data = ob_data(payload) if isinstance(payload, dict) else {}
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
                        inner = extract_videos_from_onebot_message_payload({"message": content})
                        if inner:
                            videos.extend(inner)
                except Exception:
                    continue
    return videos


def resolve_ffmpeg(config_path: str, default_path: str) -> Optional[str]:
    """解析 ffmpeg 可执行路径，优先使用配置路径，其次系统路径/ imageio-ffmpeg。"""
    path = config_path or default_path
    if path and shutil.which(path):
        return shutil.which(path)
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return None


def resolve_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    """解析 ffprobe 可执行路径：优先系统 ffprobe，其次与 ffmpeg 同目录。"""
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return sys_ffprobe
    if ffmpeg_path:
        cand = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if os.path.exists(cand):
            return cand
    return None


async def download_video_to_temp(url: str, size_mb_limit: int) -> Optional[str]:
    """下载视频到临时文件，做大小限制校验。"""

    def _safe_ext_from_url(u: str) -> str:
        try:
            path = urlparse(u).path
            base = os.path.basename(unquote(path))
            ext = os.path.splitext(base)[1]
            if isinstance(ext, str):
                ext = ext[:8]
            if not ext or not re.match(r"^\.[A-Za-z0-9]{1,6}$", ext):
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


def probe_duration_sec(ffprobe_path: Optional[str], video_path: str) -> Optional[float]:
    """使用 ffprobe（format/stream/帧率信息）探测视频时长。"""
    if not ffprobe_path:
        return None
    candidates: List[float] = []
    try:
        cmd1 = [ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
        res1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res1.returncode == 0:
            try:
                data1 = json.loads(res1.stdout.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                data1 = {}
            if isinstance(data1, dict):
                fmt = data1.get("format")
                if isinstance(fmt, dict):
                    d = fmt.get("duration")
                    try:
                        dur = float(d)
                        if dur and dur > 0:
                            candidates.append(dur)
                    except Exception:
                        pass

        cmd2 = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,nb_frames,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]
        res2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res2.returncode == 0:
            try:
                data2 = json.loads(res2.stdout.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                data2 = {}
            stream = None
            if isinstance(data2, dict):
                streams = data2.get("streams")
                if isinstance(streams, list) and streams:
                    s0 = streams[0]
                    if isinstance(s0, dict):
                        stream = s0
            if isinstance(stream, dict):
                d = stream.get("duration")
                try:
                    dur = float(d)
                    if dur and dur > 0:
                        candidates.append(dur)
                except Exception:
                    pass
                fps_txt = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
                try:
                    num, den = fps_txt.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 0.0
                except Exception:
                    fps = 0.0
                try:
                    nb_frames = stream.get("nb_frames")
                    nb = int(nb_frames) if nb_frames is not None and str(nb_frames).isdigit() else 0
                except Exception:
                    nb = 0
                if fps > 0 and nb > 0:
                    cand = nb / fps
                    if cand > 0:
                        candidates.append(cand)
    except Exception as e:
        logger.warning("zssm_explain: ffprobe duration failed: %s", e)
    if not candidates:
        return None
    c_sorted = sorted(set(candidates))
    logger.info("zssm_explain: ffprobe duration candidates=%s", [round(x, 3) for x in c_sorted])
    mid = len(c_sorted) // 2
    chosen = c_sorted[mid]
    logger.info("zssm_explain: ffprobe chosen duration=%.3f", chosen)
    return chosen
