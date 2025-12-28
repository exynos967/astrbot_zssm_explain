from __future__ import annotations

from typing import List, Optional, Any, Dict, Tuple

import asyncio
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

from .message_utils import ob_data


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
                ".gif",
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
                elif (
                    isinstance(f, str)
                    and f
                    and (_looks_like_video(f) or os.path.isabs(f))
                ):
                    cand = f
                elif (
                    isinstance(n, str)
                    and n
                    and _looks_like_video(n)
                    and isinstance(f, str)
                    and f
                ):
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


def is_http_url(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def is_abs_file(s: Optional[str]) -> bool:
    return isinstance(s, str) and os.path.isabs(s)


def is_napcat(event: AstrMessageEvent) -> bool:
    try:
        # AstrBot 的 OneBot/Napcat 适配器在不同环境下 platform_name 可能不同（如 bridge/onebot_v11 等），
        # 这里以是否具备 OneBot 风格的 call_action 能力作为主要判断依据。
        if not (hasattr(event, "bot") and hasattr(event.bot, "api")):
            return False
        api = getattr(event.bot, "api", None)
        if api is None or not hasattr(api, "call_action"):
            return False
        return True
    except Exception:
        return False


async def napcat_resolve_file_url(
    event: AstrMessageEvent, file_id: str
) -> Optional[str]:
    """使用 Napcat 接口将文件/视频的 file_id 解析为可下载 URL 或本地路径。

    说明：
    - 群文件：通常可用 get_group_file_url / get_private_file_url
    - 媒体（如视频/图片/语音）在部分场景下需要使用 get_file/get_image/get_record 等接口
      才能解析出 url/file，本函数目前以 get_file 作为兜底（兼容合并转发里的 video fileUUID）。
    """
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

    group_id_param: Any = gid
    try:
        if isinstance(gid, str) and gid.isdigit():
            group_id_param = int(gid)
        elif isinstance(gid, int):
            group_id_param = gid
    except Exception:
        group_id_param = gid

    def _stem_if_needed(s: str) -> Optional[str]:
        try:
            base, ext = os.path.splitext(s)
            if ext and ext.lower() in (
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
                ".gif",
            ):
                if base and base != s:
                    return base
        except Exception:
            pass
        return None

    # 一些 Napcat 场景下 file 值会携带扩展名（形如 hash.mp4），但接口实际需要 hash 本体。
    candidates: List[str] = [file_id]
    stem = _stem_if_needed(file_id)
    if isinstance(stem, str) and stem and stem not in candidates:
        candidates.append(stem)

    actions: List[Dict[str, Any]] = []
    # 兜底：Napcat 通用文件解析接口（可用于 message 视频/图片等的 file_id/fileUUID）
    for fid in candidates:
        actions.append({"action": "get_file", "params": {"file_id": fid}})
        actions.append({"action": "get_file", "params": {"file": fid}})
        # 图片在部分实现中需要 get_image 才能拿到本地路径或可下载 URL（优先尝试，失败则忽略）
        actions.append({"action": "get_image", "params": {"file": fid}})
        actions.append({"action": "get_image", "params": {"file_id": fid}})
        actions.append({"action": "get_image", "params": {"id": fid}})
        actions.append({"action": "get_image", "params": {"image": fid}})

    # 群文件接口：仅在能拿到群号时尝试
    if group_id_param:
        for fid in candidates:
            actions.append(
                {
                    "action": "get_group_file_url",
                    "params": {"group_id": group_id_param, "file_id": fid},
                }
            )
    for fid in candidates:
        actions.append({"action": "get_private_file_url", "params": {"file_id": fid}})

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
            # get_file/get_image/get_record 等可能返回本地路径 file
            f = data.get("file") if isinstance(data, dict) else None
            if isinstance(f, str) and f:
                lf = f.lower()
                # OneBot 常见：base64://... 或 data:image/...;base64,...
                if lf.startswith("base64://") or lf.startswith("data:image/"):
                    logger.info(
                        "zssm_explain: napcat %s ok, base64(%d)", action, len(f)
                    )
                    return f
                # OneBot 常见：file://...
                if lf.startswith("file://"):
                    try:
                        fp = f[7:]
                        # Windows: file:///C:/xxx
                        if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                            fp = fp[1:]
                        if fp and os.path.exists(fp):
                            fp = os.path.abspath(fp)
                            logger.info(
                                "zssm_explain: napcat %s ok, file=%s", action, fp[:80]
                            )
                            return fp
                    except Exception:
                        pass
                # 绝对路径或相对路径（存在则提升为绝对路径）
                try:
                    if os.path.isabs(f) and os.path.exists(f):
                        logger.info(
                            "zssm_explain: napcat %s ok, file=%s", action, f[:80]
                        )
                        return f
                    if os.path.exists(f):
                        fp = os.path.abspath(f)
                        logger.info(
                            "zssm_explain: napcat %s ok, file=%s", action, fp[:80]
                        )
                        return fp
                except Exception:
                    pass
            logger.debug(
                "zssm_explain: napcat %s returned no url/file (file_id=%s params=%s)",
                action,
                str(file_id)[:64],
                {
                    k: str(v)[:64]
                    for k, v in (params.items() if isinstance(params, dict) else [])
                },
            )
        except Exception as e:
            logger.debug(
                "zssm_explain: napcat %s failed (file_id=%s params=%s): %s",
                action,
                str(file_id)[:64],
                {
                    k: str(v)[:64]
                    for k, v in (params.items() if isinstance(params, dict) else [])
                },
                e,
            )
            continue
    logger.warning(
        "zssm_explain: napcat resolve video/file failed (file_id=%s)", str(file_id)[:64]
    )
    return None


def extract_videos_from_onebot_message_payload(
    payload: Any, prefer_file_id: bool = False
) -> List[str]:
    """从 OneBot/Napcat get_msg/get_forward_msg 返回的 payload 中提取视频 URL/路径。

    - 默认行为：优先使用 url 字段，回退 file 字段（兼容通用 OneBot 实现）。
    - 当 prefer_file_id=True 且存在 file 字段时，优先返回 file（用于 Napcat，结合 get_*_file_url
      接口将 file_id 解析为下载 URL，避免直接使用可能不稳定的 url 字段）。
    """
    videos: List[str] = []
    data = ob_data(payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        candidates = (
            data.get("message")
            or data.get("messages")
            or data.get("nodes")
            or data.get("nodeList")
        )
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
                                        if (
                                            not isinstance(name_or_url, str)
                                            or not name_or_url
                                        ):
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
                                                ".gif",
                                            )
                                        )

                                    if (
                                        isinstance(url, str)
                                        and url
                                        and _looks_like_video(url)
                                    ):
                                        videos.append(url)
                                    elif (
                                        isinstance(name, str)
                                        and _looks_like_video(name)
                                        and isinstance(url, str)
                                        and url
                                    ):
                                        videos.append(url)
                        else:
                            content = seg.get("content") or seg.get("message")
                            if isinstance(content, list):
                                inner = extract_videos_from_onebot_message_payload(
                                    {"message": content}, prefer_file_id=prefer_file_id
                                )
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
                        inner = extract_videos_from_onebot_message_payload(
                            {"message": content}
                        )
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


async def sample_frames_with_ffmpeg(
    ffmpeg_path: str,
    video_path: str,
    interval_sec: int,
    count_limit: int,
) -> List[str]:
    """按 fps=1/interval 抽帧，返回帧图片路径列表（均位于同一临时目录）。

    注意：调用方负责删除返回路径所在目录。
    """
    out_dir = tempfile.mkdtemp(prefix="zssm_frames_")
    out_tpl = os.path.join(out_dir, "frame_%03d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{max(1, interval_sec)}",
        "-frames:v",
        str(max(1, count_limit)),
        "-qscale:v",
        "2",
        out_tpl,
    ]
    loop = asyncio.get_running_loop()

    def _run():
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )

    res = await loop.run_in_executor(None, _run)
    if res.returncode != 0:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        logger.error(
            "zssm_explain: ffmpeg fps-sampler failed (code=%s)", res.returncode
        )
        raise RuntimeError("ffmpeg sample frames failed")

    frames: List[str] = []
    try:
        for name in sorted(os.listdir(out_dir)):
            if name.lower().endswith(".jpg"):
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


async def sample_frames_equidistant(
    ffmpeg_path: str,
    video_path: str,
    duration_sec: float,
    count_limit: int,
) -> List[str]:
    """按等距时间点抽帧，覆盖全片。选择 N 个时间点：t_i = (i/(N+1))*duration。

    注意：调用方负责删除返回路径所在目录。
    """
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
                ffmpeg_path,
                "-y",
                "-ss",
                f"{max(0.0, t):.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-qscale:v",
                "2",
                out_path,
            ]

            def _run_one():
                return subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )

            res = await loop.run_in_executor(None, _run_one)
            if res.returncode == 0 and os.path.exists(out_path):
                frames.append(out_path)
            else:
                logger.warning(
                    "zssm_explain: ffmpeg sample at %.3fs failed (code=%s)",
                    t,
                    res.returncode,
                )
    except Exception as e:
        logger.error("zssm_explain: equidistant sampler error: %s", e)
    if not frames:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError("no frames generated by equidistant sampler")
    return frames


async def extract_audio_wav(ffmpeg_path: str, video_path: str) -> Optional[str]:
    """从视频抽取单声道 16kHz wav，返回临时文件路径（由调用方负责删除）。"""
    out_fd, out_path = tempfile.mkstemp(prefix="zssm_audio_", suffix=".wav")
    os.close(out_fd)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_path,
    ]
    loop = asyncio.get_running_loop()

    def _run():
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )

    res = await loop.run_in_executor(None, _run)
    if res.returncode != 0:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None
    return out_path if os.path.exists(out_path) else None


async def download_video_to_temp(
    url: str, size_mb_limit: int, headers: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """下载视频到临时文件，做大小限制校验。

    headers 可选，用于为特定站点（如 B 站）附加 UA/Referer 等。
    """

    def _safe_ext_from_url(u: str) -> str:
        try:
            path = urlparse(u).path
            base = os.path.basename(unquote(path))
            ext = os.path.splitext(base)[1]
            if isinstance(ext, str):
                ext = ext[:8]
            if not ext or not re.match(r"^\.[A-Za-z0-9]{1,6}$", ext):
                lower = base.lower()
                for cand in (
                    ".mp4",
                    ".mov",
                    ".m4v",
                    ".avi",
                    ".webm",
                    ".mkv",
                    ".flv",
                    ".wmv",
                ):
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
                async with sess.get(url, timeout=20, headers=headers or {}) as resp:
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

        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=20) as r, open(tmp_path, "wb") as f:
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
        cmd1 = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]
        res1 = subprocess.run(
            cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
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
        res2 = subprocess.run(
            cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
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
                fps_txt = (
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
                )
                try:
                    num, den = fps_txt.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 0.0
                except Exception:
                    fps = 0.0
                try:
                    nb_frames = stream.get("nb_frames")
                    nb = (
                        int(nb_frames)
                        if nb_frames is not None and str(nb_frames).isdigit()
                        else 0
                    )
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
    logger.info(
        "zssm_explain: ffprobe duration candidates=%s", [round(x, 3) for x in c_sorted]
    )
    mid = len(c_sorted) // 2
    chosen = c_sorted[mid]
    logger.info("zssm_explain: ffprobe chosen duration=%.3f", chosen)
    return chosen


async def extract_forward_video_keyframes(
    event: AstrMessageEvent,
    video_sources: List[str],
    *,
    enabled: bool,
    max_count: int,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_mb: int,
    max_sec: int,
    timeout_sec: int,
) -> Tuple[List[str], List[str]]:
    """将合并转发中的视频源转换为少量关键帧图片（默认每个视频 1 张），用于“聊天记录解释”场景。

    返回:
    - frames: 本地关键帧图片路径列表（均位于临时目录）
    - cleanup_paths: 需要清理的临时路径（文件或目录）
    """
    if not enabled:
        return ([], [])
    if not video_sources:
        return ([], [])
    if not ffmpeg_path:
        return ([], [])
    try:
        max_count = int(max_count)
    except Exception:
        max_count = 0
    if max_count <= 0:
        return ([], [])

    uniq_sources: List[str] = []
    seen = set()
    for s in video_sources:
        if isinstance(s, str) and s and s not in seen:
            seen.add(s)
            uniq_sources.append(s)

    frames: List[str] = []
    cleanup: List[str] = []

    for src in uniq_sources[:max_count]:
        local_path = None
        downloaded_tmp = False
        try:
            resolved_src = src
            if (
                isinstance(resolved_src, str)
                and (not is_http_url(resolved_src))
                and (not is_abs_file(resolved_src))
            ):
                try:
                    resolved = await napcat_resolve_file_url(event, resolved_src)
                except Exception:
                    resolved = None
                if isinstance(resolved, str) and resolved:
                    resolved_src = resolved

            if isinstance(resolved_src, str) and is_http_url(resolved_src):
                try:
                    local_path = await asyncio.wait_for(
                        download_video_to_temp(resolved_src, max_mb),
                        timeout=max(2, int(timeout_sec)),
                    )
                except Exception as e:
                    logger.warning(
                        "zssm_explain: forward video download timeout/failed: %s", e
                    )
                if local_path:
                    downloaded_tmp = True
            elif (
                isinstance(resolved_src, str)
                and is_abs_file(resolved_src)
                and os.path.exists(resolved_src)
            ):
                local_path = resolved_src

            if not local_path:
                continue

            dur = probe_duration_sec(ffprobe_path, local_path) if ffprobe_path else None
            if isinstance(dur, (int, float)) and dur > max_sec:
                continue

            try:
                if isinstance(dur, (int, float)) and dur > 0:
                    sampled = await sample_frames_equidistant(
                        ffmpeg_path, local_path, float(dur), 1
                    )
                else:
                    sampled = await sample_frames_with_ffmpeg(
                        ffmpeg_path, local_path, max(1, max_sec), 1
                    )
            except Exception:
                sampled = []

            if sampled:
                frames.append(sampled[0])
                try:
                    cleanup.append(os.path.dirname(sampled[0]))
                except Exception:
                    pass
        finally:
            if downloaded_tmp and isinstance(local_path, str) and local_path:
                cleanup.append(local_path)
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

    uniq_cleanup: List[str] = []
    seen2 = set()
    for p in cleanup:
        if isinstance(p, str) and p and p not in seen2:
            seen2.add(p)
            uniq_cleanup.append(p)

    return (frames, uniq_cleanup)
