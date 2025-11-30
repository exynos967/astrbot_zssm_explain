from __future__ import annotations

from typing import List, Optional


# === 默认提示词常量（集中管理，可供用户修改） ===

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文助理，擅长从被引用的消息中提炼含义、意图和注意事项。"
    "请严格使用如下 Markdown 结构输出回答：\n"
    "1. 第一行以“关键词：”开头，列出 3~8 个核心关键词，使用“ | ”分隔，例如：\n"
    "   关键词：chcp 936 | 代码页 | GBK | 批处理 | 乱码\n"
    "2. 接着给出 1~2 句总括性说明，可以单独成段。\n"
    "3. 然后输出一行“**详细阐述：**”，在其后用若干段落进行详细解释。\n"
    "禁止输出思考过程或中间推理，只保留对用户有用的结论性内容。"
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
    "请解释这段视频的主要内容，输出简洁不超过100字。仅依据提供的关键帧与音频转写（如有）作答；"
    "若信息不足请明确说明‘无法判断’，不要编造未出现的内容。\n"
    "{meta_block}\n{asr_block}"
)

# 单帧描述提示词（逐帧多次调用使用）
DEFAULT_FRAME_CAPTION_PROMPT = (
    "请根据这张关键帧图片，用一句中文描述画面要点；少于25字。若无法判断，请回答‘未识别’。"
)


def build_user_prompt(text: Optional[str], images: List[str]) -> str:
    """根据是否包含图片选择文本/图文提示词模板。"""
    text_block = ("原始文本:\n" + text) if text else ""
    tmpl = DEFAULT_IMAGE_USER_PROMPT if images else DEFAULT_TEXT_USER_PROMPT
    return tmpl.format(text=text or "", text_block=text_block)


def build_system_prompt() -> str:
    """返回系统提示词（供 LLM 调用使用）。"""
    return DEFAULT_SYSTEM_PROMPT
