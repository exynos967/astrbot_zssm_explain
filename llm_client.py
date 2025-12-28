from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, List, Optional


LLM_TIMEOUT_SEC_KEY = "llm_timeout_sec"
DEFAULT_LLM_TIMEOUT_SEC = 90


class LLMClient:
    """封装 LLM 调用与回退逻辑（Provider 选择 / 超时 / 输出清洗）。

    设计目标：
    - main.py 只负责“业务流程编排”，LLM 细节在此模块收敛；
    - 通过注入依赖（context / get_conf_int / get_config_provider）保持可替换性；
    - 尽量保持对 AstrBot Provider 接口的最小假设（仅依赖 .text_chat）。
    """

    def __init__(
        self,
        *,
        context: Any,
        get_conf_int: Callable[[str, int, int, int], int],
        get_config_provider: Optional[Callable[[str], Optional[Any]]] = None,
        logger: Optional[Any] = None,
    ):
        self._context = context
        self._get_conf_int = get_conf_int
        self._get_config_provider = get_config_provider
        self._logger = logger

    @staticmethod
    def filter_supported_images(images: List[str]) -> List[str]:
        """只保留看起来可被 LLM 读取的图片引用：

        - http(s) 链接
        - base64://... 或 data:image/...;base64,...
        - file://...（转换为本地路径）
        - 本地路径（绝对/相对，存在则通过）
        """
        ok: List[str] = []
        for x in images:
            try:
                if isinstance(x, str) and x:
                    lx = x.lower()
                    if lx.startswith(("http://", "https://")):
                        ok.append(x)
                    # OneBot 常见：base64://... 或 data:image/...;base64,...
                    elif lx.startswith("base64://") or lx.startswith("data:image/"):
                        ok.append(x)
                    # OneBot 常见：file://...
                    elif lx.startswith("file://"):
                        try:
                            fp = x[7:]
                            # Windows: file:///C:/xxx
                            if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                                fp = fp[1:]
                            if fp and os.path.exists(fp):
                                ok.append(os.path.abspath(fp))
                        except Exception:
                            pass
                    # 本地路径：绝对/相对都接受（存在则通过）
                    elif os.path.exists(x):
                        ok.append(os.path.abspath(x))
            except Exception:
                pass
        return ok

    @staticmethod
    def provider_supports_image(provider: Any) -> bool:
        """尽力判断 Provider 是否支持图片/多模态。"""
        try:
            mods = getattr(provider, "modalities", None)
            if isinstance(mods, (list, tuple)):
                ml = [str(m).lower() for m in mods]
                if any(
                    k in ml for k in ["image", "vision", "multimodal", "vl", "picture"]
                ):
                    return True
        except (AttributeError, TypeError):
            pass
        for attr in ("config", "model_config", "model"):
            try:
                val = getattr(provider, attr, None)
                text = str(val)
                lt = text.lower()
                if any(
                    k in lt
                    for k in [
                        "image",
                        "vision",
                        "multimodal",
                        "vl",
                        "gpt-4o",
                        "gemini",
                        "minicpm-v",
                    ]
                ):
                    return True
            except (AttributeError, TypeError, ValueError):
                pass
        return False

    def select_primary_provider(
        self,
        *,
        session_provider: Any,
        image_urls: List[str],
        text_provider_key: str = "text_provider_id",
        image_provider_key: str = "image_provider_id",
    ) -> Any:
        """根据是否包含图片选择首选 Provider。

        - 图片：优先配置 image_provider_id；否则首选会话 Provider（需具备图片能力）；
          否则从全部 Provider 中挑首个具备图片能力的；否则回退会话 Provider。
        - 文本：优先配置 text_provider_id；否则采用会话 Provider。
        """
        images_present = bool(image_urls)
        if images_present:
            cfg_img = self._get_provider_from_config(image_provider_key)
            return self.select_vision_provider(
                session_provider=session_provider, preferred_provider=cfg_img
            )

        cfg_txt = self._get_provider_from_config(text_provider_key)
        return cfg_txt if cfg_txt is not None else session_provider

    def select_vision_provider(
        self,
        *,
        session_provider: Any,
        preferred_provider: Optional[Any] = None,
        preferred_provider_key: Optional[str] = None,
    ) -> Any:
        """选择一个尽可能支持图片的 Provider，用于图片/视频等多模态场景。"""
        pp = preferred_provider
        if pp is None and preferred_provider_key:
            pp = self._get_provider_from_config(preferred_provider_key)
        if pp is not None:
            return pp
        if session_provider and self.provider_supports_image(session_provider):
            return session_provider
        try:
            providers = self._context.get_all_providers()
        except Exception:
            providers = []
        for p in providers:
            if p is session_provider:
                continue
            if self.provider_supports_image(p):
                return p
        return session_provider

    def _get_provider_from_config(self, key: str) -> Optional[Any]:
        if not self._get_config_provider:
            return None
        try:
            return self._get_config_provider(key)
        except Exception:
            return None

    async def call_with_fallback(
        self,
        *,
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
        timeout_sec = self._get_conf_int(
            LLM_TIMEOUT_SEC_KEY, DEFAULT_LLM_TIMEOUT_SEC, 5, 600
        )

        async def _try_call(p: Any) -> Any:
            return await asyncio.wait_for(
                p.text_chat(
                    prompt=user_prompt,
                    context=[],
                    system_prompt=system_prompt,
                    image_urls=image_urls,
                ),
                timeout=max(5, int(timeout_sec)),
            )

        if primary is not None:
            tried.add(id(primary))
            try:
                return await _try_call(primary)
            except Exception:
                pass

        if session_provider is not None and id(session_provider) not in tried:
            tried.add(id(session_provider))
            try:
                if not images_present or self.provider_supports_image(session_provider):
                    return await _try_call(session_provider)
            except Exception:
                pass

        try:
            providers = self._context.get_all_providers()
        except Exception:
            providers = []
        for p in providers:
            if id(p) in tried:
                continue
            if images_present and not self.provider_supports_image(p):
                continue
            tried.add(id(p))
            try:
                resp = await _try_call(p)
                if self._logger is not None:
                    self._logger.info(
                        "zssm_explain: fallback %s provider succeeded",
                        "vision" if images_present else "text",
                    )
                return resp
            except Exception:
                continue

        raise RuntimeError("all providers failed for current request")

    @staticmethod
    def pick_llm_text(llm_resp: object) -> str:
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

        return "（未解析到可读内容）"

    @staticmethod
    def sanitize_model_output(text: str) -> str:
        """去除常见思考/推理内容，仅保留结论性文本（插件侧兜底）。"""
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if not s:
            return s
        s = re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", s)
        s = re.sub(
            r"(?is)```\s*(think|thinking|reasoning|cot|chain[-_ ]?of[-_ ]?thought)[\s\S]*?```",
            "",
            s,
        )
        markers = [
            r"答案[:：]",
            r"结论[:：]",
            r"回答[:：]",
            r"总结[:：]",
            r"最终答案[:：]?",
            r"Final Answer[:：]?",
            r"Result[:：]?",
        ]
        for mk in markers:
            m = re.search(rf"(?is){mk}\s*(.+)$", s)
            if m:
                s = m.group(1).strip()
                break
        s = re.sub(
            r"(?im)^(思考|推理|分析|计划|步骤|原因|链式推理|思维|思路|推导|内心独白)[:：].*(\n\s*\n|$)",
            "",
            s,
        )
        s = re.sub(
            r"(?im)^(Reasoning|Thinking|Analysis|Plan|Steps|Rationale|Chain[-_ ]?of[-_ ]?Thought)[:：].*(\n\s*\n|$)",
            "",
            s,
        )
        s = re.sub(r"(?im)^【(思考|分析|推理|思维|计划|步骤)】.*(\n\s*\n|$)", "", s)
        s = re.sub(r"^[#>*\-\s]+", "", s).strip()
        return s or text.strip()
