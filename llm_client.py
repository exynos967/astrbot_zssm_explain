from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import re
from typing import Any, Callable, List, Optional


LLM_TIMEOUT_SEC_KEY = "llm_timeout_sec"
DEFAULT_LLM_TIMEOUT_SEC = 90


class LLMClient:
    """封装 LLM 调用与回退逻辑（Provider 选择 / 超时 / 输出解析）。

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

    @staticmethod
    def _provider_label(provider: Any) -> str:
        """尽量生成稳定可读的 Provider 标识，用于日志排查。"""
        if provider is None:
            return "None"
        for key in ("provider_id", "id", "name"):
            try:
                v = getattr(provider, key, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                continue
        try:
            return provider.__class__.__name__
        except Exception:
            return "unknown_provider"

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
        errors: List[str] = []
        fail_count = 0

        def _record(p: Any, e: Exception) -> None:
            nonlocal fail_count
            fail_count += 1
            if len(errors) >= 8:
                return
            try:
                label = self._provider_label(p)
            except Exception:
                label = "unknown_provider"
            try:
                msg = str(e).replace("\n", " ").strip()
            except Exception:
                msg = ""
            if len(msg) > 240:
                msg = msg[:240] + "..."
            errors.append(f"{label}: {e.__class__.__name__}: {msg}")

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
            except Exception as e:
                _record(primary, e)

        if session_provider is not None and id(session_provider) not in tried:
            tried.add(id(session_provider))
            try:
                if not images_present or self.provider_supports_image(session_provider):
                    return await _try_call(session_provider)
            except Exception as e:
                _record(session_provider, e)

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
            except Exception as e:
                _record(p, e)
                continue

        if self._logger is not None:
            self._logger.error(
                "zssm_explain: all providers failed (images_present=%s tried=%d fail=%d) errors=%s",
                images_present,
                len(tried),
                fail_count,
                errors,
            )
        sample_errors_str = ""
        if errors:
            # Only include a few sample errors in the exception message and truncate to avoid
            # excessively long error strings.
            max_samples = 3
            sample_errors = errors[:max_samples]
            sample_errors_str = "; ".join(sample_errors)
            if len(sample_errors_str) > 500:
                sample_errors_str = sample_errors_str[:500] + "..."
            if len(errors) > max_samples:
                sample_errors_str += f" (and {len(errors) - max_samples} more)"
        raise RuntimeError(
            "all providers failed for current request"
            + (f" (sample errors: {sample_errors_str})" if sample_errors_str else "")
        )

    async def call_video_direct(
        self,
        *,
        video_path: str,
        mode: str,
        provider: Any,
        prompt: str,
        fps: int = 2,
        timeout: int = 300,
    ) -> str:
        """视频直传调用。

        mode:
        - "dashscope": 使用 DashScope MultiModalConversation SDK
        - "base64": Base64 编码后通过 OpenAI 兼容接口发送
        """
        from .video_utils import get_video_size_mb

        loop = asyncio.get_running_loop()
        timeout = max(30, min(600, int(timeout)))
        fps = max(1, min(30, int(fps)))

        if mode == "dashscope":
            size_mb = get_video_size_mb(video_path)
            if size_mb <= 0:
                raise ValueError(f"invalid video file: {video_path}")
            if size_mb <= 100:
                video_url = f"file://{os.path.abspath(video_path)}"
            else:
                video_url = await asyncio.wait_for(
                    self._upload_to_dashscope_oss(
                        video_path, provider, timeout=timeout
                    ),
                    timeout=timeout,
                )

            api_key = self._extract_api_key(provider)
            model = self._extract_model(provider)

            # 从 provider 的 base_url 动态推导 DashScope SDK 的 base_http_api_url
            # 例如 https://coding.dashscope.aliyuncs.com/v1 → https://coding.dashscope.aliyuncs.com/api/v1
            dashscope_base_url = self._derive_dashscope_base_url(provider)

            def _call_dashscope():
                import dashscope  # noqa: F401 – SDK 初始化需要
                from dashscope import MultiModalConversation

                if dashscope_base_url:
                    dashscope.base_http_api_url = dashscope_base_url

                response = MultiModalConversation.call(
                    api_key=api_key,
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"video": video_url, "fps": fps},
                                {"text": prompt},
                            ],
                        }
                    ],
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"DashScope error: {response.code} {response.message}"
                    )
                try:
                    return response.output.choices[0].message.content[0]["text"]
                except (IndexError, KeyError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        f"DashScope response format unexpected: {exc}"
                    ) from exc

            return await asyncio.wait_for(
                loop.run_in_executor(None, _call_dashscope),
                timeout=timeout,
            )

        elif mode == "base64":
            size_mb = get_video_size_mb(video_path)
            if size_mb <= 0:
                raise ValueError(f"invalid video file: {video_path}")
            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
            mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
            data_url = f"data:{mime_type};base64,{video_b64}"

            # 绕过 AstrBot 的 image_urls 处理（resolve_image_part 会把
            # 非 http/base64:// 前缀当文件路径 open，导致 ENAMETOOLONG），
            # 直接构造 OpenAI 多模态 content 格式发送。
            api_key = self._extract_api_key(provider)
            model = self._extract_model(provider)

            import httpx
            from openai import AsyncOpenAI

            base_url = None
            try:
                cfg = getattr(provider, "provider_config", None)
                if isinstance(cfg, dict):
                    base_url = cfg.get("api_base") or cfg.get("base_url")
                if not base_url:
                    base_url = getattr(provider, "base_url", None)
                if not base_url:
                    client_obj = getattr(provider, "client", None)
                    if client_obj:
                        base_url = str(getattr(client_obj, "base_url", ""))
            except Exception:
                pass
            if not base_url:
                raise RuntimeError(
                    "cannot extract base_url from provider for base64 video"
                )

            proxy = None
            try:
                cfg = getattr(provider, "provider_config", None)
                if isinstance(cfg, dict):
                    proxy = cfg.get("proxy") or None
            except Exception:
                pass
            http_client = httpx.AsyncClient(proxy=proxy) if proxy else None

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": data_url},
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    ),
                    timeout=timeout,
                )
                text = completion.choices[0].message.content
                if isinstance(text, str) and text.strip():
                    return text.strip()
                return "（未解析到可读内容）"
            finally:
                await client.close()
                if http_client:
                    await http_client.aclose()

        raise ValueError(f"unsupported video_direct_mode: {mode}")

    @staticmethod
    def _derive_dashscope_base_url(provider: Any) -> Optional[str]:
        """从 provider 的 base_url 推导 DashScope SDK 的 base_http_api_url。

        provider 配置的是 OpenAI 兼容端点（如 https://coding.dashscope.aliyuncs.com/v1），
        DashScope SDK 需要的是 https://coding.dashscope.aliyuncs.com/api/v1。
        去掉末尾的 /v1 再拼 /api/v1。
        """
        raw_url = None
        try:
            cfg = getattr(provider, "provider_config", None)
            if isinstance(cfg, dict):
                raw_url = cfg.get("api_base") or cfg.get("base_url")
            if not raw_url:
                raw_url = getattr(provider, "base_url", None)
            if not raw_url:
                client_obj = getattr(provider, "client", None)
                if client_obj:
                    raw_url = str(getattr(client_obj, "base_url", ""))
        except Exception:
            pass
        if not isinstance(raw_url, str) or not raw_url.strip():
            return None
        url = raw_url.strip().rstrip("/")
        # 已经是 /api/v1 结尾，直接用
        if url.endswith("/api/v1"):
            return url
        # https://xxx.aliyuncs.com/compatible-mode/v1 → https://xxx.aliyuncs.com/api/v1
        if "/compatible-mode/v1" in url:
            return url.replace("/compatible-mode/v1", "/api/v1")
        # https://xxx.aliyuncs.com/v1 → https://xxx.aliyuncs.com/api/v1
        if url.endswith("/v1"):
            base = url[:-3]
            return f"{base}/api/v1"
        return None

    async def _upload_to_dashscope_oss(
        self, file_path: str, provider: Any, *, timeout: int = 300
    ) -> str:
        """上传文件到百炼临时 OSS，返回 oss:// URL（48h 有效）。"""
        loop = asyncio.get_running_loop()
        api_key = self._extract_api_key(provider)
        model = self._extract_model(provider)
        upload_timeout = max(30, min(600, int(timeout)))

        # 动态推导上传凭证 URL
        ds_base = self._derive_dashscope_base_url(provider)
        uploads_url = (
            f"{ds_base}/uploads"
            if ds_base
            else "https://dashscope.aliyuncs.com/api/v1/uploads"
        )

        def _upload():
            import time

            import requests

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            log = self._logger

            # 1. 获取上传凭证
            if log:
                log.info("zssm_explain: [oss] getting upload certificate...")
            t0 = time.monotonic()
            policy_resp = requests.get(
                uploads_url,
                headers={"Authorization": f"Bearer {api_key}"},
                params={"action": "getPolicy", "model": model},
                timeout=30,
            )
            policy_resp.raise_for_status()
            policy_data = policy_resp.json()["data"]
            t1 = time.monotonic()
            if log:
                log.info("zssm_explain: [oss] certificate ok in %.1fs", t1 - t0)

            # 2. 上传到 OSS
            file_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(file_path))
            key = f"{policy_data['upload_dir']}/{file_name}"

            if log:
                log.info(
                    "zssm_explain: [oss] uploading %.1fMB '%s'...",
                    file_size_mb,
                    file_name,
                )
            t2 = time.monotonic()
            with open(file_path, "rb") as f:
                files = {
                    "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
                    "Signature": (None, policy_data["signature"]),
                    "policy": (None, policy_data["policy"]),
                    "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
                    "x-oss-forbid-overwrite": (
                        None,
                        policy_data["x_oss_forbid_overwrite"],
                    ),
                    "key": (None, key),
                    "success_action_status": (None, "200"),
                    "file": (file_name, f),
                }
                upload_resp = requests.post(
                    policy_data["upload_host"],
                    files=files,
                    timeout=(30, upload_timeout),
                )
                upload_resp.raise_for_status()
            t3 = time.monotonic()
            speed = file_size_mb / (t3 - t2) if (t3 - t2) > 0 else 0
            if log:
                log.info(
                    "zssm_explain: [oss] done %.1fMB in %.1fs (%.1fMB/s)",
                    file_size_mb,
                    t3 - t2,
                    speed,
                )

            return f"oss://{key}"

        return await asyncio.wait_for(
            loop.run_in_executor(None, _upload), timeout=upload_timeout
        )

    @staticmethod
    def _extract_api_key(provider: Any) -> str:
        """从 Provider 对象提取 API Key。"""
        # 优先：AstrBot OpenAI provider 把 key 存在 chosen_api_key / api_keys
        for attr in ("chosen_api_key", "api_key", "key", "token"):
            try:
                v = getattr(provider, attr, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                # api_keys 是 list[str]
                if isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, str) and first.strip():
                        return first.strip()
            except Exception:
                continue
        try:
            cfg = getattr(provider, "provider_config", None)
            if isinstance(cfg, dict):
                for k in ("api_key", "key", "token"):
                    v = cfg.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                    # AstrBot 的 key 字段是 list
                    if isinstance(v, list) and v:
                        first = v[0]
                        if isinstance(first, str) and first.strip():
                            return first.strip()
        except Exception:
            pass
        raise RuntimeError("cannot extract api_key from provider")

    @staticmethod
    def _extract_model(provider: Any) -> str:
        """从 Provider 对象提取模型名称。"""
        # AstrBot 的 Provider 基类用 get_model() / model_name
        try:
            gm = getattr(provider, "get_model", None)
            if callable(gm):
                v = gm()
                if isinstance(v, str) and v.strip() and v.strip() != "unknown":
                    return v.strip()
        except Exception:
            pass
        for attr in ("model_name", "model", "model_id"):
            try:
                v = getattr(provider, attr, None)
                if isinstance(v, str) and v.strip() and v.strip() != "unknown":
                    return v.strip()
            except Exception:
                continue
        try:
            cfg = getattr(provider, "provider_config", None)
            if isinstance(cfg, dict):
                for k in ("model", "model_name", "model_id"):
                    v = cfg.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
        raise RuntimeError("cannot extract model from provider")

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
