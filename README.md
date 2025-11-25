# zssm_explain

- 触发方式：`/zssm` 指令；或忽略前缀的关键词 `zssm`；支持在同一条消息中“zssm + 内容”直接解释。
- 用法：
  - 直接解释：`zssm 这段内容是什么含义` 或 `/zssm 这是什么`。
  - 回复解释：先“回复”消息或图片，再触发 `zssm`，会解释被回复内容。
  - 网页摘要：`zssm https://example.com/...`；或“回复一条包含 URL 的消息后发送 `zssm`”，将自动抓取网页并输出简版摘要（2-8句）。
- 引用+@：同时包含“回复 + @某人”时也可正确触发。
- 合并转发支持：QQ 合并转发（forward/nodes）可自动拉取节点并合并解释。
- 图片支持：Napcat/OneBot 回复仅含 message_id 时自动回溯 `get_msg`；图片将优先使用配置的“图片转述模型”（Provider ID），失败回退到其它具备图片能力的 Provider。
- 文本支持：可在配置中指定“文本模型”（Provider ID）；失败回退到当前会话 Provider。
- 提示词：默认提示词位于 `prompt_utils.py` 常量，直接修改即可。
- 注意：未回复且未携带内容时，会提示“请输入要解释的内容。”
- Cloudflare：内置降级策略。若检测到 Cloudflare 防护，自动调用 `https://urlscan.io/liveshot/` 获取网页截图，解析 `<img>` 地址并将图片下载到临时文件，再交给多模态模型；若截图也失败，则提示用户手动提供截图/摘录。
- 截图降级会在后台等待截图生成（最多约 10 秒）；若最终仍未生成，会提示“截图降级失败”，此时可重试或手动上传截图。

支持视频解释

# TODO

解析b站链接

- 版本：v2.1.0

## Cloudflare 请求降级配置

| 配置键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `cf_screenshot_enable` | bool | `true` | 是否启用 Cloudflare 截图降级。当 HTML 抓取被屏蔽时自动走图片解释。 |
| `cf_screenshot_width` | int | `1280` | 截图宽度（px），传入 urlscan `width` 参数。 |
| `cf_screenshot_height` | int | `720` | 截图高度（px），传入 urlscan `height` 参数。 |

> 当 `cf_screenshot_enable=false` 或 urlscan 截图失败时，会提示用户“Cloudflare 防护导致无法抓取”，同时建议手动提供截图或开启降级。
