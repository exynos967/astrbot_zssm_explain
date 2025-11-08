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
- 提示词：默认提示词位于 `main.py` 顶部 `DEFAULT_*` 常量，直接修改即可。
- 注意：未回复且未携带内容时，会提示“请输入要解释的内容。”
- Cloudflare：若目标站点启用 Cloudflare 导致抓取失败，将在日志中标注 `Cloudflare protection detected`，并向用户发送专门提示“目标站点启用 Cloudflare 防护，暂无法抓取网页内容...”。

支持视频解释

# TODO

解析b站链接

- 版本：v1.2.0
