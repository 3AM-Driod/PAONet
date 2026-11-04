# tab5_8_structure —— 表 5.8 两段式结构消融

- 两段式：`weight/formal/`（前段 + 后段）；
- 单段式：`single_stage/after_net.pt`，扰动响应网络直接输出完整场，采用 `MODE=full` 训练。

当前读数：

| 方案 | RMSPE_mean / % |
|---|---|
| 两段式 | 0.2761 |
| 单段式 | 36.1746 |
