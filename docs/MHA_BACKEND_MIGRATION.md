# Mistral / Llama / Qwen3：算法迁移与原生接入分开

`vllm_redknot/model_backends/mha_reuse.py` 是显式 callback 算法资产，
`RUNTIME_INTEGRATED = False`。它不会注册插件、修改现有 runner、加载模型、
接受当前不支持的模型配置，或改变 Flash 路径。来源 revision、源文件完整 SHA、
目标 SHA 和非机械改写范围见 [mha_migration_provenance.json](mha_migration_provenance.json)。

## Mistral 的原生 SWA 离线复用

源 `driver_batched.py::run_redknot_swa_offlinekv` 的专有语义现以三个接口表达：

- `plan_swa_reuse`：只用标准库，按文档长度计算绝对 offset 和边界长度。
  首文档始终不重算；后续文档默认 `max(1, int(length * 0.20))` 个 token。
  可显式用 `recompute_ratio=None` 切换到 `min(recompute_prefix, length)`。
- `replay_swa_documents`：将离线 K 从文档局部位置搬至拼接位置，V 不旋转；
  按文档顺序调用原生 boundary forward，用新前缀 K/V 替换对应离线前缀，
  后缀保留已搬移的离线 K/V。后一个文档能看到之前文档已修正的边界。
- `SWAReuseResult`：返回完整逻辑 KV、各文档 KV、query 绝对起点及原生 window。
  **结果不是 vLLM 的物理页面，也没有安装 query/decode 滑窗缓存。**

纯计划示例，不导入 Torch、Transformers、vLLM、SGLang 或初始化 GPU：

```python
from vllm_redknot.model_backends.mha_reuse import plan_swa_reuse

plan = plan_swa_reuse([7500, 7500, 7500, 7500], sliding_window=4096)
assert plan.offsets == (0, 7500, 15000, 22500)
assert plan.boundary_lengths == (0, 1500, 1500, 1500)
assert plan.query_position == 30000
```

`sliding_window` 必须来自实际 checkpoint，不能强行给 full-attention 模型套上
4096。默认 20% 是源 Mistral benchmark 的实验策略，不是任意上下文/层数下
精确等价的证明；边界可能短于实际有效感受野，真实质量仍需单独验证。

### callback 契约

调用者显式提供每个 chunk 的 token IDs 及每层 `[1, heads, tokens, dim]` 的
浮点 K/V。源内容身份、checkpoint、精度、层映射与离线 artifact 的一致性由
调用者验证；仅通过长度检查不能证明 artifact 属于这些 token。

1. `reposition_key(key, *, layer_index, document_index, src_start, dst_start)`
   必须使用对应模型真正的 RoPE 操作，返回同 shape/dtype/device 的 K。
   本模块不猜 static/llama3/scaled RoPE 公式。
2. `forward_boundary(BoundaryReplay)` 接收精确前缀 token IDs、起始绝对位置、
   前序文档已组装的 K/V 和原生 SWA window。必须执行原生 SWA，返回每层
   **仅该边界 token** 的 post-RoPE K/V，不是整份 past cache。
3. 原生 adapter 另外负责 query/decode、slot mapping、sliding-cache 留存、
   position bookkeeping 和并发请求隔离。本模块不复制这些 engine 实现。

callbacks 只接收 owned tensor 副本；异常不发布部分结果，也不修改输入
artifact。参考实现的 clone/concat 是可审计正确性边界，不是零拷贝优化。
完整 KV 与工作副本可能增加存储，不能宣称它已经带来显存或 TTFT 改善。
模型生成循环、停止规则、计时和权重加载均未复制。

## Llama / Qwen3 策略复用，不再复制算法

`build_head_class_policy(source_json, ...)` 延迟调用已有
`core.head_config.HeadClassConfig`，保留：

- `kv_head_classification` 的 `local_full` 等源 legacy alias；
- `kv_head_max_distance`、可选 `kv_head_sink_size`、dense-prefix 与 retrieval；
- 源 benchmark 的 retrieval-to-global 归并；可显式关闭；
- 固定窗口优先，否则 `max(window_min, int(total_context_tokens * window_ratio))`。

原矩阵会先复制，窗口调整和归并不修改调用者 JSON。沿用源
`HeadClassConfig.from_json` 的默认值；不能根据历史 rationale/summary 自行
改写矩阵。尤其 `local_full` 是源解析器里的 alias，不自动代表全局头。

`build_sparse_ffn_policy` 复用 `core.sparse_ffn.SparseFFNSchedule` 的浅/中/深
层阈值与 recent/min-keep 契约。Llama JSON 的 `local_window` 属于头窗口配置，
不传给 FFN schedule；其值应由调用者显式传为 `fixed_window`。
选择算法和 `apply_sparse_ffn` 仍在 core；这里没有 MLP compact/scatter hook。

## 仍未接入的原生能力

| 模型/能力 | 本次迁移状态 | 当前 native vLLM 运行限制 |
| --- | --- | --- |
| Mistral native SWA reuse | offset/replay/splice callback 算法与契约已迁移 | Mistral 架构和 SWA 仍被现有 runner 拒绝；未安装真实 KV 页面/滑窗 decode |
| Llama3.3 头分类与 FFN 策略 | 借用已提取 core，提供明确的组合接口 | 源 Llama3.3 使用 llama3 RoPE scaling；当前 runner 只支持 base static RoPE，不等于该模型已可运行 |
| Qwen3 头分类与 FFN 策略 | 保留 source alias/window/sink 与分层 FFN schedule | 现有 active head-only importer 仍拒绝 legacy alias；源默认 NF4 不在现有 unquantized runner 范围 |
| SegPaged/FA3 与按头 query/decode | 仅已有部分 core 策略/存储资产 | 本次未迁移原完整 attention driver、并发执行或 KV allocator |

不会为了“入口能通过”删除这些 guard。代码迁移、实际接入、模型质量与性能
验收必须分别记录；本次没有新增任何模型运行或速度/精度结论。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests \
  -p test_mha_reuse_migration.py -v
```

测试覆盖 20%/固定边界计划、首文档保留、import 隔离、来源与目标 SHA，及
可选 CPU tensor callback oracle：搬移、前缀替换、后缀复用、顺序可见性、
输入隔离、非法回调拒绝与 core 策略复用。没有 Torch 时会明确跳过 tensor
测试，不将跳过计为通过；没有运行 GPU。若要复查本地源文件 SHA，设置
`REDKNOT_SOURCE_ROOT` 为固定 revision 的源仓库目录。

2026-09-14 本地 Python 3.12 检查：16 项中 9 项通过（含四个源文件 SHA），
7 项因未安装 Torch 明确跳过；Ruff check/format 检查通过。未执行 tensor
oracle、GPU 或模型实验，不能由计划测试推断这些路径已验证。
