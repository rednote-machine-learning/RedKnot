# 从 SGLang 到 vLLM 的迁移设计

## 保持的算法语义

RedKnot 固定版本中的 `python/sglang/srt/layers/attention/redknot/` 是主要参考。
`driver_batched.py` / `offline_cache.py` 的离线 artifact 按独立 chunk 生成，
`rope_helper.py` 按逻辑 token 位置移动缓存的 K。这里不把物理 KV slot 当作
token，也不把第二块开始的 token 丢弃来伪造前缀命中。

| SGLang 责任 | 本项目 | vLLM 接入点 |
| --- | --- | --- |
| 离线 chunk/head artifact | `runtime.py`, `cache.py` | 原生完整 prefill 结束时提交所有选中层 |
| 按头分类、local/global attention | `vllm_backend.py` | CUSTOM FlashAttentionImpl.forward |
| RoPE 位置搬移 | `ops.py` | 从源位置到目标逻辑位置的 R(dst)R(src)^-1 |
| 请求/层信息传递 | `runner.py` | V1 GPUModelRunner._model_forward / ForwardContext |
| 持久 KV 一致性 | `vllm_backend.py` | 原生 cache update 后只覆盖 clean local K/V |
| DSV4 分头输出投影 | `dsv4_projection.py`, `dsv4_backend.py` | native FlashMLA prefill + 原生 inverse-RoPE/FP8 wo_a + 一次 wo_b |

MHA/GQA 的不同头是不同输出通道；必须按 head ID scatter 回原顺序，再由原模型
执行 output projection。不能跨 local/global **不同头**用 softmax LSE 混合。
同一头的不同 KV 分片才可做 LSE merge；本实现用完整因果 FA，不需要此分片合并。

## 一次完整请求

1. `capture`：单个 chunk 从位置 0 原生 prefill；每个选中层保存 local K/V 与
   local attention output 的独立 CPU 副本。所有层成功后一次性提交。
2. `reuse`：根据真实 token、namespace、checkpoint、实际模型配置、dtype、
   TP 与 head/RoPE 策略定位全部 artifact，并同时持有租约。
3. preflight：验证全部选中层、完整请求、静态 RoPE、KV layout、逻辑 block
   table 与真实 slot mapping。任何 miss/不兼容都在写缓存前回退整个执行步。
4. global heads 计算全部行。local heads 的 clean 行直接恢复输出；chunk 移位
   边界和所有新/query 行按完整可见 mixed local KV 重算。
5. 恢复 local K/V 到真实物理页的对应 head；其他 head、dirty 行和未用槽不动。
   后续原生 decode 读取这些混合历史 KV。异常始终释放租约及请求上下文。

原生 QKV projection、QK norm、RoPE、output projection、residual、FFN 均保留。
预填充 reuse 只跳过部分 attention query rows，不声称跳过整个模型层。

## 为什么独立 chunk 是近似

独立离线运行 chunk 时，后层 hidden states 没见过前面其他 chunk。
即使 RoPE 搬移是正确的，缓存 local K/V 和 local attention output 也不保证
等于全上下文 dense 的值。边界重算只能减轻影响，不能证明误差被完全消除。
默认显式关闭近似；全局配置与单请求同时允许才启用。头策略、dense 前层和
边界宽度需要真实数据校准；精度门槛不能用 CPU 单元测试替代。

## DeepSeek V4 Flash 专用接入

SGLang `dsv4_mla_offload.py` / `models/deepseek_v4.py` 的核心是
`z_off = wo_a(local-head attention after inverse RoPE)`，按 group 保存
`[T,G,R]` 的低秩贡献。`mla.py` 已实现并测试：

- 按真实 head 列切分 grouped wo_a，而不是套用 per-KV-head 的 MHA 存储。
- clean 行用 `z_off + z_global`，dirty/query 行替换为
  `z_local_online + z_global`；完整和只调用一次 wo_b。
- 不推断量化尺度、inverse RoPE 或 TP 集合通信，调用方必须提供正确契约。

vLLM DSV4 位于 `vllm/models/deepseek_v4/`，通过专用 `forward_mqa`
执行，不经过普通 CUSTOM AttentionImpl。新增 `deepseek_v4_flash` family
在 pinned native FlashMLA 完成状态更新和候选 gather 后，仅拦截最后 sparse
prefill attention 调用。原生 SWA packed latent、C4/C128 compressed KV、
indexer key/scales、compressor checkpoint/terminal state **全部在线生产**，
不把这些有时序依赖的状态当作 local head KV 去复用。

`dsv4_sparse.py` 按原始逻辑 query-head ID 和行选择计算，共享同一 latent KV，
保留 native candidate visibility、重复候选、sink softmax 和 invalid mask。
原生 FlashMLA 的 64/128-head 约束不能靠补齐再声明省算，因此选头内核采用
16-head tile 并报告真实 padding 工作量。dirty/query local 行保留在线输出，
clean local 行清零；全部 global 行在线计算。

`dsv4_projection.py` 用原生 inverse-RoPE/FP8 投影 callback 保存 z_off。
reuse 只对 clean 行相加，所有 chunk 共用一次在线投影和一次 wo_b。
128 维量化块不跨 512 维 head，因此保留头的输入量化尺度不因 masking 改变；
但 full-width wo_a 没有减算，BF16 部分 GEMM 与相加也不是逐位等价。
独立 chunk 的上下文近似和真实输出质量必须另外评测。

本接入目前限定 TP1、64 heads、text-only Flash-0731、V1/eager/full-prefill。
不支持 Pro、多请求/TP8；不能删除这些 guard 直接当作支持。

## 测试设计与发布要求

最便宜且能捕获错误的测试先行：

- Cache I/O：key → payload 租约；超预算、重复 key、并发 pin 和异常释放。
- Attention I/O：Q/K/V + logical spans → head output + persistent pages；
  检查 dirty 行因果、clean query 实际不参与 FA、global 页未被覆盖。
- Decode I/O：回填 pages + 新 query → 下一 token attention；检查读取混合历史。
- MLA I/O：分头 contribution → full projection；对照 dense 代数 oracle，
  检查 dirty 替换与 wo_b 只调用一次。
- Benchmark I/O：paired outputs/metrics/counters → 可审计统计；无 hit、无
  latency、没有参考答案均不能给出合格的性能/精度结论。

CPU tests 的 FA 调用以确定性 oracle 替代，不验证真实 CUDA kernel。
GPU 发布门槛：目标模型真实 checkpoint，H200/B300 分别完成原生基线、
热态 paired 长输出、多数据集质量和显存/CPU内存测量。当前尚未执行。
单请求版本不能报告吞吐或并行加速；并发需要将请求计划、租约和 batch 索引
提升到多请求执行，并补充 decode/preemption/eviction 并发测试。
