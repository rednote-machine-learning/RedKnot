# RedKnot 提取与 vLLM 接入边界

本说明区分三个状态：**源算法已识别/提取**、**已接入 vLLM 执行链**、
**真实模型已验证**。前一个状态不自动推出后一个状态。文件改名、复制进新目录、
CPU 契约测试通过，都不能证明原 SGLang 全部能力已经适配。

审计参考：RedKnot 固定提交 `55ee4e8401603f8d2612877e4053e18b37b1c1bd`。
下文源路径均相对于该仓库的 `python/sglang/srt/`，源行号仅用于定位这一版本。
core 提取和复现入口的具体范围，以各自清单及测试结果为准；
它们的存在不代表下列 native 依赖已解决。

2026-09-14 多模型增量（0.2.2）已将四个命名 benchmark 的数据/策略协议，以及
Mistral SWA、Llama/Qwen3 头/FFN 策略组合、Qwen3.5 full/linear/conv/GDN/MoE
专有算法迁至 `benchmarks/` 与 `vllm_redknot/model_backends/`。
下文提到的 SGLang globals/pools 是**源代码的耦合点**；新模块通过显式
配置、tensor buffer 和 callback 替代它们，不导入这些原生类型，也没有默认
安装相应 vLLM hooks。对应来源与修改范围见
[MHA/SWA](MHA_BACKEND_MIGRATION.md)、[Qwen3.5](MULTIMODEL_BACKEND.md) 和
[benchmark](MULTIMODEL_BENCHMARKS.md)。用户要求本次只迁移代码，不跑模型实验。

## 应保留的结构边界

- `vllm_redknot/`：本项目实际启用的插件、vLLM 适配层、运行时和缓存管理。
- 独立算法/参考模块：保留 RedKnot 的策略、类型、算法和来源信息；显式标记
  未接入能力。不能在默认插件注册时导入未经适配的 SGLang adapter。
- benchmark 与策略数据：可迁移实验协议、用例和指标口径，但不能复用 SGLang
  启动命令后声称它运行的是 vLLM。源码提取清单与性能报告必须分开。
- 不复制原生 `sglang` 包、完整模型类、scheduler、runner、内存池和通用 kernel
  树到本项目。`torch`、Triton、FlashMLA 等运行依赖也不等于 RedKnot 专有源码。

## RedKnot 专有实现不只在 attention/redknot

| 源模块 | 专有责任 | 不可直接搬用的部分 |
| --- | --- | --- |
| `layers/moe/redknot_progressive_topk.py` | 分层 routed Top-K schedule、解析与 assignment-count 估算 | `schedule_from_server_args()` 在第 130 行读取 SGLang 全局参数；需要显式配置注入，估算不是全模型 FLOPs/TTFT |
| `layers/moe/redknot_adaptive_topk.py` | 按原 mixing weight 累计质量选择 K；保留权重不重归一化；可选物理紧凑 K | 第 99 行起依赖 `ForwardBatch.redknot_reuse_plan`；第 193/245 行写 `_sglang_moe_*` tensor marker；依赖 native TopKOutput、expert `-1` 过滤、workspace 和 reduction 契约 |
| `layers/moe/redknot_adaptive_topk_profile.py` | 只读 router 集中度直方图，不修改 routing | 第 50 行入口依赖 request plan 和 TP rank；限制 Flash 的 K6/E256、sqrtsoftplus，不是通用 Pro/Qwen profiler |
| `models/redknot_sparse_moe.py` | Qwen3.5 设计中的 immutable policy、请求局部 context、keep mask/alignment | context 使用 `forward_batch.model_specific_states`；该文件本身不是 MoE executor，不能算 Qwen3.5 模型已接入 |

特别注意：`models/redknot_sparse_moe.py` 的说明声称 sparse executor 在
`models/qwen2_moe.py`，但本次固定源码的 `qwen2_moe.py`、`qwen3_5.py`、
`qwen3_next.py` 未检出 RedKnot 引用。只能确认策略模块存在，不能依据这段
说明文字确认模型调用链已经连接。

Assignment-sparse（减少每 token 的 routed expert assignment）和 token-sparse
（只让部分 token 执行 routed experts）不是同一优化，必须分别标记、启用、
计数和做精度实验，不能互相替代或与 MLA 头比例相加得出全模型节约率。

## 原生文件中的 RedKnot 接入块

以下列出迁移时需要重写的连接点，不是允许复制整份原生文件的清单。

| SGLang 连接点 | 源码位置/责任 | 当前 vLLM 对应及边界 |
| --- | --- | --- |
| `layers/attention/redknot_backend.py` | 第 115 行 `RedKnotAttnBackend(AttentionBackend)`；依赖 ForwardBatch、RadixAttention、原生页面 | `vllm_backend.py` + `runner.py`：MHA/GQA 分头 attention、真实 KV scatter；未迁入原 SegPaged 全链 |
| `layers/attention/redknot_mla_backend.py` | 第 980 行继承 `DeepseekV4AttnBackend`；TP 集体提交、shared-state restore、z_off transaction | `dsv4_backend.py` + `dsv4_runtime.py` + `dsv4_runner.py`：Flash 专用选头和 z_off；不是原 TP/共享状态后台的逐项等价迁移 |
| `layers/attention/deepseek_v4_backend.py` | 第 531 行 snapshot/restore hook；读取/写入原生 SWA/C4/C128 pool | 当前 vLLM 路径保持 native latent KV、compressor、indexer 在线生成，不能宣称已复用这些原生状态 |
| `models/deepseek_v4.py` | 第 1974 行起 attention/projection、异常清理；第 3184 行 token FFN selector；第 3557 行 sparse executor 分支 | `dsv4_projection.py`/`mla.py` 实现 local z_off + online z 聚合并只执行一次 wo_b；FFN selector/executor 没有进入现有插件路径 |
| `models/deepseek_v2.py` | 第 490 行 progressive K，第 821/847 行 adaptive profile/routing，第 1427 行 sparse MoE executor | 需要另写 vLLM MoE adapter，当前原生 FFN/MoE 仍全部在线执行 |
| `model_executor/model_runner.py` | 第 2259 行 RoPE 绑定；第 3307 行 selected rows；第 4600 行 boundary/compressor replay | V1 `GPUModelRunner` hooks 负责本插件上下文/检查；不是 SGLang selected-row、checkpoint-island、TP replay 的移植 |
| `model_executor/forward_batch_info.py` | 第 363 行起 plan、selected rows、compressor schedule 等字段 | 改用 `SamplingParams.extra_args["redknot"]` + vLLM ForwardContext；不复制 ForwardBatch |
| `managers/{io_struct,tokenizer_manager,schedule_batch,scheduler}.py` | plan 规范化/传递、radix 前缀限制、merged-prefill admission | 当前 vLLM 单请求完整 prefill；无原并发/merged-prefill scheduler 的对应实现 |
| `model_executor/piecewise_cuda_graph_runner.py` | 第 308 行 RoPE cache dtype 转换 | 当前插件 eager；不把这段 SGLang graph 初始化逻辑当作 vLLM CUDA graph 支持 |

MoE 的外围原生改动也必须重新适配：

- `layers/moe/fused_moe_triton/fused_marlin_moe.py:185`、
  `layers/moe/moe_runner/triton_utils/fused_moe.py:388` 和 `:518` 消费 route mask。
  只提取 adaptive Top-K 函数，不会自动让 vLLM 的相应 kernel 跳过 assignment。
- `layers/moe/fused_moe_triton/layer.py:586` 有 expert physical-shrink 的
  global-to-local weight-loader 映射。它改变权重/专家布局，不能只复制函数，
  更不能把完整 native loader 纳入 RedKnot 专有代码。

## attention/redknot 内仍有引擎耦合

目录名不能证明模块已独立。例子包括：

- `dsv4_offline_reuse_v2.py`、`dsv4_rope_reloc.py` 使用
  `sglang.jit_kernel.dsv4.attn`、SGLang packed-KV 布局、DSA 量化/旋转。
- `dsv4_shared_latent_sglang.py` 和 `dsv4_shared_snapshot_sglang.py` 是明确的
  SGLang adapter；它们的页面、state pool、stream/event 和 transaction 证据
  必须由 vLLM 的独立 adapter 重新提供。
- `dsv4_reuse_backend_runtime.py` 会回引 `redknot_mla_backend`，不能因导入路径
  被替换就声称脱离引擎；需要断开 native 类型依赖或保留为不可执行参考。
- `qwen35_offline_reuse.py` 动态读取 SGLang server args/DP 状态；Qwen3.5 的
  recurrent state 不能套用普通 MHA KV-cache 接入。

提取时可以保留纯策略/计划/租约/校验逻辑，但应对静态、延迟和动态导入分别
检查。没有直接 `import sglang` 不等于 payload 布局或函数契约已经兼容 vLLM。

## 当前可报告的完成度

现有接入保留非前缀 chunk 内容身份、CPU 字节预算/LRU/租约/全 chunk 原子提交，
并提供 MHA/GQA 以及 Flash 专用的 local/global 在线聚合路径。Flash 的原生
latent KV、compressor、indexer、全宽 FP8 wo_a、FFN/MoE 仍在线执行。

现有运行边界是固定源码、V1、TP/PP/DP=1、eager、单请求完整 prefill。
Pro、Qwen3.5 hybrid state、SegPaged/共享 latent 后台、Sparse FFN/adaptive MoE
执行、TP8、多请求调度，均不能仅因相关参考代码被提取而标记为已接入。

真实模型生成、热态 TTFT、参考答案 F1、并发 QPS、完整存储/传输开销仍需独立
验收。已有 CPU 和 GPU 微内核结果见 `VALIDATION.md`；它们不代替模型复现。
