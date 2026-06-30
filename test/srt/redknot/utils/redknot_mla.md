# RedKnot MLA for DeepSeek V4 — 设计与实现

> 适用范围：`attention_backend == "redknot_mla"`（DeepSeek‑V4‑Flash / DeepSeek V4 MLA 系列）。
> 本文档说明 RedKnot MLA 的**动机（为什么）**、**设计（怎么做）** 和 **实现（代码在哪、如何工作）**，并给出可复现的运行/调参方式。

---

## 0. TL;DR

RedKnot MLA 是面向 **DeepSeek V4 MLA + MoE** 的一套 **推理期稀疏化** 方案，目标是在长上下文 RAG 场景下降低 KV 访问与 FFN 计算量，同时尽量不损精度。它由两块互相独立、可分别开关的机制组成：

1. **逻辑头级 KV 稀疏（Attention 侧）**：把每层的逻辑注意力头分成 `dense / global / local` 三类。`local` 头只看最近的滑窗（SWA）缓存，`global`/`dense` 头额外看压缩 KV。物理 KV 缓存仍是 DeepSeek V4 原生的打包 FlashMLA latent 缓存，**不展开成逐头 K/V**。
2. **Token 选择性 FFN（FFN 侧）**：在每层 FFN 之前，用 post‑attention 激活幅度作为 token 重要度的代理，按 **top‑p 质量阈值** 选出少数 token 走 MoE，其余 token 走残差恒等路径（不进 FFN）。

此外还提供一个 **离线 Profiler**，在单序列 prefill 上测量每个 (layer, head) 的「注意力质量 vs 距离」分布，自动产出逐头分类的 JSON 策略文件。

---

## 1. 动机（Why）

### 1.1 长上下文推理的两个主要成本

在 RAG / 长文档问答里，序列长度 `T` 很大（8K~128K），DeepSeek V4 推理的两个主要开销是：

- **Attention 的 KV 访问**：即便 MLA 已经把 KV 压成单个 latent 流，prefill/decode 仍要在长序列上做注意力，KV 读取与 FlashMLA 计算随上下文增长。
- **MoE FFN 的计算**：每个 token 都要过一遍 MoE（路由 + 专家 GEMM）。长上下文里**多数 token 对最终答案贡献很小**，对它们全量计算 FFN 是浪费。

### 1.2 关键观察

- **头的局部性差异**：不是所有注意力头都需要全局视野。很多头的注意力质量集中在最近的一小段窗口内（局部头），只有少数头需要回看很远（全局头/检索头）。如果能识别出局部头，就能把它们限制在 SWA 窗口内，省掉对远端 KV 的访问。
- **token 的重要度差异**：经过若干层后，token 的 hidden 表示幅度分布高度不均；少量 token 携带了大部分「能量」。对长尾低幅度 token 跳过 FFN，对结果影响有限。

### 1.3 为什么是「MLA 变体」而不是直接复用原 RedKnot

原始 RedKnot（`redknot` 后端，见 `redknot_backend.py`）面向**显式逐头 K/V**的常规模型（Llama/Qwen 等），在 `num_key_value_heads` 粒度上做分类（local/global/retrieval/dense）。

DeepSeek V4 是 **MLA**：物理上只有 **1 个 KV 头**（共享 latent KV），逻辑上有很多 attention 头。把 latent KV 展开成逐头 K/V 代价高、且违背 MLA 的初衷。所以 RedKnot MLA：

- 在 **逻辑 attention 头**（`num_attention_heads`）粒度上分类，而不是物理 KV 头；
- **不展开 latent KV**，而是把头分组后对同一份 latent 缓存发起**多次 FlashMLA 调用**（每组一次），靠 FlashMLA 自身的 SWA/extra-cache 机制实现不同头看不同范围；
- 分类只用 `local/global/dense`（无 `retrieval`），更简单。

---

## 2. 设计（How）

RedKnot MLA = **逻辑头 KV 稀疏** ⊕ **Token 选择性 FFN**，两者解耦，可独立开关。

### 2.1 逻辑头分类策略

每个 (layer, logical_head) 被分到三类之一：

| 类型 | 含义 | 看到的 KV |
|------|------|-----------|
| `dense` | 全注意力，不做稀疏 | 走原生 DSV4 路径（SWA + 压缩 extra） |
| `global` | 需要全局视野 | SWA 窗口 + 压缩 extra-cache |
| `local` | 只需局部视野 | 只看 SWA 窗口（不取 extra-cache） |

**默认（无 profiler 时）的保守策略**（`from_model_config`）：

1. 前 `dense_prefix_layers` 层（默认 2）整层 `dense` —— 保护早期层信号；
2. 其余层：每隔 `global_head_stride`（默认 8）个头取 1 个为 `global`，其余为 `local`，局部窗口 = `local_window`（默认 128）；
3. 若 `global_layer_stride > 0`：层号 `% global_layer_stride == 0` 的整层全设为 `global`（默认 0，关闭）。

示意（`dense_prefix=2, global_head_stride=8, local_window=128`）：

```
layer 0,1 : 全部 dense
layer 2+  : head 0,8,16,... = global；其余 = local(window=128)
```

**策略来源有两种**：
- 内置默认策略（上面规则）；或
- 离线 Profiler 产出的逐头 JSON（`redknot_head_config_path` 指定），更精细。

### 2.2 离线 Profiler（自动产策略）

在单序列 prefill 上，对每层每个头测量「注意力质量随 query‑key 距离的分布」，据此判断每个头需要多大的窗口才能覆盖 `coverage`（默认 95%）的注意力质量：

- 覆盖窗口 `w ≥ global_window_ratio × 上下文长度` ⇒ 判为 `global`；
- 否则判为 `local`，窗口 = `ceil(w × window_safety)` 向上取整到 64 的倍数；
- 前 `dense_prefix_layers` 层强制 `dense`。

产出标准 JSON（`mla_head_classification` / `mla_head_max_distance` / `mla_head_sink_size`），供 `redknot_mla` 后端直接加载。

### 2.3 Token 选择性 FFN

每层 FFN 之前，按 token 重要度选子集走 MoE：

1. 重要度 = post‑attention（pre‑FFN）hidden 的 L2 范数 `‖h_t‖`；
2. 对重要度降序排序，取累计质量达到 `mass_thresh` 之前的 token（top‑p）；
3. 强制保留：第 1 个 token、最近 `recent_n` 个 token；
4. 选中的 token 走 MoE，未选中的输出置零 → 经残差后等价于「跳过 FFN，走恒等」。

分层阈值：
- 前 `dense_until` 层（默认 4）不稀疏（全量 FFN）；
- 普通层用 `mass_thresh`（默认 0.30）；
- 深层（`layer ≥ deep_start`，默认 24）用更激进的 `mass_thresh_deep`（默认 0.10）。

> 直觉：越深的层，token 能量越集中，可以更激进地丢弃长尾 token。

### 2.4 安全回退（Fallback）

逻辑头后端在以下情况自动回退到原生 DSV4 路径，避免错误地套用逻辑头策略：
- forward metadata 不是 `DSV4AttnMetadata`；
- 头数与策略不匹配（如 TP / 投机解码导致的头视图变化）；
- 某层的 `local` 或 `global` 头集合为空。

---

## 3. 实现（Where & How）

### 3.1 文件总览

| 组件 | 文件 | 关键符号 |
|------|------|----------|
| MLA 注意力后端 | `python/sglang/srt/layers/attention/redknot_mla_backend.py` | `RedKnotMLAAttnBackend.forward` |
| 逻辑头分类配置 | `python/sglang/srt/layers/attention/redknot/deepseek_v4_mla.py` | `DeepSeekV4MLAHeadConfig` |
| 离线 Profiler | `python/sglang/srt/layers/attention/redknot/mla_head_profiler.py` | `MLAHeadLocalityCollector` |
| 稀疏 FFN + 逐层 forward | `python/sglang/srt/models/deepseek_v4.py` | `_select_redknot_sparse_ffn_tokens`、`DeepseekV4DecoderLayer.forward` |
| 服务端参数 / CLI | `python/sglang/srt/server_args.py` | `redknot_mla_*` / `redknot_sparse_ffn_*` |
| 原始 RedKnot（非 MLA，对照） | `python/sglang/srt/layers/attention/redknot_backend.py` | `RedKnotAttnBackend` |
| 基准脚本 | `test/srt/redknot/benchmark_RedKnot_DeepSeek V4_RAG.py` | `_engine_kwargs` / profiling 流程 |

### 3.2 逻辑头注意力后端

`RedKnotMLAAttnBackend` 继承自 `DeepseekV4AttnBackend`，复用其缓存管理与 metadata，只在 `forward` 里加入逻辑头分组。

构造（`redknot_mla_backend.py:38-61`）：优先从 `redknot_head_config_path` 加载 JSON 策略；否则用 `from_model_config` 构建默认策略。

`forward` 核心步骤（`redknot_mla_backend.py:63-225`）：

1. metadata 校验，非 `DSV4AttnMetadata` 直接回退父类（`:79-90`）；
2. 取该层逐头策略张量并得到 `is_local` 布尔掩码（`:107-108`），头数不匹配/缺一类则回退（`:109-138`）；
3. 准备 SWA latent 缓存 `swa_k_cache`（`:140-145`），以及按 `compress_ratio ∈ {4,128}` 取压缩 `extra_k_cache` 与对应 page indices / topk lengths（`:147-169`）；
4. 分两组各发一次 FlashMLA（`:196-220`）：
   - `global` 组：`use_extra=True`（SWA + 压缩 extra）；
   - `local` 组：`use_extra=False`（仅 SWA）；
   - 关键：两组都用同一份 latent `swa_k_cache`，靠 `extra_*` 参数区分可见范围，**不展开逐头 K/V**；
5. 用 `index_copy_` 把两组输出按原头位置拼回（`:222-225`）。

```python
# redknot_mla_backend.py:215-224（节选）
q_global = q.index_select(2, global_idx)
o_global = run_flashmla(q_global, attn_sink.index_select(0, global_idx), True)   # SWA + extra
q_local  = q.index_select(2, local_idx)
o_local  = run_flashmla(q_local,  attn_sink.index_select(0, local_idx),  False)  # SWA only
out.index_copy_(2, global_idx, o_global)
out.index_copy_(2, local_idx,  o_local)
```

### 3.3 逻辑头分类配置 `DeepSeekV4MLAHeadConfig`

`deepseek_v4_mla.py`：

- `from_model_config(config, dense_prefix_layers, local_window, global_head_stride, global_layer_stride)`：构建默认保守策略（见 §2.1）。
- `get_strategy(layer, head) -> MLAHeadStrategy{head_type, window, sink_size}`：查单个 (layer, head) 策略。
- `layer_tensors(layer_id, device)`：返回并缓存该层张量 `{type_ids, windows, sinks, is_local}`，供后端按设备复用。
- `to_json / from_json`：与 Profiler 产出的 JSON 互通（键：`mla_head_classification`、`mla_head_max_distance`、`mla_head_sink_size`）。
- `is_deepseek_v4_mla_config(config)`：识别 DeepSeek V4 MLA（`model_type=="deepseek_v4"` 且 `num_key_value_heads==1` 且具备 `q_lora_rank/o_lora_rank/qk_rope_head_dim`）。

### 3.4 离线 Profiler `MLAHeadLocalityCollector`

`mla_head_profiler.py`：

- `MLAHeadProfileConfig`：`coverage=0.95`、`sample_queries=256`、`global_window_ratio=0.5`、`window_safety=1.5`、`window_round_to=64`、`window_min=64`、`dense_prefix_layers=2`，以及对数距离分箱 `bin_edges`。
- `observe_layer(layer_id, q, latent_k, softmax_scale)`：对采样的 query 行计算与 latent key 的 logits → softmax → 按 `dist=i-j` 分箱累计质量，得到每头「质量 vs 距离」直方图。
- `_coverage_window`：对每头求达到 `coverage` 的最小距离箱。
- `build_head_config`：按 §2.2 规则分类，产出 `DeepSeekV4MLAHeadConfig`。
- `export_json(path)`：导出策略 JSON。
- 进程级单例：`enable_global_collector` / `get_global_collector`。

接入点：`MQALayer._maybe_profile_mla_heads`（`deepseek_v4.py:774-829`），仅在 **TP=1 的单序列 prefill** 路径触发，取 MLA 压缩前的 `q` 与 latent `kv` 喂给 collector，最后一层后导出 JSON。

> 注意：profiler 走的是非融合的 `_forward_prepare` 路径，因此开启 profiling 时会**自动禁用多流 overlap 和 CUDA graph**（见 `arg_groups/deepseek_v4_hook.py`）。

### 3.5 稀疏 FFN

选择函数 `_select_redknot_sparse_ffn_tokens`（`deepseek_v4.py:1212-1246`）：

```python
# 关键逻辑（节选）
if (not redknot_sparse_ffn_enable
        or layer_id < redknot_sparse_ffn_dense_until
        or hidden_states.shape[0] <= 1):
    return None                      # 不稀疏

mass_thresh = mass_thresh_deep if layer_id >= deep_start else mass_thresh
importance = hidden_states.float().norm(dim=-1)        # token 重要度 = L2 范数
sorted_imp, sorted_idx = torch.sort(importance, descending=True)
cum = torch.cumsum(sorted_imp, 0) / importance.sum()
rank_keep = cum < mass_thresh
rank_keep[0] = True                  # 至少保留最重要的 token
keep = scatter(rank_keep -> 原顺序)
keep[-recent_n:] = True              # 强制保留最近 recent_n 个 token
return keep
```

逐层 forward 组合（`DeepseekV4DecoderLayer.forward`，`deepseek_v4.py:1248-1378`）：

```
# 注意力子层
residual = h
h = hc_pre(h, attn_fn, ..., norm=input_layernorm)     # mHC 预混合（含 RMSNorm 融合）
h = self_attn(h)                                       # ← redknot_mla 后端在此生效
h = hc_post(h, residual, post, comb)                   # mHC 后混合 + 残差

# FFN 子层
residual = h
h = hc_pre(h, ffn_fn, ..., norm=post_attention_layernorm)
keep = _select_redknot_sparse_ffn_tokens(h)            # ← 稀疏 FFN 选择
if keep is None or keep.all():
    h = self.mlp(h, ...)                               # 全量 MoE
else:
    idx = nonzero(keep)
    out = zeros_like(h)
    out[idx] = self.mlp(h[idx], ...)                   # 仅选中 token 走 MoE
    h = out                                            # 其余置零 → 残差恒等
h = hc_post(...)                                        # 残差
```

其中 `hc_pre`/`hc_post` 是 DeepSeek V4 的 mHC（multi‑Head‑Channel mixing）算子，默认走 TileLang/DeepGEMM 高性能 kernel。

### 3.6 与原始 RedKnot 的关系

`redknot_backend.py` 的 `RedKnotAttnBackend` 是面向**显式逐头 K/V** 的通用实现，分类粒度是 `num_key_value_heads`，类型含 `local/global/retrieval/dense`，并支持 SegPagedAttention、离线 KV 段拼接等。`redknot/` 子目录里 `head_config.py`（通用）与 `deepseek_v4_mla.py`（MLA 专用）是两套并列的分类配置；MLA 变体是其在 MLA 架构上的「不展开 KV、逻辑头分组多次 FlashMLA」的特化。

---

## 4. 参数参考

### 4.1 逻辑头策略（`server_args.py`）

| 参数 / CLI | 默认 | 说明 |
|------------|------|------|
| `redknot_mla_dense_prefix_layers` / `--redknot-mla-dense-prefix-layers` | 2 | 前 N 层强制 dense |
| `redknot_mla_local_window` / `--redknot-mla-local-window` | 128 | local 头滑窗大小（token） |
| `redknot_mla_global_head_stride` / `--redknot-mla-global-head-stride` | 8 | 每隔 N 个头取 1 个 global |
| `redknot_mla_global_layer_stride` / `--redknot-mla-global-layer-stride` | 0 | >0 时每隔 N 层整层 global；0 关闭 |
| `redknot_head_config_path` / `--redknot-head-config-path` | None | 逐头分类 JSON 路径（优先于默认策略） |

### 4.2 稀疏 FFN（`server_args.py`）

| 参数 / CLI | 默认 | 说明 |
|------------|------|------|
| `redknot_sparse_ffn_enable` / `--redknot-sparse-ffn-enable` | False | 开启 token 选择性 FFN |
| `redknot_sparse_ffn_dense_until` / `--redknot-sparse-ffn-dense-until` | 4 | 前 N 层不稀疏 |
| `redknot_sparse_ffn_mass_thresh` / `--redknot-sparse-ffn-mass-thresh` | 0.30 | 普通层 top‑p 质量阈值 |
| `redknot_sparse_ffn_deep_start` / `--redknot-sparse-ffn-deep-start` | 24 | 深层阈值起始层 |
| `redknot_sparse_ffn_mass_thresh_deep` / `--redknot-sparse-ffn-mass-thresh-deep` | 0.10 | 深层 top‑p 质量阈值 |
| `redknot_sparse_ffn_recent_n` / `--redknot-sparse-ffn-recent-n` | 256 | 强制保留最近 N 个 token |

### 4.3 Profiler（`server_args.py`，经 Engine kwargs 使用）

| 参数 | 默认 | 说明 |
|------|------|------|
| `redknot_mla_profile_enable` | False | 开启离线头局部性 profiling |
| `redknot_mla_profile_out` | None | 策略 JSON 导出路径 |
| `redknot_mla_profile_coverage` | 0.95 | 覆盖目标（注意力质量占比） |
| `redknot_mla_profile_sample_queries` | 256 | 每层采样 query 行数 |
| `redknot_mla_profile_global_window_ratio` | 0.5 | 覆盖窗口超过上下文该比例 ⇒ global |
| `redknot_mla_profile_window_safety` | 1.5 | 实测窗口的安全放大系数 |

---

## 5. 基准脚本用法（`benchmark_RedKnot_DeepSeek V4_RAG.py`）

脚本通过环境变量驱动，`_engine_kwargs` 将其转换为 `sgl.Engine(...)` 参数。

### 5.1 离线产策略

```bash
REDKNOT_MLA_PROFILE=1 \
REDKNOT_MLA_PROFILE_OUT=head_class/dsv4_mla_head_config.json \
REDKNOT_MLA_PROFILE_COVERAGE=0.95 \
REDKNOT_MLA_PROFILE_GLOBAL_RATIO=0.5 \
REDKNOT_MLA_PROFILE_WINDOW_SAFETY=1.5 \
python "test/srt/redknot/benchmark_RedKnot_DeepSeek V4_RAG.py"
```

### 5.2 推理对比（baseline dsv4 vs redknot_mla + 稀疏 FFN）

```bash
REDKNOT_MLA_LOCAL_WINDOW=256 \
REDKNOT_MLA_GLOBAL_HEAD_STRIDE=8 \
REDKNOT_MLA_DENSE_PREFIX_LAYERS=2 \
REDKNOT_FFN_MASS=0.30 \
REDKNOT_FFN_MASS_DEEP=0.10 \
REDKNOT_FFN_DENSE_UNTIL=4 \
REDKNOT_FFN_RECENT_N=256 \
REDKNOT_HEAD_CFG=head_class/dsv4_mla_head_config.json \
python "test/srt/redknot/benchmark_RedKnot_DeepSeek V4_RAG.py"
```

### 5.3 常用运行/资源相关环境变量

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `REDKNOT_MODEL_PATH` | （内置 DeepSeek‑V4‑Flash 路径） | 模型权重路径 |
| `REDKNOT_DATASETS` | `hotpotqa,2wikimqa,musique,multifieldqa_en` | LongBench 数据集 |
| `REDKNOT_LENGTHS` | `8K` | 上下文长度档位（逗号分隔） |
| `REDKNOT_N_SAMPLES` | 1 | 每数据集样本数 |
| `REDKNOT_MAX_NEW` | 32 | 生成 token 数 |
| `REDKNOT_TP_SIZE` | 1 | tensor parallel 大小 |
| `REDKNOT_MEM_FRACTION_STATIC` | （引擎默认） | 静态显存占比 |
| `REDKNOT_MAX_TOTAL_TOKENS` | 0（不限） | KV 总 token 上限 |
| `REDKNOT_MOE_RUNNER_BACKEND` | （引擎默认/auto） | MoE runner 后端（如 `deep_gemm`） |
| `REDKNOT_DISABLE_CUDA_GRAPH` | 0 | 1 = 禁用 CUDA graph |
| `REDKNOT_SKIP_SERVER_WARMUP` | 0 | 1 = 跳过 warmup |

---

## 6. 设计取舍小结

1. **不展开 latent KV**：逻辑头分组后多次 FlashMLA，保留 MLA 的内存优势，代价是每层多一次 kernel 调用（按头分组）。
2. **离线 profiling 先于压缩**：用单序列实测每个头「需要多大窗口」，再生成保守策略，避免推理期昂贵的逐 token 重要度评分。
3. **FFN 用激活幅度做重要度代理**：避免额外打分网络/注意力质量回读，`O(T)` 排序即可；并用「保最近 `recent_n` + 保第一个」兜底，防止丢关键 token。
4. **可组合、可回退**：逻辑头稀疏与稀疏 FFN 各自独立开关；头视图不匹配或 metadata 异常时自动回退原生 DSV4，保证正确性优先。
5. **全参数化**：所有阈值（dense_prefix / local_window / global_stride / mass_thresh 等）均可经 CLI 或 Engine kwargs 调整，便于经验性策略搜索。
