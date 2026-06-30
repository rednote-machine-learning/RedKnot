# SegPaged v2 — 收益记录（H2O 吞吐 & DuoAttention）

本文件记录 head-level 段页式存储（`segpaged_v2`）在两类工业多头 attention 场景下
的实测收益：

- **H2O / Heavy-Hitter decode**：decode 阶段 per-head 稀疏 KV。
- **DuoAttention prefill**：retrieval/global 头 + streaming/local 头的多头策略。

所有数字在 **NVIDIA L20Y, bf16** 上实测，随机合成数据。脚本均自带数值等价校验
（`h2o_dense vs h2o_segpaged` / `segpaged vs dense+mask` 的 cosine ≈ 1）。

> 重要前提：延迟对比只覆盖 **attention 单算子**。端到端 TPOT/TTFT 还会被
> QKV/O projection、MLP 等摊薄，真实端到端提速通常小于此处的算子倍数。
> KV 节省与数值等价是确定性结论，延迟倍数随基线/形状变化。

---

## 1. H2O / Heavy-Hitter decode

脚本：`test/srt/redknot/test_H2O_SegPagedAttention.py`

机制（对齐 H2O, Zhang et al., NeurIPS 2023）：

- per-head 累积 softmax 注意力分数（真实逐步概率）。
- 固定预算 `budget = heavy + recent`；prompt 先压到 budget，再 rolling 贪心驱逐
  recent window 外累积分最低的 token。
- 每个 KV head 独立累计、独立驱逐 —— 这正是 head-level 差异。

对比三路，**统一使用同一个生产 decode kernel** `flash_attn_with_kvcache`
（batched flash-decoding），仅 KV 长度不同，保证公平：

- `dense_full`：每头读全量 L。
- `h2o_dense`：H2O 保留集，从一体化 dense KV gather。
- `h2o_segpaged`：H2O 保留集物理存进 SegPaged v2 每头页表。

### 一键 suite 结果（`--suite`）

| batch | L_total | budget | dense_ms | seg_ms | speedup | seg 吞吐(tok/s) | KV 节省 | cos(dense=seg) |
|------:|--------:|-------:|---------:|-------:|--------:|----------------:|--------:|---------------:|
| 8 | 8224  | 768  | 0.099 | 0.021 | 4.78x  | 12,112 | 90.7% | 1.00000 |
| 8 | 16416 | 768  | 0.182 | 0.021 | 8.83x  | 12,143 | 95.3% | 1.00000 |
| 8 | 32800 | 1280 | 0.349 | 0.023 | 14.94x | 10,700 | 96.1% | 1.00000 |

要点：

- **吞吐提升随上下文增长**：4.78x → 14.94x（dense 要读的 KV 越多，差距越大）。
- **KV 节省 90–96%**：固定预算决定，直接对应显存与并发容量。
- `h2o_dense vs h2o_segpaged cos = 1.0`：段页式存储**不改变计算结果、不损失算力**；
  二者算的是同一稀疏集，单次 attention 耗时几乎相同（约 1.0x）。段页式的增量价值
  在**存储**（显存/并发），而非单算子速度。
- `dense_full vs h2o cos` 偏低是 H2O 稀疏近似误差（随机数据放大），真实质量需接
  真实模型评测。

### 关于早期 “50x”

早期版本用单 query varlen 调用做 dense 基线（decode 最差布局），导致 speedup 虚高
到 ~50x。改用公平的 batched flash-decoding kernel 后为 **4.78x ~ 14.94x**，
这才是可引用的数。

### 运行方式

```bash
# 一键 suite（推荐）
CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_H2O_SegPagedAttention.py --suite

# 单组自定义配置
CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_H2O_SegPagedAttention.py \
    --batch 16 --prefill 32768 --gen 64 --heavy 1024 --recent 256 --repeat 50
```

---

## 2. DuoAttention prefill

脚本：`test/srt/redknot/test_segpaged_duo_attention.py`

机制：同一 DuoAttention 多头策略（global/retrieval 头看全量，streaming/local 头看
`sink + recent`），对比两条引擎：

- `dense_fa2`：token-level 一体化存储，每头全量 KV，FA-2 计算；local 头用
  sliding `window_size` 表达稀疏（稀疏只在计算时，存储不省）。
- `segpaged_v2`：per-head 段页式存储，local 头物理只存 `sink+recent`，一次 FA-2
  varlen 按每头真实长度计算（无 mask、不存不可见 KV）。

### 实测结果（Hkv=8 / Hq=32 / global=2 / local=6, q_len=256）

| 上下文 L | 数值 cos | 每层延迟 dense→seg | speedup | TTFT(×32层) | 吞吐 dense→seg | KV 节省 |
|---------:|---------:|-------------------:|--------:|------------:|---------------:|--------:|
| 8192  | 0.9999974 | 5.53 → 0.67 ms | 8.28x | 177 → 21 ms  | 46k → 383k tok/s | 72.6% |
| 32768 | 0.9999973 | 6.29 → 2.44 ms | 2.57x | 201 → 78 ms  | 41k → 105k tok/s | 74.4% |
| 65536 | 0.9999974 | 7.46 → 4.07 ms | 1.83x | 239 → 130 ms | 34k → 63k  tok/s | 74.7% |

要点：

- **数值与 dense+mask 基线等价**（cos ≈ 1）。
- **KV 节省约 72–75%**：local 头只存窗口。
- speedup 在更短上下文更高（dense 的 sliding-window 在长序列上 kernel 效率更高，
  segpaged 优势收窄但仍 > 1）。
- 关键工程点：段页式的布局工作（gather + pack + cu_seqlens）必须在 KV 写入时
  **一次性完成并复用**；若每次 attention 现算，host 开销会吃掉收益（早期实测过
  0.70x 的反例）。

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_segpaged_duo_attention.py
CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/test_segpaged_duo_attention.py \
    --seq-len 65536 --q-len 256 --global-ratio 0.25 --sink 4 --window 256 --repeat 20
```

---

## 3. 结论

| 场景 | 阶段 | attention 算子提速 | KV 节省 | 数值等价 |
|------|------|--------------------|---------|----------|
| H2O / Heavy-Hitter | decode | 4.8x ~ 14.9x（随上下文） | 90–96% | cos = 1.0 (vs 同策略 dense) |
| DuoAttention | prefill | 1.8x ~ 8.3x（随上下文） | 72–75% | cos ≈ 1.0 (vs dense+mask) |

head-level 段页式存储的核心价值：在不改变计算结果的前提下，把模型侧已存在的
head 级稀疏/差异，落到 KV **存储层**——decode 阶段减少 KV 读量、prefill 阶段
减少计算与存储，并通过 KV 容量节省提升并发。延迟倍数仅为 attention 算子级，
端到端需结合 projection/MLP 占比综合评估。
```
