# Full / RedKnot · LongBench 21 × 100

在已安装 DeepInfer GPU 依赖的 Linux 环境中，从仓库或交付包根目录运行以下命令。模型权重使用本地 DeepSeek-V4.1-Flash-1017 checkpoint。数据、评分器、参考日志和 RedKnot 实现均随包提供。

```bash
python3.12 test/benchmark_rag_redknot_21.py --model-path /models/DeepSeek-V4.1-Flash-1017
```

这条命令使用同一驻留模型，在相同输入上运行 Full 与 RedKnot，生成两版完整回答并调用 LongBench v1 的官方评分函数。配置为 8 张 SM90 GPU、EP8、真实 B4、D-Spark 和主模型及草稿模型的 CUDA Graph。运行目录会打印在终端，默认位于 `benchmark-output/redknot-longbench21/`；可用 `--output runs/lb21` 指定新目录。八张卡应由本轮实验独占。首次安装 GPU 软件栈及 SM90 扩展可参见[环境说明](../longbench300/ENVIRONMENT.md)。

## 现有的 500 题

[历史 HTML 对照（下载后用浏览器打开）](recorded/index.html) · [500 题完整问答日志](recorded/answers.jsonl) · [统计 JSON](recorded/summary.json)

主日志来自原实验完整跑完的 HotpotQA、2WikiMultihopQA、MuSiQue、DuReader、NarrativeQA，每集 100 题。每行是一道题，保留下列字段：

- `question`、`references`、`dataset`、`index`、`source_id` 及输入长度、缓存块数、输入 token 哈希。对于原数据没有逐题问题的任务，另附 `task_instruction`。
- `arms.Full.prediction_text` 和 `arms.RedKnot.prediction_text` 是完整的可见回答。各臂还保留原始解码文字、输出 token 数与哈希、停止原因、官方得分、TTFT 和总耗时。
- `metric` 给出该子集的官方评分函数；每臂 `score` 使用 0–1，报告以 0–100 展示。`speedup_vs_full` 为该配对的 TTFT 比值，`first_divergence_token_1based` 是首次分叉位置。

这份日志收录上述五个数据集各 100 题的配对记录。本地新跑的回答写入各自的运行目录。[来源与文件校验值](recorded/provenance.json)

在 CPU 上从主日志重算 1,000 个官方分数并重建报告：

```bash
python3 test/benchmark_rag_redknot_21.py --reference
```

## 冻结输入与计时

21 个子集各 100 个唯一目标，沿用原实验的 527 个 B4。跨数据集的混合批次保留原分组；8 个补位槽仅参与模型执行，报告分母为 2,100。各输入包含 5–15 个自然块，长度档 16K、32K、64K、128K、256K 分别有 359、462、554、545、180 题。108 题保留原始用户提示，1,992 题按原实验拼接。报告分别汇总两类输入。

TTFT 从接收请求计至吐出首个 token，离线 KV 构建另计。每个长度档和子集按配对样本计算 Full 平均 TTFT 除以 RedKnot 平均 TTFT。每个子集报告其官方指标；跨不同指标的总表汇总耗时。每道题都可展开查看问题、全部参考和两版完整回答。

独立验证所有本地输入，或在 GPU 主机上仅做 CPU 运行时检查：

```bash
python3 test/benchmark_rag_redknot_21.py --check
python3.12 test/benchmark_rag_redknot_21.py --check-runtime --model-path /models/DeepSeek-V4.1-Flash-1017
```

输入缓存按冻结数据内容复用，可设置 `--cache /path/to/cache`。首次准备约需数百 MB 的可写磁盘空间。上游 LongBench 数据修订为 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，评分代码修订为 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`。题目、来源标识、参考、完整文档顺序及自然块范围可由随包分片还原。[冻结清单](frozen/cohort.json) · [官方评分源文件](official/metrics.py)

## 查看进度与续跑

每个 B4 的两臂分别落盘。任何时候可在另一个终端按已提交的配对生成快照，写入同一运行目录下的 `report/index.html`、`report/summary.json` 和 `report/answers.jsonl`。使用实际运行目录替换 `runs/lb21`。

```bash
python3 test/benchmark_rag_redknot_21.py --report-only --output runs/lb21
```

中断后在原目录恢复。已封存的单臂和配对会在校验输入与配置后复用，统计只收录提交了两臂的目标题。

```bash
python3.12 test/benchmark_rag_redknot_21.py --resume --output runs/lb21
```
