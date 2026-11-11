# RedKnot / DeepInfer 实验代码保全（2026-11-11）

本目录保全此前已经写入本地的 S1.2 RAG、S2 D-Spark / CUDA Graph，以及长上下文 code-agent 准备代码。它是源码归档，**不是一个新的部署 stage，也不是已完成端到端验证的一键运行包**。此次复制没有执行模型、GPU 实验、服务器部署或评分。

## 必须保留的实验状态

| 部分 | 此次保全的含义与边界 |
| --- | --- |
| S1.2 RAG | 保留 Full/S1.2 的复现入口、算法与系统实现、生命周期和独立结果审计代码。历史资格与结果不等于在新服务器上重新通过准入。64K 入口存在不代表已经获得对应的正式 GPU 性能结论。 |
| 最新 S2 D-Spark / CUDA Graph | **GPU startup FAILED / UNQUALIFIED**。最新一轮 S2 服务未进入可接收正式任务的 ready 状态，closed S2 请求为 0；不能称为运行成功、完成精度验证或获得 agent 加速结果。 |
| 同轮 Full / Prefix(copy) | 服务曾运行并接收请求，但两个官方 agent 任务均达到各自时限，不能据此声称任务完成或官方评分通过。 |
| 长上下文 agent | 仅完成 CPU 侧源码、测试和输入准备设计；尚未完成约 100K 长输入的在线 agent 验证。新增共享仓库材料属于明确的上下文增强协议，不是原始 SWE-bench 默认提示词协议。 |

本目录不携带原始运行日志或结果。上表只记录保全时已经确认的状态，不替代原始证据，也不将失败改写为成功。

## 目录与代码范围

```text
benchmark_rag_redknot.py                 原 Mac 侧 RAG 入口，原字节保留
work/benchmark_rag/                     S1.2 入口、合同、生命周期、reader 与 GatedEngram
work/s2-dspark-agent-v1/                 最新 S2 / Prefix / Full 服务和 D-Spark 适配
  GatedEngram/                         含 DFlash 尾窗桥和图回放兼容的系统运行时
  service/                            chat codec、持久 engine、Prefix(copy)、原生 descriptor
  redknot/                            文档准入、bank、资格与独立验收
  workflow/                           三臂调度、进程身份和生命周期
  framework-overlay/                  既有 OpenHands 启动与 greedy 配置适配
  postrun/                            终态后官方评分和收集入口
  report/                             独立结果 reader
work/s2-agent-v1/
  checkpoint-encoding/                实际 checkpoint 的文本/工具编码器及其原始测试用例
  runtime-readonly/                   原生 profile / runner / request binding 依赖
  framework/                         自研框架适配器、测试、guard 与上游来源/锁文件
work/s2-long-agent-v1/                 两任务自然共享上下文 builder 与首次消息适配
SOURCE_MANIFEST.json                  每个复制文件的来源、SHA256、大小及复制方式
NATIVE_SNAPSHOT_MANIFEST.json          完整历史 NativeSnapshot 的逐文件校验清单
```

`SOURCE_MANIFEST.json` 中 `copy_mode=verbatim` 的文件均保留原始字节。原源码里的绝对机器路径、模型路径、旧 stage 名称和配置约束未被悄悄重写；它们属于历史运行环境的一部分。

唯一从历史源码容器中提取的文件是
`work/s2-dspark-agent-v1/adapted-gated-sources.json`。其内容通过只读 AST 的字面量解析，从旧 `deploy_server.py` 的 `files` 字典取出，没有执行部署脚本。提取后的原始 SHA256 为：

```text
b59d7b430a7b3c915921058b6e036b6473ef1310c774007e0414323d011bc775
```

该 SHA 与当时部署 pin 相同。**这是历史 stage 的 source map，不能被视为此归档目录的新准入证明。** 原始大体积生成式部署载荷未复制进本目录。

## 原生实现与外部依赖

当前 Git 分支根目录已经有原生 DeepInfer 实现，因此本保全目录不再复制另一份原生源码，也不复制历史 Linux `_C.so`。`NATIVE_SNAPSHOT_MANIFEST.json` 记录此前完整 403 文件快照的路径、SHA256 和大小，其中包含历史二进制的哈希而不包含二进制本身。原始快照文件映射摘要为：

```text
d89dad6c23f9df603b7ba27ca3d07717c315bc9d4dcbf7471346552b300cb94f
```

**尚未在这里宣称分支根目录与该快照或远端 PR148 完全一致。** 重新使用前应进行逐文件比较，按实际 Python / PyTorch / CUDA ABI 重建原生扩展和只读 Prefix descriptor，并建立新 CPU 与 GPU 准入。

OpenHands 的完整第三方源码和依赖环境未复制。保留的 `framework-manifest.json` 及锁文件绑定了以下版本：

- `OpenHands/benchmarks`：`60b3548cea8b41276ac0d44ea36a01b82936bbe4`
- `OpenHands/software-agent-sdk`：`43376f1868ffd702746080714a59c16d3f69ec12`
- `OpenHands/extensions`：`4f91a480f74732b2653fdd6f1357637e5575bf91`

仍需另外准备：模型权重及 tokenizer、原始 benchmark 数据及其许可和来源记录、官方 SWE-bench 任务镜像与评分环境、对应 SDK / inference Python 环境、Apptainer、GPU driver / CUDA，以及重新绑定实际部署路径后的配置和证据。此目录不包含 credentials、私有环境变量、模型权重、数据集或 gold、tokenizer、SIF、wheel/venv、完整第三方 vendor、历史运行 session、原始日志、模型输出或会话包。

## 后续恢复时的使用方式

先按 `SOURCE_MANIFEST.json` 校验保全文件，再核对分支原生实现与 `NATIVE_SNAPSHOT_MANIFEST.json`。随后恢复匹配的公开上游代码和合法取得的外部资产，配置新机器路径，重新生成独立 stage/config，并依次执行所需 CPU 测试、真实服务准入和 GPU 数值资格。

不要直接运行或导入历史 `prepare_stage.py`、`make_deployment.py` 等操作脚本来“验证归档”：部分原脚本在顶层就有文件写入或流程动作，且仍指向旧环境。应先检查并为新的环境准备独立配置和调用入口。GPU 实验和评分只能在其对应的新授权与生命周期约束下执行。

长上下文目录中的 `framework_long.install(...)` 只改变真实官方 Conversation 的第一条 user 消息，将原 issue 放在前面、完整自然仓库文件分别作为后续文本块；不会生成预写模型答案，也不以静态 replay 冒充 agent。新 Django pair 所需的任务、镜像、环境基线和实际完整 SDK 输入准入尚未随本归档提供，不能套用旧 Sphinx 的 admission。
