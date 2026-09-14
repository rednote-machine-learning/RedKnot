# 验证记录

本记录只包含实际执行结果，CPU 检查不代表 GPU 推理、精度或性能验收。

验证环境：2026-09-13，新建独立 vllm-redknot 项目，Mac Python 3.12.14。
目标服务器为 B300 环境，CPU 数值测试使用已有 vLLM 虚拟环境中的 Torch，
设置 `CUDA_VISIBLE_DEVICES=''`，不运行模型或 GPU 实验。

检查范围：缓存管理、请求事务、真实 adapter 的 CPU attention oracle、
paged-KV decode oracle、RoPE/MLA 投影、配置/启动、安全回退、benchmark 统计。

## 实际执行结果

### 四模型实现与 benchmark 资产迁移（0.2.2）

2026-09-14，按用户要求只迁移代码、不运行模型实验。新增四个命名入口、
共享 RAG 数据/指标实现、源模型配置库存，以及 Mistral SWA/MHA 策略、
Qwen3.5 full/linear/conv/GDN 和 sparse MoE 算法模块；native hooks 的待接入
状态分别列于 README 和三个多模型迁移说明。

- Mac Python3.12.14 全套 **255项：200通过、55因缺少Torch/Triton明确跳过**，
  1.096秒；指定固定原仓库路径，来源与目标SHA检查通过。
- 其中新增49项：20项benchmark、16项MHA/SWA、11项Qwen3.5、2项README/索引。
- Ruff check和format检查通过；四个新入口的默认CPU计划和无重型引擎导入检查通过。
- 原Flash runtime、core算法、通用paired benchmark和Flash复现脚本未修改。
- 没有加载模型，没有新增GPU、TTFT、F1、QPS或多卡实验结果；代码迁移不等于
  完整运行时适配。服务器本次增量的复查日志应单独核对，不沿用0.2.1结果。

### 独立 RedKnot 代码迁移与框架导航（0.2.1）

2026-09-14：从同一固定 RedKnot revision 提取43个专有实现模块，另有4个
轻量包初始化文件。仅调整专有模块命名空间并保留来源算法/版权，不复制原生
SGLang engine。每个文件的来源/目标SHA、未提取依赖及运行时接入状态均记录
在 `core_provenance.json`。`core/` 不自动注册到推理链。

- Mac Python3.12.14 全套 **206项：162通过、44因缺少Torch/Triton明确跳过**。
- 新增14项提取契约测试，本地13通过/1跳过：SHA与机械迁移、依赖闭包、轻量
  导入、共享latent事务、存储字节、head行计数、非连续compressor位置。
- 新增20项Flash复现入口CPU测试全部通过；默认预检不导入Torch/vLLM/SGLang。
- 新增4项代码标记/无原生SGLang边界测试全部通过；`code-map`可无GPU读取。
- Ruff check、Ruff format检查通过。保留来源的core使用明确的格式/规则例外，
  不对上游算法做自动重写；来源SHA测试仍覆盖所有提取文件。
- Flash shell的bash语法与入口help通过。只有冻结用例元数据，不伪造token导出；
  真实运行还需完整模型SHA、精确token输入、容量检查及显式GPU授权。

本次不运行GPU模型实验。未增加任何真实模型生成、TTFT/F1/QPS或TP8验收结果。
服务器独立复查于2026-09-14 20:09:40 CST完成：**206项全部通过、无跳过**，
5.957秒，验证退出码0；源码随后交付到 `/workspace/vllm-RedKnot`。
测试显式隐藏GPU，原 `/workspace/RedKnot` 跟踪文件保持不变，共享推理环境未改动。
原始日志保留在 `/workspace/.vllm-RedKnot-deploy.Pa2XoK/verify.log`。
这是0.2.1当时的检查结果，不自动覆盖之后新增的多模型迁移代码。

### Flash 专用路径增量（0.2.0，尚未完成模型复现）

2026-09-13，用户明确选择 vLLM + Flash + RedKnot，新增独立 Flash family。
服务器隔离测试目录 `/workspace/.vllm-flash-smoke.CJDXUhPE`，原仓库不修改。
驱动报告设备为 NVIDIA H20G、Blackwell capability10.3（用户称 B300）。
只使用空闲 GPU0；GPU7 的其他 vLLM 任务保持不动。

- 服务器 **142 项 CPU 测试全部通过**，0.961s、无跳过。
- 随后添加投影 micro-check CLI tests：本地共154项，111通过/43缺Torch跳过；
  服务器最终 **154项全部通过**，1.167s、无跳过。
- DSV4真实 native runner/attention 注册及重复注册通过，15个源码指纹匹配，
  注册测试结束CUDA仍未初始化。
- Sparse GPU micro-check：3个任意head、128候选，以及Flash真实8个global
  head、1024候选两种规模；各自4类案例全部通过native FlashMLA和CPU oracle。
  混合案例最大绝对误差：native=0.001953125，FP32 oracle=0.0078125。
- 原生 FP8 inverse-RoPE/wo_a 的 GPU 投影微测试：4类案例全部通过。
  使用随机权重、G8/H64/D512/R128；native FP8 post-load与E8M0 scale路径。
  部分和最终输出最大绝对误差0.00390625、相对RMS最高约0.00354；
  z最大绝对误差0.0078125。dirty/query行与原生逐位相同；全dirty案例误差0。
  验证只调用一次在线投影和一次wo_b，原始attention、量化权重、scale不变。
- 微测试首次编译/预热不计入计时；不是完整模型TTFT。小行数下选头wrapper
  比原生慢（首组约0.232ms vs0.053ms），包含同步/H2D/分配开销，不隐瞒此点。
- 0.2.0 wheel 构建成功；服务器仅安装到该stage的 `isolated-site`，
  共享 `/workspace/vllm/.venv` 未被改动。

原始服务器结果为 `cpu-tests.log`、`native-registration.log`、
`sparse-global8-report.json`、`projection-report.json`。vLLM日志可能出现在
投影JSON前，解析时要保留原文件并从顶层JSON开始读取。

**权重正在下载；尚未加载真实Flash模型，无完整生成/TTFT/F1结果。**
上述投影实验是同上下文数学分解，不能验证独立非前缀chunk的模型误差。

### 初始 MHA/GQA 交付（0.1.0，历史记录）

| 检查 | 结果 |
| --- | --- |
| 本地 unittest | 80 项中 66 通过、14 项因本地没有 Torch 明确跳过 |
| B300 服务器 CPU unittest | **80 项全部通过，无跳过**，测试体耗时 0.614 秒 |
| Ruff lint | All checks passed |
| Ruff format | 通过 |
| 独立本地包构建/安装 | vllm-redknot 0.1.0 成功；general plugin entry point 可发现 |
| 真实 vLLM 源码指纹 | SOURCE_CONTRACT_OK，11 个文件匹配固定版本 |
| 真实 native backend 导入 | NATIVE_BACKEND_IMPORT_OK CUSTOM |
| 真实 V1 runner 插件注册两次 | PLUGIN_REGISTRATION_IDEMPOTENT_OK |
| 导入和插件检查后的 CUDA 状态 | torch.cuda.is_initialized() 为 False |

服务器环境使用 `/workspace/vllm/.venv/bin/python`（Python 3.12、
Torch 2.13.0+cu130）。完整 suite 在独立暂存源码上执行，未安装依赖到共享环境，
未修改 `/workspace/vllm` 或 `/workspace/RedKnot`，未启动或停止 GPU 进程。
导入时的 “No CUDA runtime is found” 提示来自显式隐藏 GPU 的 CPU 检查，
不代表服务器 CUDA 驱动故障。

交付目录中可复查：

```bash
cd /workspace/vllm-redknot
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. /workspace/vllm/.venv/bin/python -B -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /workspace/vllm/.venv/bin/python -B \
  -m vllm_redknot doctor
```

服务器是源码交付，未往共享 vLLM 环境安装插件，所以该环境的 doctor 会报告
`plugin_installed: false`。这与本地隔离环境的安装验证不同；真正启动服务前应
在选定独立推理环境安装项目。未启动真实 LLM、未加载 checkpoint、未测
GPU TTFT/F1/QPS；也未执行 DeepSeek V4 serving。CPU 契约通过不能替代这些验证。
