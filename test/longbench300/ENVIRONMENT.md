# Environment for the 300-question Full / S1.6 replay

Run commands from the repository root. You can audit all 300 frozen prompts or rescore the 600 archived answers on a CPU with Python 3.12 and its standard library.

```sh
python3.12 test/benchmark_rag_redknot.py --check
python3.12 test/benchmark_rag_redknot.py --reference
```

## GPU host

The recorded environment used eight Hopper GPUs reporting compute capability `(9, 0)`, Linux x86-64, CPython 3.12.3, PyTorch `2.13.0+cu132`, CUDA toolkit `13.2` (`nvcc` 13.2.86) and Triton `3.7.1`. The upstream build requires GCC 13 or later. Building native extensions needs a full CUDA toolkit, including headers, libraries and `nvcc`. Set `CUDA_HOME` to the toolkit root if it is installed outside the default path.

In a fresh environment, install Python dependencies using the checkout's NVIDIA requirements and the NCCL pin from its build script. The PyTorch wheel location for CUDA 13.2 is already in the requirements file. The pins also cover FlashAttention 4, TileLang, Tile Kernels and the tokenizer. [NVIDIA requirements](../../requirements-nvidia.txt) · [Base requirements](../../requirements.txt) · [Upstream build script](../../build.sh)

```sh
python3.12 -m venv .venv
. .venv/bin/activate
export DEEPINFER_ACCELERATOR=nvidia
python -m pip install 'setuptools>=77.0.3' wheel
python -m pip install -r requirements-nvidia.txt
python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
nvcc --version
g++ --version
python - <<'PY'
import torch, triton
caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
print(torch.__version__, torch.version.cuda, triton.__version__, caps)
assert caps == [(9, 0)] * 8, 'Expose the eight SM90 devices for this replay'
PY
```

## Native code and speculative decoding

The upstream NVIDIA instructions target SM100. Their vendor build generates local wheels for DeepSelect, FlashMLA, DeepEP and DeepGEMM. The Hopper replay needs the matching SM90 `deepinfer._C` extension and vendor revisions. A fresh benchmark run builds a missing native extension automatically if the vendor packages and toolkit are installed. To prepare an extension ahead of time, the SM90 source helper builds `_C` against the active Python and PyTorch ABI. `--build-vendors` fetches the pinned Hopper sources for DeepSelect, DeepEP and DeepGEMM, builds their wheels and installs them in this environment. [Upstream build instructions](../../README.md) · [Upstream vendor build](../../scripts/build_vendor.sh)

```sh
python test/longbench300/sm90_source.py --repo "$PWD" \
  --target "$PWD/benchmark-output/NativeSnapshot" --build-vendors --build
export DEEPINFER_SM90_ROOT="$PWD/benchmark-output/NativeSnapshot"
```

Use the S1.6 runner and its matched GatedEngram implementation. The top-level GatedEngram example implements an earlier adjacent-SWA / FFN-only experiment. D-Spark runs inside DeepInfer. The saved S1.6 configuration uses five speculative tokens, both CUDA graphs, four requests per batch and eight expert-parallel ranks. [GatedEngram example](../../GatedEngram/README.md)

## Model and validation

Use the full `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint. The replay checks the frozen model metadata, tokenizer and encoder identities and checks that every shard in the weights index exists and has content. Point `REDKNOT_MODEL_PATH` at an existing snapshot or download one before launching. [Model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

```sh
export REDKNOT_MODEL_PATH="$PWD/models/DeepSeek-V4.1-Flash-1017"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$REDKNOT_MODEL_PATH"
python test/benchmark_rag_redknot.py --check-runtime
```

For an existing compatible S1.6 release, pass its root with `--runtime-root` or set `REDKNOT_RUNTIME_ROOT`. The runtime check validates configuration on CPU. A fresh timing measurement requires the GPU replay. The reported performance came from the recorded eight-GPU environment. The source build commands above have not been qualified through a fresh install and a 300-question GPU run in a clean environment.
