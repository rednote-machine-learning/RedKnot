"""Materialize the pinned DeepInfer Hopper source tree used by the RedKnot S1.6 benchmark.

The default operation performs no builds, installs, network access or GPU work. On the GPU host,
--build compiles deepinfer._C with the current Python and PyTorch C++ ABI. --build-vendors separately
builds the three pinned public Hopper operator packages if the environment needs them.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "sm90_overlay"
PINNED_INVENTORY = "76fa676184577adc68eecbde4a54cbbf17295ae74b2bfbce349bbfa827a1370a"
VENDORS = {
    "DeepSelect": ("https://github.com/deepseek-ai/DeepSelect.git", "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18", "deep_select"),
    "DeepEP": ("https://github.com/deepseek-ai/DeepEP.git", "93eb6eb238127e96c6d7a4a625a6dad158348509", "deep_ep"),
    "DeepGEMM": ("https://github.com/deepseek-ai/DeepGEMM.git", "1fc9c66975ee322d628b3288534ca0e9ce12a64d", "deep_gemm"),
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition, msg):
    if not condition:
        raise ValueError(msg)


def _inventory():
    path = OVERLAY/"sources.json"
    _check(_sha(path) == PINNED_INVENTORY, "Hopper source inventory checksum mismatch")
    return json.loads(path.read_bytes())


def _checked(base, rel, expected):
    path = base / rel
    _check(path.is_file() and _sha(path) == expected, f"Expected pinned source {path} (sha256 {expected})")
    return path


def extension_path(native):
    """Expected extension file matching this interpreter's exact ABI and machine."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("This Python has no extension ABI suffix")
    return Path(native)/"deepinfer"/"_so"/("nvidia-"+platform.machine())/("_C"+suffix)


def _copy_verified_extension(source, target):
    local = extension_path(target)
    suff = sysconfig.get_config_var("EXT_SUFFIX")
    candidates = (extension_path(source), source/"deepinfer"/("_C"+suff))
    for binary in candidates:
        if not binary.is_file():
            continue
        with binary.open("rb") as reader:
            _check(reader.read(4) == bytes.fromhex("7f454c46"), f"Expected a Linux ELF extension at {binary}")
        if local.exists():
            _check(_sha(binary) == _sha(local), f"Existing extension disagrees with {binary}")
            return
        local.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=local.parent, delete=False) as temp:
            temp_name = Path(temp.name)
        try:
            shutil.copy2(binary, temp_name)
            temp_name.replace(local)
        finally:
            temp_name.unlink(missing_ok=True)
        return


def prepare_sm90(repo: Path, target: Path) -> Path:
    """Create/reuse an isolated NativeSnapshot and return its path. Checks all sources byte-for-byte."""
    repo, target = Path(repo).resolve(), Path(target).resolve()
    external_root = os.environ.get("DEEPINFER_SM90_ROOT")
    external = Path(external_root).expanduser().resolve() if external_root else None
    _check(repo.is_dir() and repo != target and target not in repo.parents,
           "Specify a separate output directory and the existing RedKnot checkout")
    inventory = _inventory()
    target.parent.mkdir(parents=True,exist_ok=True)
    lock_path = target.parent / ("."+target.name+".sm90-prepare.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if external is not None:
            for name, rule in inventory.items():
                _checked(external, name, rule["sha256"])
            if external == target:
                return target
        target.mkdir(parents=True,exist_ok=True)
        for name, rule in inventory.items():
            dst = target / name
            expected = rule["sha256"]
            if dst.exists():
                _checked(target, name, expected)
                continue
            dst.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as temp:
                temp_name = Path(temp.name)
            try:
                if external is not None or rule["from"] != "patch":
                    if external is not None:
                        src = external / name
                    elif rule["from"] == "repo":
                        src = _checked(repo, name, expected)
                    elif rule["from"] == "preserved":
                        src = _checked(repo, rule["source"], expected)
                    else:
                        _check(rule["from"] == "new", f"Unknown source rule for {name}")
                        src = _checked(OVERLAY/"new", name, expected)
                    shutil.copyfile(src, temp_name)
                else:
                    src = _checked(repo, name, rule["base_sha256"])
                    patch = _checked(OVERLAY/"patches", name+".patch", rule["patch_sha256"])
                    subprocess.run(["patch","-f","-s","-o",str(temp_name),str(src),str(patch)],
                                   cwd=temp_name.parent,check=True)
                _check(_sha(temp_name) == expected, f"Hopper output checksum mismatch for {name}")
                temp_name.chmod(0o644)
                temp_name.replace(dst)
            finally:
                temp_name.unlink(missing_ok=True)
        # setuptools reads the checkout README as standard package metadata.
        if not (target/"README.md").exists():
            (target/"README.md").write_text("DeepInfer SM90 runtime source for the RedKnot benchmark.\n")
        if external is not None:
            _copy_verified_extension(external, target)
        elif platform.system() == "Linux":
            # This checkout's extension builds the exact C++ source inventory pinned above.
            _copy_verified_extension(repo, target)
        return target


def build_extension(native):
    """Build the CPU/CUDA binding against this environment; source artifacts contain no binaries."""
    if platform.system() != "Linux" or sys.version_info < (3,12):
        raise RuntimeError("The native build requires Linux and Python 3.12+")
    env = dict(os.environ, DEEPINFER_ACCELERATOR="nvidia")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    env.setdefault("MAX_JOBS", "8")
    out = native/"build"/"redknot-sm90"
    subprocess.run([sys.executable,"setup.py","build_ext","--build-lib",str(out/"lib"),
                    "--build-temp",str(out/"tmp")], cwd=native, env=env, check=True)
    dst = extension_path(native)
    built = out/"lib"/"deepinfer"/dst.name
    _check(built.is_file(), f"Compilation did not produce {built}")
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(built,dst)
    subprocess.run([sys.executable, "-c",
                    "import torch; from deepinfer import _C; print('DeepInfer native extension:', _C.__file__)"],
                   cwd=native, env=env, check=True)
    return dst


def build_vendors(native):
    if platform.system() != "Linux" or sys.version_info < (3,12):
        raise RuntimeError("Hopper vendors require Linux and Python 3.12+")
    env = dict(os.environ, DEEPINFER_ACCELERATOR="nvidia", DEEP_SELECT_BUILD_TARGET_PLATFORM="CUDA", DG_FORCE_BUILD="1")
    env.setdefault("TORCH_CUDA_ARCH_LIST","9.0")
    env.setdefault("MAX_JOBS", "8")
    root = native/"build"/"redknot-sm90-vendors"
    root.mkdir(parents=True, exist_ok=True)
    git = ["git","-c","url.https://github.com/.insteadOf=git@github.com:"]
    for name, (url, commit, module) in VENDORS.items():
        folder = root/name
        if not (folder/".git").is_dir():
            subprocess.run(git+["clone","--no-checkout",url,str(folder)],check=True)
            subprocess.run(git+["-C",str(folder),"checkout","--detach",commit],check=True)
        actual = subprocess.check_output(git+["-C",str(folder),"rev-parse","HEAD"],text=True).strip()
        _check(actual == commit, f"Expected {name} {commit}, found {actual}")
        subprocess.run(git+["-C",str(folder),"diff","--quiet","HEAD","--"],check=True)
        subprocess.run(git+["-C",str(folder),"submodule","update","--init","--recursive"],check=True)
        driver = native/"scripts"/"build_deepselect.py" if name == "DeepSelect" else folder/"setup.py"
        with tempfile.TemporaryDirectory(prefix="wheels-",dir=root) as temp:
            subprocess.run([sys.executable,str(driver),"bdist_wheel","--dist-dir",temp],cwd=folder,env=env,check=True)
            wheels = list(Path(temp).glob("*.whl"))
            _check(len(wheels) == 1, f"Expected one wheel from {name}")
            subprocess.run([sys.executable,"-m","pip","install","--no-deps","--force-reinstall",str(wheels[0])],check=True)
        subprocess.run([sys.executable,"-c",f"import {module}; print({module}.__file__)"],cwd=native,env=env,check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Root of the public RedKnot checkout")
    parser.add_argument("--target", required=True, type=Path, help="Isolated generated NativeSnapshot")
    parser.add_argument("--build-vendors", action="store_true", help="Fetch, build and install pinned public Hopper vendors")
    parser.add_argument("--build", action="store_true", help="Build the deepinfer._C binding against current PyTorch")
    args = parser.parse_args()
    try:
        native = prepare_sm90(args.repo, args.target)
        if args.build_vendors:
            build_vendors(native)
        if args.build:
            build_extension(native)
        print(f"SM90 source: {native}")
        ext = extension_path(native)
        print(f"Native binding: {ext}" if ext.is_file() else "Native binding not present for this interpreter.")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
