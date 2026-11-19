"""Bind the bundled S1.6 code, SM90 DeepInfer sources and a local checkpoint."""
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile

if __package__:
    from .sm90_source import prepare_sm90
else:
    from sm90_source import prepare_sm90


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SERVICE_PROFILE_SHA = "4ce249c4bbebaaeaca8b04902d7ee70ab87cd909d9ea8666be4acc2ce459c58e"


NATIVE_INVENTORY_SHA = "ff8810322b5ed35c6227b66f45f9b23a5343707acb8b0c8fa8c5195003aa1949"


def load_native_assets(model_path, output):
    """Prepare a run-local source tree without importing Torch or initializing CUDA."""
    model = Path(model_path).expanduser().resolve(strict=True)
    native = Path(output).resolve() / "runtime"
    config_bytes = (HERE / "service_profile.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != SERVICE_PROFILE_SHA:
        raise ValueError("Frozen S1.6 service profile differs from the original settings")
    profile = json.loads(config_bytes)
    for key, value in profile.pop("prefill_environment").items():
        os.environ[key] = value
    import_paths = {}
    index_bytes = (HERE / "native_sources.json").read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != NATIVE_INVENTORY_SHA:
        raise ValueError("Changed S1.6 native dependency inventory")
    inventory = json.loads(index_bytes)
    for relative, rule in inventory.items():
        source_root = REPO if rule["origin"] == "checkout" else HERE / "native"
        source = source_root / relative
        dest = native / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != rule["sha256"]:
            raise ValueError(f"Bundled source checksum mismatch for {relative}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            copyfile(source, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != rule["sha256"]:
            raise ValueError(f"Changed file in output directory: {dest}")
        import_paths[str(dest)] = rule["sha256"]
    snapshot = prepare_sm90(REPO, native / "NativeSnapshot")
    profile.update(experiment_root=str(native / "dependencies"), native_repo=str(snapshot),
                   gated_repo=str(native / "GatedEngram"), stage=str(native), model_path=str(model),
                   tokenizer_path=str(model / "tokenizer.json"), encoder_path=str(model / "encoding/encoding.py"),
                   session_id="longbench-full-s16", source_sha256=import_paths)
    return native, profile
