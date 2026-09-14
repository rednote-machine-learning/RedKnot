"""vLLM's general-plugin entry point. Loading alone does not enable RedKnot."""


# REDKNOT: RK-PLUGIN — opt-in registration; native engines remain external.
def register() -> None:
    from .config import load_config

    config = load_config()
    if not config or not config.get("enabled", False):
        return
    from .compat import verify_vllm_sources
    from .runtime import RedKnotSettings

    family = config.get("engine_family", "mha")
    verify_vllm_sources(engine_family=family)
    settings = RedKnotSettings.from_mapping(config)

    if family == "deepseek_v4_flash":
        from .dsv4_backend import install_dsv4_attention
        from .dsv4_runner import install_dsv4_runner

        install_dsv4_runner(settings)
        install_dsv4_attention()
        return

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    from .runner import install_runner_hooks

    register_backend(
        AttentionBackendEnum.CUSTOM, "vllm_redknot.vllm_backend.RedKnotBackend"
    )
    install_runner_hooks(settings)
