from pathlib import Path

import yaml

REMOVED_KEYS = {"enable_alpha_sweep", "alpha_sweep_values", "track_symmetry"}
VALID_OPTIMIZERS = {"adamw", "muon", "normuon", "dion"}
VALID_FREQUENCY_BUCKET_REDUCTIONS = {"rank0_local", "distributed_covariance"}
VALID_SPECTRAL_ERROR_POLICIES = {"warn", "raise", "nan"}


def load_yaml(path: Path):
    with path.open() as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"{path} did not parse to a mapping"
    return cfg


def test_public_paper_configs_load_and_have_core_fields():
    paths = sorted(Path("configs/paper_runs").rglob("*.yaml"))
    assert paths, "No paper-run configs found"
    for path in paths:
        cfg = load_yaml(path)
        for key in ["optimizer", "model_dim", "n_layer", "n_head", "ffn_mult"]:
            assert key in cfg, f"{path} missing {key}"
        assert REMOVED_KEYS.isdisjoint(cfg), f"{path} contains removed keys"


def test_paper_configs_have_human_metadata():
    for path in Path("configs/paper_runs").rglob("*.yaml"):
        cfg = load_yaml(path)
        for key in [
            "run_name",
            "paper_experiment",
            "model_scale",
            "width_multiplier",
            "optimizer_name",
            "optimizer_variant",
            "num_gpus_used",
        ]:
            assert key in cfg, f"{path} missing metadata field {key}"
        assert cfg["optimizer_name"] in VALID_OPTIMIZERS
        assert cfg["width_multiplier"] == cfg["ffn_mult"]
        assert cfg.get("seed") == 1337
        assert cfg.get("spectral_error_policy") in VALID_SPECTRAL_ERROR_POLICIES


def test_frequency_bucket_configs_point_to_released_table_and_paper_reduction_mode():
    freq_path = Path("results/processed/token_frequencies.npy")
    assert freq_path.exists()
    for path in Path("configs/paper_runs").rglob("*.yaml"):
        cfg = load_yaml(path)
        if cfg.get("track_by_frequency", False):
            assert cfg.get("token_freq_file") == str(freq_path), path
            assert cfg.get("frequency_bucket_reduction") == "rank0_local", path
            assert cfg.get("frequency_bucket_reduction") in VALID_FREQUENCY_BUCKET_REDUCTIONS


def test_gpu_metadata_matches_experiment_family():
    for path in Path("configs/paper_runs/main_160m_width_sweep").rglob("*.yaml"):
        assert load_yaml(path)["num_gpus_used"] == 4
    for path in Path("configs/paper_runs/main_350m_width_sweep").rglob("*.yaml"):
        assert load_yaml(path)["num_gpus_used"] == 8
    for path in Path("configs/paper_runs/dion_rank_sweep/160m").rglob("*.yaml"):
        assert load_yaml(path)["num_gpus_used"] == 4


def test_example_configs_load():
    for path in Path("configs/examples").rglob("*.yaml"):
        cfg = load_yaml(path)
        assert REMOVED_KEYS.isdisjoint(cfg), f"{path} contains removed keys"
        assert cfg.get("paper_experiment") == "tiny_debug"
