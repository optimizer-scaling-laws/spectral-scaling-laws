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
    for path in Path("configs/paper_runs/matched_loss/160m").rglob("*.yaml"):
        assert load_yaml(path)["num_gpus_used"] == 4
    for path in Path("configs/paper_runs/architecture_vs_optimizer/160m").rglob("*.yaml"):
        assert load_yaml(path)["num_gpus_used"] == 4


def test_main_160m_full_config_grid():
    root = Path("configs/paper_runs/main_160m_width_sweep")
    groups = {
        "adamw": {"optimizer": "adamw", "variant": "adamw"},
        "muon": {"optimizer": "muon", "variant": "muon"},
        "normuon": {"optimizer": "normuon", "variant": "normuon"},
        "dion_r1_2": {"optimizer": "dion", "variant": "dion_r1_2", "rank_fraction": 0.5},
        "dion_r1_16": {"optimizer": "dion", "variant": "dion_r1_16", "rank_fraction": 0.0625},
    }
    for group, expected in groups.items():
        for width in range(1, 9):
            path = root / group / f"{width}x.yaml"
            assert path.exists(), f"missing 160M config: {path}"
            cfg = load_yaml(path)
            assert cfg["paper_experiment"] == "main_160m_width_sweep"
            assert cfg["model_scale"] == "160m"
            assert cfg["model_dim"] == 768
            assert cfg["n_layer"] == 12
            assert cfg["n_head"] == 12
            assert cfg["width_multiplier"] == width
            assert cfg["ffn_mult"] == width
            assert cfg["num_gpus_used"] == 4
            assert cfg["num_iterations"] == 6000
            assert cfg["eigen_log_steps"] == 200
            assert cfg["frequency_bucket_reduction"] == "rank0_local"
            assert cfg["optimizer"] == expected["optimizer"]
            assert cfg["optimizer_variant"] == expected["variant"]
            if "rank_fraction" in expected:
                assert cfg["rank_fraction"] == expected["rank_fraction"]


def test_example_configs_load():
    for path in Path("configs/examples").rglob("*.yaml"):
        cfg = load_yaml(path)
        assert REMOVED_KEYS.isdisjoint(cfg), f"{path} contains removed keys"
        assert cfg.get("paper_experiment") == "tiny_debug"


def test_dion_rank_sweep_full_config_grid():
    root = Path("configs/paper_runs/dion_rank_sweep/160m")
    groups = {
        "adamw": None,
        "dion_r1_2": 0.5,
        "dion_r1_4": 0.25,
        "dion_r1_8": 0.125,
        "dion_r1_16": 0.0625,
    }
    for group, rank_fraction in groups.items():
        for width in range(1, 9):
            path = root / group / f"{width}x.yaml"
            assert path.exists(), f"missing Dion rank-sweep config: {path}"
            cfg = load_yaml(path)
            assert cfg["paper_experiment"] == "dion_rank_sweep"
            assert cfg["model_scale"] == "160m"
            assert cfg["width_multiplier"] == width
            assert cfg["ffn_mult"] == width
            assert cfg["num_gpus_used"] == 4
            assert cfg["num_iterations"] == 6000
            assert cfg["eigen_log_steps"] == 200
            assert cfg["frequency_bucket_reduction"] == "rank0_local"
            if group == "adamw":
                assert cfg["optimizer"] == "adamw"
                assert cfg["optimizer_name"] == "adamw"
            else:
                assert cfg["optimizer"] == "dion"
                assert cfg["optimizer_name"] == "dion"
                assert cfg["optimizer_variant"] == group
                assert cfg["rank_fraction"] == rank_fraction


def test_matched_loss_full_config_grid():
    root = Path("configs/paper_runs/matched_loss/160m")
    groups = {
        "adamw_6k": {"optimizer": "adamw", "variant": "adamw_6k", "steps": 6000},
        "adamw_12k": {"optimizer": "adamw", "variant": "adamw_12k", "steps": 12000},
        "dion_r1_16": {"optimizer": "dion", "variant": "dion_r1_16", "steps": 6000},
    }
    for group, expected in groups.items():
        for width in range(1, 9):
            path = root / group / f"{width}x.yaml"
            assert path.exists(), f"missing matched-loss config: {path}"
            cfg = load_yaml(path)
            assert cfg["paper_experiment"] == "matched_loss_scaling_break"
            assert cfg["model_scale"] == "160m"
            assert cfg["width_multiplier"] == width
            assert cfg["ffn_mult"] == width
            assert cfg["num_gpus_used"] == 4
            assert cfg["num_iterations"] == expected["steps"]
            assert cfg["eigen_log_steps"] == 200
            assert cfg["frequency_bucket_reduction"] == "rank0_local"
            assert cfg["optimizer"] == expected["optimizer"]
            assert cfg["optimizer_name"] == expected["optimizer"]
            assert cfg["optimizer_variant"] == expected["variant"]
            if group == "dion_r1_16":
                assert cfg["rank_fraction"] == 0.0625
            else:
                assert "rank_fraction" not in cfg


def test_main_350m_full_config_grid():
    root = Path("configs/paper_runs/main_350m_width_sweep")
    groups = {
        "adamw": {"optimizer": "adamw", "variant": "adamw"},
        "muon": {"optimizer": "muon", "variant": "muon"},
        "normuon": {"optimizer": "normuon", "variant": "normuon"},
        "dion_r1_16": {"optimizer": "dion", "variant": "dion_r1_16"},
    }
    for group, expected in groups.items():
        for width in range(1, 5):
            path = root / group / f"{width}x.yaml"
            assert path.exists(), f"missing 350M config: {path}"
            cfg = load_yaml(path)
            assert cfg["paper_experiment"] == "main_350m_width_sweep"
            assert cfg["model_scale"] == "350m"
            assert cfg["model_dim"] == 1024
            assert cfg["n_layer"] == 24
            assert cfg["n_head"] == 32
            assert cfg["width_multiplier"] == width
            assert cfg["ffn_mult"] == width
            assert cfg["num_gpus_used"] == 8
            assert cfg["num_iterations"] == 8000
            assert cfg["eigen_log_steps"] == 400
            assert cfg["frequency_bucket_reduction"] == "rank0_local"
            assert cfg["optimizer"] == expected["optimizer"]
            assert cfg["optimizer_variant"] == expected["variant"]
            if group == "dion_r1_16":
                assert cfg["rank_fraction"] == 0.0625


def test_architecture_vs_optimizer_full_config_grid():
    root = Path("configs/paper_runs/architecture_vs_optimizer/160m")
    groups = {
        "adamw": {"optimizer": "adamw", "variant": "adamw"},
        "muon": {"optimizer": "muon", "variant": "muon"},
        "normuon": {"optimizer": "normuon", "variant": "normuon"},
        "dion_r1_2": {"optimizer": "dion", "variant": "dion_r1_2", "rank_fraction": 0.5},
        "dion_r1_16": {"optimizer": "dion", "variant": "dion_r1_16", "rank_fraction": 0.0625},
    }
    for head_group, n_head in {"heads_12": 12, "heads_6": 6}.items():
        for group, expected in groups.items():
            for width in range(1, 9):
                path = root / head_group / group / f"{width}x.yaml"
                assert path.exists(), f"missing architecture-vs-optimizer config: {path}"
                cfg = load_yaml(path)
                assert cfg["paper_experiment"] == "architecture_vs_optimizer"
                assert cfg["model_scale"] == "160m"
                assert cfg["model_dim"] == 768
                assert cfg["n_layer"] == 12
                assert cfg["n_head"] == n_head
                assert cfg["width_multiplier"] == width
                assert cfg["ffn_mult"] == width
                assert cfg["num_gpus_used"] == 4
                assert cfg["num_iterations"] == 6000
                assert cfg["eigen_log_steps"] == 200
                assert cfg["frequency_bucket_reduction"] == "rank0_local"
                assert cfg["optimizer"] == expected["optimizer"]
                assert cfg["optimizer_name"] == expected["optimizer"]
                assert cfg["optimizer_variant"] == expected["variant"]
                if "rank_fraction" in expected:
                    assert cfg["rank_fraction"] == expected["rank_fraction"]
                else:
                    assert "rank_fraction" not in cfg
