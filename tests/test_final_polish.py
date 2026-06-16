from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tomllib
import yaml


def _read_text_files(root: Path):
    for base in [root / "README.md", root / "docs", root / "scripts", root / "configs", root / "optimizer_ssl"]:
        if base.is_file():
            yield base, base.read_text(errors="ignore")
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml", ".sh", ".toml"}:
                yield path, path.read_text(errors="ignore")


def test_public_docs_do_not_leak_internal_analysis_paths():
    root = Path(__file__).resolve().parents[1]
    banned = ["logs_inductive_bias", "plots_scaling_laws"]
    offenders = []
    for path, text in _read_text_files(root):
        if path.name == "test_final_polish.py":
            continue
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(root)} contains {token}")
    assert not offenders, "\n".join(offenders)


def test_project_distribution_and_repo_slug_are_aligned():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert data["project"]["name"] == "spectral-scaling-laws"
    assert data["project"]["urls"]["Repository"].endswith("/spectral-scaling-laws")


def test_adamw_component_and_tiny_debug_use_adamw_scalar_optimizer():
    for path in [Path("configs/components/optimizers/adamw.yaml"), Path("configs/examples/tiny_debug.yaml")]:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["optimizer"] == "adamw"
        assert cfg["scalar_opt"] == "adamw"


def test_top_level_import_does_not_eagerly_import_torch():
    cmd = [
        sys.executable,
        "-c",
        "import sys; import optimizer_ssl; print('torch' in sys.modules)",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    assert out == "False"


def test_readme_lists_released_token_frequency_artifacts_once():
    text = Path("README.md").read_text()
    block = "results/processed/token_frequencies.npy\nresults/processed/token_frequencies.pt\nresults/processed/token_frequency_stats.json"
    assert block in text
    assert "token_frequences" not in text
    assert text.count("results/processed/token_frequencies.npy") >= 2
