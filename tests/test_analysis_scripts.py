import csv
import subprocess
import sys
from pathlib import Path


def test_parse_and_aggregate_scripts_on_sample_logs(tmp_path):
    log_dir = tmp_path / "adamw" / "1d" / "eigen_metrics_logs"
    freq_dir = log_dir / "frequency_tertiles"
    freq_dir.mkdir(parents=True)
    (log_dir / "layer_0_eigen.txt").write_text(
        Path("results/sample_logs/adamw_160m_1x_layer0_global_legacy.txt").read_text()
    )
    (freq_dir / "layer_0_eigen_freq.txt").write_text(
        Path("results/sample_logs/adamw_160m_1x_layer0_frequency_legacy.txt").read_text()
    )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "model_scale",
                "model_dim",
                "n_layer",
                "base_ffn_dim",
                "width_multiplier",
                "ffn_hidden_dim",
                "optimizer",
                "optimizer_variant",
                "dion_rank_fraction",
                "num_gpus",
                "seed",
                "frequency_bucket_reduction",
                "config_path",
                "log_dir",
                "paper_experiment",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "adamw_160m_1x",
                "model_scale": "160m",
                "model_dim": "768",
                "n_layer": "1",
                "base_ffn_dim": "768",
                "width_multiplier": "1",
                "ffn_hidden_dim": "768",
                "optimizer": "adamw",
                "optimizer_variant": "adamw",
                "dion_rank_fraction": "",
                "num_gpus": "4",
                "seed": "not_recorded",
                "frequency_bucket_reduction": "rank0_local",
                "config_path": "configs/paper_runs/main_160m_width_sweep/adamw/1x.yaml",
                "log_dir": str(log_dir),
                "paper_experiment": "unit_test",
            }
        )

    layer_metrics = tmp_path / "layer_metrics.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/parse_eigen_logs_to_csv.py",
            "--manifest",
            str(manifest),
            "--output",
            str(layer_metrics),
            "--require-all-buckets-per-step",
        ],
        check=True,
    )
    assert layer_metrics.exists()
    rows = list(csv.DictReader(layer_metrics.open()))
    assert {row["bucket"] for row in rows} == {"global", "head", "mid", "tail"}

    out_dir = tmp_path / "processed"
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/aggregate_rank_scaling.py",
            "--layer-metrics",
            str(layer_metrics),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )
    assert (out_dir / "global_rank_scaling_points.csv").exists()
    assert (out_dir / "frequency_bucket_rank_scaling_points.csv").exists()
