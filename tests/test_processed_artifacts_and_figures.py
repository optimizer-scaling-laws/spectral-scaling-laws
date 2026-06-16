from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_processed_rank_scaling_artifacts_exist_and_are_sanitized():
    root = Path(__file__).resolve().parents[1]
    processed = root / "results" / "processed"
    required = [
        "run_metadata.csv",
        "global_rank_scaling_points.csv",
        "frequency_bucket_rank_scaling_points.csv",
        "main_beta_table.csv",
        "frequency_bucket_beta_table.csv",
        "dion_rank_sweep_run_metadata.csv",
        "dion_tail_rank_sweep_points.csv",
        "dion_tail_rank_sweep_beta_table.csv",
        "dion_rank_sweep_summary.json",
        "analysis_summary.json",
        "matched_loss_run_metadata.csv",
        "matched_loss_terminal_beta_table.csv",
        "matched_loss_beta_dynamics.csv",
        "matched_loss_pr_trajectories.csv",
        "matched_loss_summary.json",
        "tail_350m_run_metadata.csv",
        "tail_350m_rank_scaling_points.csv",
        "tail_350m_beta_table.csv",
        "tail_350m_summary.json",
        "architecture_vs_optimizer_beta_values.csv",
        "architecture_vs_optimizer_comparison.csv",
    ]
    for name in required:
        path = processed / name
        assert path.exists(), name
        text = path.read_text(errors="ignore")
        assert "logs_inductive_bias" not in text
        assert "/home/" not in text
        assert "nj2049" not in text
    assert not (processed / "token_frequencies.pt").exists()
    assert (processed / "token_frequencies.npy").exists()


def test_figure_manifest_is_complete_and_pdf_only():
    manifest_path = Path("results/figure_manifest.csv")
    assert manifest_path.exists()
    rows = list(csv.DictReader(manifest_path.open()))
    expected_figures = {
        "global_hard_rank_scaling.pdf",
        "global_soft_rank_scaling.pdf",
        "frequency_bucket_rank_grid.pdf",
        "dion_tail_hard_rank_sweep.pdf",
        "dion_tail_soft_rank_sweep.pdf",
        "matched_loss_scaling_breakdown.pdf",
        "matched_loss_beta_dynamics.pdf",
        "matched_loss_pr_trajectories_by_width.pdf",
        "tail_350m_hard_rank_scaling.pdf",
        "tail_350m_soft_rank_scaling.pdf",
        "architecture_vs_optimizer.pdf",
    }
    assert {row["figure_file"] for row in rows} == expected_figures
    for row in rows:
        assert row["figure_file"].endswith(".pdf")
        assert (Path("results/figures") / row["figure_file"]).exists()
        for input_name in row["processed_inputs"].split(";"):
            assert (Path("results/processed") / input_name).exists(), input_name
        assert row["reproduction_command"].startswith("bash scripts/reproduce/")
        assert row["raw_log_reproduction_status"] in {
            "raw_logs_supported_for_main_manifest",
            "processed_csv_only_unless_external_logs_available",
            "processed_beta_only",
            "processed_beta_only_unless_external_logs_available",
        }


def test_no_png_figures_or_empty_placeholder_assets():
    assert not list(Path("results/figures").glob("*.png"))
    assert not Path("assets").exists()


def test_headline_notebook_is_polished_and_executed():
    notebook = Path("notebooks/reproduce_main_figures.ipynb")
    assert notebook.exists()
    assert (notebook.parent / "README.md").exists()
    text = notebook.read_text(errors="ignore")
    assert "Open in Colab" in text
    assert "global_rank_scaling_points.csv" in text
    assert "frequency_bucket_rank_scaling_points.csv" in text
    nb = json.loads(text)
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    # At least one table/print output and one plotted image should be committed for GitHub preview.
    outputs = [out for cell in code_cells for out in cell.get("outputs", [])]
    assert any(out.get("output_type") in {"stream", "execute_result", "display_data"} for out in outputs)
    assert any("image/png" in out.get("data", {}) for out in outputs)


def test_beta_tables_have_expected_rows():
    main_rows = list(csv.DictReader(Path("results/processed/main_beta_table.csv").open()))
    freq_rows = list(csv.DictReader(Path("results/processed/frequency_bucket_beta_table.csv").open()))
    dion_rows = list(csv.DictReader(Path("results/processed/dion_tail_rank_sweep_beta_table.csv").open()))
    matched_rows = list(csv.DictReader(Path("results/processed/matched_loss_terminal_beta_table.csv").open()))
    tail_350m_rows = list(csv.DictReader(Path("results/processed/tail_350m_beta_table.csv").open()))
    arch_beta_rows = list(csv.DictReader(Path("results/processed/architecture_vs_optimizer_beta_values.csv").open()))
    arch_comparison_rows = list(csv.DictReader(Path("results/processed/architecture_vs_optimizer_comparison.csv").open()))
    assert len(main_rows) == 10
    assert len(freq_rows) == 30
    assert len(dion_rows) == 10
    assert len(matched_rows) == 18
    assert len(tail_350m_rows) == 8
    assert len(arch_beta_rows) == 60
    assert len(arch_comparison_rows) == 30
    assert {row["metric"] for row in main_rows} == {"soft_rank", "hard_rank"}
    assert {row["bucket"] for row in freq_rows} == {"head", "mid", "tail"}
    assert {row["bucket"] for row in dion_rows} == {"tail"}
    assert {row["optimizer_folder"] for row in dion_rows} == {
        "adamw",
        "dion_r1_2",
        "dion_r1_4",
        "dion_r1_8",
        "dion_r1_16",
    }
    assert {row["optimizer_folder"] for row in matched_rows} == {"adamw_6k", "adamw_12k", "dion_r1_16"}
    assert {row["bucket"] for row in matched_rows} == {"head", "mid", "tail"}
    assert {row["bucket"] for row in tail_350m_rows} == {"tail"}
    assert {row["model_scale"] for row in tail_350m_rows} == {"350m"}
    assert {row["optimizer_folder"] for row in tail_350m_rows} == {"adamw", "muon", "normuon", "dion_r1_16"}
    assert {row["head_count"] for row in arch_beta_rows} == {"6", "12"}
    assert {row["bucket"] for row in arch_beta_rows} == {"head", "mid", "tail"}
    assert {row["metric"] for row in arch_beta_rows} == {"soft_rank", "hard_rank"}
    assert {row["metric"] for row in arch_comparison_rows} == {"soft_rank", "hard_rank"}


def test_figure_scripts_generate_outputs_from_processed_csvs(tmp_path):
    out_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_global_rank_figures.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_frequency_bucket_figures.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_dion_rank_sweep_figures.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    assert (out_dir / "global_hard_rank_scaling.pdf").exists()
    assert (out_dir / "global_soft_rank_scaling.pdf").exists()
    assert (out_dir / "frequency_bucket_rank_grid.pdf").exists()
    assert (out_dir / "dion_tail_hard_rank_sweep.pdf").exists()
    assert (out_dir / "dion_tail_soft_rank_sweep.pdf").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_matched_loss_figures.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    assert (out_dir / "matched_loss_scaling_breakdown.pdf").exists()
    assert (out_dir / "matched_loss_beta_dynamics.pdf").exists()
    assert (out_dir / "matched_loss_pr_trajectories_by_width.pdf").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_350m_tail_figures.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    assert (out_dir / "tail_350m_hard_rank_scaling.pdf").exists()
    assert (out_dir / "tail_350m_soft_rank_scaling.pdf").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/make_architecture_vs_optimizer_figure.py",
            "--out-dir",
            str(out_dir),
            "--formats",
            "pdf",
        ],
        check=True,
    )
    assert (out_dir / "architecture_vs_optimizer.pdf").exists()
