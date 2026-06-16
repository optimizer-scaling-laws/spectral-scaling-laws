from pathlib import Path

from optimizer_ssl.analysis.eigen_log_parser import (
    parse_frequency_layer_file,
    parse_global_layer_file,
    parse_run_log_dir,
)


def test_parse_global_legacy_sample():
    path = Path("results/sample_logs/adamw_160m_1x_layer0_global_legacy.txt")
    rows = parse_global_layer_file(path, layer=0, run_id="sample")
    assert len(rows) > 0
    first = rows[0]
    assert first["run_id"] == "sample"
    assert first["bucket"] == "global"
    assert first["layer"] == 0
    assert first["step"] == 1
    assert "spectral_entropy_post" in first
    assert "soft_rank_post" in first
    assert "hard_rank_post" in first
    assert "EEE_post" not in first
    assert "JS" not in first


def test_parse_frequency_legacy_sample():
    path = Path("results/sample_logs/adamw_160m_1x_layer0_frequency_legacy.txt")
    rows = parse_frequency_layer_file(path, layer=0, run_id="sample", require_all_buckets_per_step=True)
    assert len(rows) > 0
    buckets = {row["bucket"] for row in rows if row["step"] == 1}
    assert buckets == {"head", "mid", "tail"}
    head = next(row for row in rows if row["step"] == 1 and row["bucket"] == "head")
    assert head["n_tokens"] == 44571
    assert head["hard_rank_post"] == 10.53
    assert head["soft_rank_post"] > 0


def test_parse_run_log_dir_sample(tmp_path):
    # Build the minimal expected log-dir layout from the two sample files.
    log_dir = tmp_path / "eigen_metrics_logs"
    freq_dir = log_dir / "frequency_tertiles"
    freq_dir.mkdir(parents=True)
    global_src = Path("results/sample_logs/adamw_160m_1x_layer0_global_legacy.txt")
    freq_src = Path("results/sample_logs/adamw_160m_1x_layer0_frequency_legacy.txt")
    (log_dir / "layer_0_eigen.txt").write_text(global_src.read_text())
    (freq_dir / "layer_0_eigen_freq.txt").write_text(freq_src.read_text())

    rows = parse_run_log_dir(log_dir, run_id="sample", num_layers=1, require_all_buckets_per_step=True)
    assert {row["bucket"] for row in rows} == {"global", "head", "mid", "tail"}
