from optimizer_ssl.analysis.rank_aggregation import (
    aggregate_final_window_points,
    split_global_and_frequency_points,
)
from optimizer_ssl.analysis.scaling_fits import fit_power_law_with_ci, fit_rank_scaling_points


def _row(run_id, layer, step, bucket, soft, hard, width=768, optimizer="adamw"):
    return {
        "run_id": run_id,
        "layer": layer,
        "step": step,
        "bucket": bucket,
        "soft_rank_post": soft,
        "hard_rank_post": hard,
        "model_scale": "160m",
        "optimizer": optimizer,
        "optimizer_variant": optimizer,
        "width_multiplier": width // 768,
        "ffn_hidden_dim": width,
        "frequency_bucket_reduction": "rank0_local",
        "seed": "not_recorded",
        "paper_experiment": "unit_test",
    }


def test_aggregate_final_window_matches_layer_then_global_median():
    rows = []
    for layer in [0, 1]:
        for step in [1, 2, 3, 4, 5, 6]:
            rows.append(_row("r1", layer, step, "global", soft=10 * (layer + 1) + step, hard=step))
    points = aggregate_final_window_points(rows, final_samples=5)
    soft = next(p for p in points if p["metric"] == "soft_rank")
    # Layer medians over steps 2..6 are 14 and 24; median over layers is 19.
    assert soft["value"] == 19.0
    assert soft["n_layers"] == 2
    assert soft["aggregation"] == "median_final5_then_layer_median"


def test_split_global_and_frequency_points():
    points = [
        {"bucket": "global"},
        {"bucket": "head"},
        {"bucket": "tail"},
    ]
    global_points, bucket_points = split_global_and_frequency_points(points)
    assert len(global_points) == 1
    assert len(bucket_points) == 2


def test_fit_power_law_with_ci_positive_points():
    fit = fit_power_law_with_ci([1, 2, 4, 8], [2, 4, 8, 16], min_points=3)
    assert fit["valid"]
    assert abs(fit["beta"] - 1.0) < 1e-12
    assert fit["n_widths"] == 4
    assert fit["ci_method"] == "ols_loglog_t_interval"


def test_fit_rank_scaling_points_groups_by_metric_and_bucket():
    points = []
    for width in [768, 1536, 3072, 6144]:
        points.append(
            {
                "model_scale": "160m",
                "bucket": "global",
                "metric": "hard_rank",
                "optimizer": "adamw",
                "optimizer_variant": "adamw",
                "ffn_hidden_dim": width,
                "value": width / 768,
            }
        )
    fits = fit_rank_scaling_points(points, min_points=3)
    assert len(fits) == 1
    assert abs(fits[0]["beta"] - 1.0) < 1e-12
