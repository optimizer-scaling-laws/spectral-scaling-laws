import math

from optimizer_ssl.analysis.log_schema import parse_layer_metric_line


def test_legacy_layer_metric_line_normalizes_to_public_schema():
    line = "Step 200: SE_pre=5.109, SE_post=6.555, PR_pre=79.89, PR_post=254.98, EEE_pre=0.955, EEE_post=0.786, JS=0.0996"
    row = parse_layer_metric_line(line)
    assert row["step"] == 200
    assert row["spectral_entropy_pre"] == 5.109
    assert row["hard_rank_post"] == 254.98
    assert math.isclose(row["soft_rank_pre"], math.exp(5.109))
    assert "EEE_pre" not in row
    assert "JS" not in row


def test_current_layer_metric_line_is_preserved():
    line = "Step 200: soft_rank_pre=165.0, soft_rank_post=703.5, hard_rank_pre=79.89, hard_rank_post=254.98, spectral_entropy_pre=5.109, spectral_entropy_post=6.555"
    row = parse_layer_metric_line(line)
    assert row["soft_rank_pre"] == 165.0
    assert row["hard_rank_pre"] == 79.89
