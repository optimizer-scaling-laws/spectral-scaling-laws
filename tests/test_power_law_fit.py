from optimizer_ssl.scaling.fit_power_law import fit_power_law


def test_power_law_fit_recovers_beta():
    x = [1, 2, 4, 8]
    y = [3 * (v ** 2) for v in x]
    out = fit_power_law(x, y)
    assert abs(out["beta"] - 2.0) < 1e-8
