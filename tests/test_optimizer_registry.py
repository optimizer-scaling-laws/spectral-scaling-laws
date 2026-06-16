from optimizer_ssl.optimizers.registry import OPTIMIZER_NAMES, get_optimizer_class


def test_registry_names_declared():
    assert set(OPTIMIZER_NAMES) == {"adamw", "dion", "muon", "normuon"}


def test_registry_knows_adamw_without_vendor_dependencies():
    assert get_optimizer_class("adamw") is not None
