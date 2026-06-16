import torch


def get_optimizer_class(name: str):
    """Return optimizer class by public name. Vendor imports are lazy."""
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW
    if key == "dion":
        from dion import Dion
        return Dion
    if key == "muon":
        from dion import Muon
        return Muon
    if key == "normuon":
        from dion import NorMuon
        return NorMuon
    raise KeyError(f"Unknown optimizer: {name}")


OPTIMIZER_NAMES = ("adamw", "dion", "muon", "normuon")
