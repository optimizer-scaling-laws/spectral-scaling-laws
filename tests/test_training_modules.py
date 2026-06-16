from argparse import Namespace

from optimizer_ssl.train import Hyperparameters
from optimizer_ssl.train.config import override_args_from_cli, validate_hyperparameters
from optimizer_ssl.train.experiment_logging import build_run_name, print0, set_master_process


def test_training_config_override_and_validation():
    hp = Hyperparameters()
    args = Namespace(lr=0.123, model_dim=256, unknown_metadata="ignored")
    hp = override_args_from_cli(hp, args)
    assert hp.lr == 0.123
    assert hp.model_dim == 256
    assert not hasattr(hp, "unknown_metadata")
    validate_hyperparameters(hp)


def test_build_run_name_includes_optimizer_details():
    hp = Hyperparameters(optimizer="dion", scalar_opt="lion", rank_fraction=0.0625)
    args = Namespace(
        dp_size=None,
        fs_size=None,
        tp_size=None,
        replicate_mesh_grad_sync=False,
        wandb_job_name="smoke",
    )
    run_name = build_run_name(hp, args)
    assert "dion" in run_name
    assert "frac=0.0625" in run_name
    assert run_name.endswith("_smoke")


def test_rank_zero_print_helper_can_be_toggled(capsys):
    set_master_process(False)
    print0("hidden")
    assert capsys.readouterr().out == ""
    set_master_process(True)
    print0("visible")
    assert "visible" in capsys.readouterr().out
