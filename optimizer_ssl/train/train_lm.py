import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from optimizer_ssl.models.gpt_model import GPT, GPTConfig, parallelize_gpt_model
from optimizer_ssl.models.gpt_utils import DistributedDataLoader
from optimizer_ssl.spectra import GPTEigenMetricsTracker
from optimizer_ssl.train.checkpointing import CheckpointManager
from optimizer_ssl.train.config import (
    Hyperparameters,
    override_args_from_cli,
    parse_cli_args,
    validate_hyperparameters,
)
from optimizer_ssl.train.distributed import cleanup_distributed, init_distributed
from optimizer_ssl.train.evaluation import estimate_validation_loss
from optimizer_ssl.train.experiment_logging import (
    build_run_name,
    is_master_process,
    log_train_if_enabled,
    log_validation_if_enabled,
    print0,
    setup_wandb_if_enabled,
)
from optimizer_ssl.train.optimizer_factory import init_optimizer
from optimizer_ssl.utils.seed import seed_everything


def main():
    torch._dynamo.config.cache_size_limit = 100
    # --- Parse command line arguments and set hyperparams ---
    cli_args = parse_cli_args()
    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args, printer=print0)

    if cli_args.inv_rank_fraction:
        hp.rank_fraction = 1.0 / cli_args.inv_rank_fraction

    validate_hyperparameters(hp)
    seed_everything(hp.seed)
    print0(f"Random seed: {hp.seed}")

    # --- Distributed training initialization ---
    device_mesh = init_distributed(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )
    print0("=" * 80)

    # --- DataLoader Setup ---
    if device_mesh is not None:
        # Combine replicated and sharded data parallel meshes for data loading
        data_parallel_mesh = device_mesh["dp", "fs"]._flatten()
        data_parallel_size = data_parallel_mesh.size()
        data_parallel_rank = data_parallel_mesh.get_local_rank()
    else:
        # We are using DDP with one global process group
        data_parallel_mesh = None
        data_parallel_size = dist.get_world_size()
        data_parallel_rank = dist.get_rank()

    if cli_args.debug:
        # in debug mode, make batch size very small
        hp.batch_size = 2 * data_parallel_size
        hp.device_batch_size = 1

    # Calculate validation steps
    tokens_in_global_batch = (
        hp.device_batch_size * hp.sequence_length * data_parallel_size
    )
    assert hp.val_tokens % tokens_in_global_batch == 0, "Invalid val_tokens"
    val_steps = hp.val_tokens // tokens_in_global_batch

    if cli_args.debug:
        # train for just a few steps
        hp.num_iterations = 20
        val_steps = min(val_steps, 2)

    # Calculate gradient accumulation steps
    sequences_in_global_batch = hp.device_batch_size * data_parallel_size
    assert hp.batch_size % sequences_in_global_batch == 0, "Invalid batch_size"
    grad_accum_steps = hp.batch_size // sequences_in_global_batch
    assert grad_accum_steps >= 1, "Invalid grad_accum_steps"

    print0(f"Global batch size: {hp.batch_size} sequences")
    print0(f"Per-device batch size: {hp.device_batch_size} sequences")
    print0(f"Sequence length: {hp.sequence_length} tokens")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0("=" * 80)

    train_glob = os.path.join(hp.data_dir, "fineweb_train_*.bin")
    val_glob = os.path.join(hp.data_dir, "fineweb_val_*.bin")

    print0(f"Training data: {train_glob}")
    print0(f"Validation data: {val_glob}")

    # Each data parallel rank gets different data
    # TP ranks must all use identical data
    train_loader = DistributedDataLoader(
        train_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )
    val_loader = DistributedDataLoader(
        val_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )

    print0(f"Training DataLoader: {len(train_loader.files)} files")
    print0(f"Validation DataLoader: {len(val_loader.files)} files")
    print0("=" * 80)

    # --- Model Initialization ---
    print0(f"Model dimension: {hp.model_dim}")
    print0(f"Number of layers: {hp.n_layer}")
    print0(f"Number of heads: {hp.n_head}")

    num_vocab = 50304  # nearest multiple of 128 for efficiency
    gpt_config = GPTConfig(
        sequence_len=hp.sequence_length,
        vocab_size=num_vocab,
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        n_embd=hp.model_dim,
        ffn_mult=hp.ffn_mult,
        postln_frac=hp.postln_frac,
    )
    with torch.device("meta"):
        model = GPT(gpt_config)

    # Shard the model if using a device mesh
    # If replicate_mesh_grad_sync is True, FSDP will not handle data-parallel gradient sync
    # If replicate_mesh_grad_sync is False, we use Pytorch HSDP to do data-parallel gradient sync
    if device_mesh is not None:
        parallelize_gpt_model(
            model,
            device_mesh=device_mesh,
            dp_name=(None if hp.replicate_mesh_grad_sync else "dp"),
            fs_name="fs",
            tp_name="tp",
            fsdp_reshard_after_forward=(not cli_args.fast_fsdp),
        )
        raw_model = model

    # Move model to GPU
    model.to_empty(device="cuda")
    model.init_weights()
    if not cli_args.no_compile:
        model.compile()

    # If no device mesh, we are using DDP
    if device_mesh is None:
        # Use LOCAL_RANK here (per-node GPU index)
        # This ensures each process is pinned to the correct local GPU
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module  # the underlying model

    # Ensure parameters are contiguous
    for i, p in enumerate(model.parameters()):
        if not p.is_contiguous():
            raise ValueError(f"Parameter {i} is not contiguous")

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Total parameters: {num_params}")
    print0(f"Using torch.compile: {not cli_args.no_compile}")

    # Print model architecture
    print0(model)
    print0("=" * 80)

    # --- Initialize Eigen Metrics Tracker ---
    eigen_metrics_tracker = None
    
    if hp.enable_eigen_metrics:
        # Determine eigen metrics output directory
        if hp.eigen_metrics_dir:
            eigen_output_dir = hp.eigen_metrics_dir
        else:
            if hp.checkpoint_dir:
                eigen_output_dir = os.path.join(hp.checkpoint_dir, "eigen_metrics_logs")
            else:
                eigen_output_dir = "eigen_metrics_logs"
        
        # Ensure the directory exists (only on rank 0)
        if is_master_process():
            os.makedirs(eigen_output_dir, exist_ok=True)
            print0(f"Eigenvalue metrics will be saved to: {eigen_output_dir}")
        
        # Get rank and world size
        global_rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Check frequency tracking requirements
        track_by_frequency = hp.track_by_frequency
        token_freq_file = hp.token_freq_file
        frequency_bucket_reduction = hp.frequency_bucket_reduction
        
        if track_by_frequency and token_freq_file is None:
            raise ValueError("track_by_frequency=True requires token_freq_file to be specified")
        
        if track_by_frequency and is_master_process():
            print0(f"Frequency-bucketed metrics enabled with: {token_freq_file}")
            print0(f"Frequency bucket reduction: {frequency_bucket_reduction}")
        
        
        # Initialize eigen metrics tracker
        eigen_metrics_tracker = GPTEigenMetricsTracker(
            model=raw_model,
            log_steps=hp.eigen_log_steps,
            device='cuda',
            output_dir=eigen_output_dir,
            num_layers=hp.n_layer,
            global_rank=global_rank,
            world_size=world_size,
            console_log_level="ERROR",
            gather_statistics=True,
            track_by_frequency=track_by_frequency,
            token_freq_file=token_freq_file,
            frequency_bucket_reduction=frequency_bucket_reduction,
            error_policy=hp.spectral_error_policy,
        )
        
        print0("Eigen metrics tracker initialized")

    # --- Optimizer Setup ---
    print0(f"Optimizer: {hp.optimizer}")
    print0(f"Scalar optimizer: {hp.scalar_opt}")
    print0(f"Base learning rate: {hp.lr}")

    optimizer = init_optimizer(
        model=raw_model,
        device_mesh=device_mesh,
        ddp_model=model if isinstance(model, DDP) else None,
        hp=hp,
        cli_args=cli_args,
    )

    # Learning rate scheduler
    def get_lr(it):
        warmup_iters = round(hp.warmup_ratio * hp.num_iterations)
        warmdown_iters = round(hp.warmdown_ratio * hp.num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= hp.num_iterations - warmdown_iters:
            return 1.0
        else:
            return (hp.num_iterations - it) / warmdown_iters

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    print0("=" * 80)

    # --- Logging initialization ---
    run_name = build_run_name(hp, cli_args)

    # --- Set up checkpointing ---
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=hp.checkpoint_dir,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_id=None,
    )

    print0(f"Run name: {run_name}")
    print0(f"Debug mode: {cli_args.debug}")
    print0(f"Checkpoint directory: {hp.checkpoint_dir}")
    print0(
        f"Checkpoint frequency: {hp.checkpoint_freq if hp.checkpoint_freq > 0 else 'disabled'}"
    )

    # Load the latest checkpoint if it exists
    if hp.checkpoint_dir:
        checkpoint_manager.load(allow_missing=True)
        if checkpoint_manager.step is not None:
            print0(f"Resuming from step {checkpoint_manager.step}")
        else:
            print0("No previous checkpoint found, training model from scratch")
    else:
        # No checkpoint path provided
        print0("Training model from scratch")

    print0("=" * 80)

    # --- WandB initialization ---
    setup_wandb_if_enabled(hp, cli_args, checkpoint_manager, run_name)

    # --- Training Loop ---
    x, y = train_loader.next_batch()
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()

    # Use autocast for mixed precision
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    start_step = 0 if checkpoint_manager.step is None else checkpoint_manager.step + 1
    pbar = tqdm(total=hp.num_iterations, desc="Training", disable=not is_master_process())
    pbar.update(start_step)
    for step in range(start_step, hp.num_iterations + 1):
        # Skip the first few steps for timing to avoid torch.compile overhead
        if step == 10:
            training_time_ms = 0
            torch.cuda.synchronize()
            t0 = time.time()
        timed_steps = (step - 10) if step > 10 else float("nan")

        # --- Validation ---
        last_step = step == hp.num_iterations
        if last_step or (hp.val_loss_every > 0 and step % hp.val_loss_every == 0):
            # Disable eigen metrics collection during validation
            if eigen_metrics_tracker is not None:
                eigen_metrics_tracker.disable_collection()
            
            # Measure elapsed time for training
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)

            # Run validation
            val_loss = estimate_validation_loss(model, val_loader, val_steps, autocast_ctx, device=x.device)
            log_message = (
                f"step:{step}/{hp.num_iterations} val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps):.2f}ms"
            )
            print0(log_message)
            log_validation_if_enabled(cli_args, step, val_loss, training_time_ms)
            pbar.set_postfix(val_loss=f"{val_loss:.4f}")

            # Re-enable metrics collection after validation
            if eigen_metrics_tracker is not None:
                eigen_metrics_tracker.enable_collection()

            # Restart training time for the next iteration
            torch.cuda.synchronize()
            t0 = time.time()

        # Track gradient accumulation for eigen metrics
        should_log_eigen = False
        if eigen_metrics_tracker is not None:
            should_log_eigen = (step % hp.eigen_log_steps == 0) or (step == 1) or last_step
            if should_log_eigen:
                eigen_metrics_tracker.start_accumulation_tracking(step, grad_accum_steps)

        if last_step:
            # On last step, do a metrics-only forward pass (no backward/optimizer)
            # to capture final model representations
            if eigen_metrics_tracker is not None and should_log_eigen:
                model.eval()
                with torch.no_grad():
                    for i in range(1, grad_accum_steps + 1):
                        eigen_metrics_tracker.set_input_tokens(x)
                        eigen_metrics_tracker.process_micro_batch(i - 1)
                        with autocast_ctx:
                            _ = model(x, y)
                        x, y = train_loader.next_batch()
                    eigen_metrics_tracker.end_accumulation()
                    eigen_metrics_tracker.on_step_end(step, force=True)  # Force logging on last step
            break

        model.train()
        for i in range(1, grad_accum_steps + 1):
            # Set input tokens for frequency-bucketed analysis (before forward pass)
            if eigen_metrics_tracker is not None:
                eigen_metrics_tracker.set_input_tokens(x)
                eigen_metrics_tracker.process_micro_batch(i - 1)
            
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()  # for logging
            loss = loss / grad_accum_steps
            x, y = train_loader.next_batch()

            # Turn off DDP grad sync if replicate_mesh_grad_sync is True
            ddp_no_sync = i < grad_accum_steps or hp.replicate_mesh_grad_sync
            if isinstance(model, DDP) and ddp_no_sync:
                with model.no_sync():
                    loss.backward()
            else:
                if isinstance(model, FSDPModule):
                    # Gradient accumulation for DP on top of FSDP
                    model.set_is_last_backward(i == grad_accum_steps)
                    if cli_args.fast_fsdp:
                        # Only reshard and reduce-scatter gradients upon the last backward pass
                        # Keep the entire unsharded model in memory during gradient accumulation
                        model.set_reshard_after_backward(i == grad_accum_steps)
                        model.set_requires_gradient_sync(i == grad_accum_steps)
                    else:
                        # FSDP always synchronizes sharded gradients via reduce-scatter
                        model.set_requires_gradient_sync(True)
                loss.backward()

        # Compute eigen metrics at end of step
        if eigen_metrics_tracker is not None:
            eigen_metrics_tracker.end_accumulation()
            eigen_metrics_tracker.on_step_end(step)

        # Gradient norm
        grad_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)

        # Approximate updated training time just before logging
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        log_train_if_enabled(
            cli_args,
            step=step,
            train_loss=train_loss.item(),
            grad_norm=grad_norm.item(),
            training_time_ms=approx_time,
        )
        if is_master_process() and cli_args.debug:
            print0(
                f"Step {step}: train_loss={train_loss.item():.4f}, grad_norm={grad_norm.item():.4f}"
            )
        pbar.update(1)
        pbar.set_postfix(train_loss=f"{train_loss.item():.4f}")

        if hp.checkpoint_freq > 0 and step % hp.checkpoint_freq == 0 and step > 0:
            # See if optimizer defines synchronize_for_checkpoint()
            if hasattr(optimizer, "synchronize_for_checkpoint"):
                # Dion with replicate_mesh_grad_sync will have decoupled optimizer states
                # Calling this is necessary to synchronize state across the replicate mesh
                # Otherwise, checkpoint results will not be consistent
                optimizer.synchronize_for_checkpoint()

            # Save a distributed checkpoint
            checkpoint_manager.save(step=step)

        torch.cuda.synchronize()
        t0 = time.time()  # reset timer after optimizer step

    pbar.close()
    print0(
        f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    
    # Cleanup eigen metrics tracker
    if eigen_metrics_tracker is not None:
        eigen_metrics_tracker.cleanup()
        print0("Eigen metrics tracker cleaned up")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
