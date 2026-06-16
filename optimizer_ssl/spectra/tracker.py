"""Training-time spectral telemetry for GPT FFN activations."""

import gc
import os

import torch
import torch.distributed as dist

from optimizer_ssl.spectra.covariance import compute_covariance, compute_sorted_eigs, normalize_eigs
from optimizer_ssl.spectra.effective_rank import (
    compute_participation_ratio,
    compute_soft_rank,
    compute_spectral_entropy,
)
from optimizer_ssl.spectra.frequency_metrics import TokenFrequencyTable, covariance_stats_from_2d


class GPTEigenMetricsTracker:
    """
    Standalone class for tracking and computing eigenvalue metrics for GPT models.
    Designed to work with custom training loops and multi-GPU setups.
    
    Supports both:
    1. Pooled soft/hard spectral-rank telemetry across all tokens
    2. Frequency-bucketed telemetry for HEAD/MID/TAIL tokens
    
    Usage:
        tracker = GPTEigenMetricsTracker(
            model, 
            track_by_frequency=True,
            token_freq_file="path/to/frequencies.pt",
            ...
        )
        
        for step in range(num_steps):
            tracker.start_accumulation_tracking(step, grad_accum_steps)
            
            for i in range(grad_accum_steps):
                tracker.set_input_tokens(x)  # Required for frequency tracking
                tracker.process_micro_batch(i)
                loss = model(x, y)
                loss.backward()
            
            tracker.end_accumulation()
            tracker.on_step_end(step)
            
            optimizer.step()
    """

    def __init__(self, 
                 model,
                 log_steps=100, 
                 device=None,
                 output_dir="eigen_metrics_logs",
                 num_layers=None,
                 global_rank=0,
                 world_size=1,
                 console_log_level="ERROR",
                 gather_statistics=True,
                 # Frequency tracking options
                 track_by_frequency=False,
                 token_freq_file=None,
                 token_freq_table=None,
                 min_samples_per_bucket=1000,
                 frequency_bucket_reduction="rank0_local",
                 error_policy="warn"):
        """
        Initialize the eigen metrics tracker.
        
        Args:
            model: GPT model (raw or DDP-wrapped)
            log_steps: Compute metrics every N steps
            device: Device for computation
            output_dir: Directory for log files
            num_layers: Number of transformer layers (auto-detected if None)
            global_rank: Rank in distributed training
            world_size: Total number of processes
            console_log_level: Logging level
            gather_statistics: Whether to gather stats across ranks
            track_by_frequency: Enable frequency-bucketed metrics (tertiles: HEAD/MID/TAIL)
            token_freq_file: Path to precomputed token frequencies (.pt file)
            token_freq_table: Pre-initialized TokenFrequencyTable (alternative to file)
            min_samples_per_bucket: Minimum samples required to compute bucket metrics
            frequency_bucket_reduction: ``rank0_local`` reproduces the submitted-paper bucket telemetry;
                ``distributed_covariance`` reduces bucket covariance statistics across ranks.
            error_policy: ``warn`` logs explicit missing markers and continues, ``raise`` fails fast,
                and ``nan`` writes missing/NaN markers where possible.
        """
        # Device configuration
        self.compute_device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.storage_device = 'cpu'
        
        # Multi-GPU settings
        self.global_rank = global_rank
        self.world_size = world_size
        self.gather_statistics = gather_statistics
        allowed_error_policies = {"warn", "raise", "nan"}
        if error_policy not in allowed_error_policies:
            raise ValueError(
                "error_policy must be one of "
                f"{sorted(allowed_error_policies)}, got {error_policy!r}"
            )
        self.error_policy = error_policy
        
        # Logging configuration
        self.log_steps = log_steps
        
        # Determine number of layers
        if num_layers is None:
            if hasattr(model, 'module'):
                if hasattr(model.module, 'transformer') and hasattr(model.module.transformer, 'h'):
                    num_layers = len(model.module.transformer.h)
                else:
                    raise ValueError("Could not determine number of layers from DDP-wrapped model")
            else:
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    num_layers = len(model.transformer.h)
                else:
                    raise ValueError("Could not determine number of layers from model")
                
            if global_rank == 0:
                print(f"Auto-detected {num_layers} layers in GPT model")
                
        self.num_layers = num_layers
        
        # State tracking
        self.current_step = 0
        self.last_logged_step = 0
        self.collecting_activations = False
        
        # Gradient accumulation tracking
        self.current_accumulation_step = 0
        self.accumulation_size = 1
        self.tracking_full_batch = False
        
        # Storage for activations
        self.layer_pre_acts = {}
        self.layer_post_acts = {}
        self.hook_handles = []
        
        # =============================================
        # Frequency tracking setup
        # =============================================
        self.track_by_frequency = track_by_frequency
        self.min_samples_per_bucket = min_samples_per_bucket
        self.frequency_bucket_reduction = frequency_bucket_reduction
        allowed_frequency_reductions = {"rank0_local", "distributed_covariance"}
        if self.frequency_bucket_reduction not in allowed_frequency_reductions:
            raise ValueError(
                "frequency_bucket_reduction must be one of "
                f"{sorted(allowed_frequency_reductions)}, got {self.frequency_bucket_reduction!r}"
            )
        self.current_token_ids = None
        self.token_ids_buffer = []  # Store token IDs for each micro-batch
        
        if track_by_frequency:
            # Initialize token frequency table
            if token_freq_table is not None:
                self.token_freq_table = token_freq_table
            elif token_freq_file is not None:
                self.token_freq_table = TokenFrequencyTable()
                self.token_freq_table.load_from_file(token_freq_file)
            else:
                raise ValueError("track_by_frequency=True requires either token_freq_file or token_freq_table")
            
            if global_rank == 0:
                print(f"Frequency tracking enabled")
                print(f"Frequency bucket reduction: {self.frequency_bucket_reduction}")
                print(self.token_freq_table.summary())
        else:
            self.token_freq_table = None
        
        
        # =============================================
        # Setup output directories and files
        # =============================================
        if global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            self.log_file_path = f"{output_dir}/eigen_metrics.log"
            
            try:
                self.log_file = open(self.log_file_path, "a")
                self.layer_files = []
                
                # Create per-layer spectral telemetry log files
                for i in range(num_layers):
                    layer_file = f"{output_dir}/layer_{i}_eigen.txt"
                    with open(layer_file, "a") as f:
                        f.write("")
                    self.layer_files.append(layer_file)
                
                # Setup frequency-bucketed output directory (tertiles: HEAD/MID/TAIL)
                if track_by_frequency:
                    self.freq_output_dir = os.path.join(output_dir, "frequency_tertiles")
                    os.makedirs(self.freq_output_dir, exist_ok=True)
                    
                    # Create tertile log files for each layer
                    self.layer_freq_files = []
                    for i in range(num_layers):
                        layer_freq_file = f"{self.freq_output_dir}/layer_{i}_eigen_freq.txt"
                        with open(layer_freq_file, "a") as f:
                            f.write("")
                        self.layer_freq_files.append(layer_freq_file)
                    
                    # Create summary file for Tail Integrity Index
                    self.tii_file = f"{self.freq_output_dir}/tail_integrity_index.txt"
                    with open(self.tii_file, "a") as f:
                        f.write("# Tail Integrity Index (TII) per step\n")
                        f.write("# TII = (1/L) * sum_l (hard_rank_post^(l,tail) - hard_rank_post^(l,head))\n")
                        f.write("# Higher TII (closer to 0) = more equitable treatment of rare tokens\n\n")
                
                
                # Write header to log file
                self.log_file.write(f"# Eigenvalue metrics log - {num_layers} layers\n")
                if self.gather_statistics and world_size > 1:
                    self.log_file.write(f"# Computing metrics across all {world_size} ranks\n")
                else:
                    self.log_file.write(f"# Computing metrics from rank 0 only\n")
                self.log_file.write(f"# Spectral telemetry error policy: {self.error_policy}\n")
                if track_by_frequency:
                    self.log_file.write(f"# Frequency-bucketed metrics enabled\n")
                    self.log_file.write(f"# Frequency bucket reduction: {self.frequency_bucket_reduction}\n")
                    self.log_file.write(f"# Frequency logs in: {self.freq_output_dir}\n")
                self.log_file.flush()
                
                print(f"Eigenmetrics output directory: {output_dir}")
                if track_by_frequency:
                    print(f"Frequency-bucketed logs in: {self.freq_output_dir}")
                    print(f"Frequency bucket reduction: {self.frequency_bucket_reduction}")
                    
            except Exception as e:
                self._handle_metric_error("setting up spectral log files", e)
        
        # Store metric functions
        self.eigen_funcs = {
            'compute_covariance': compute_covariance,
            'compute_sorted_eigs': compute_sorted_eigs,
            'normalize_eigs': normalize_eigs,
            'compute_spectral_entropy': compute_spectral_entropy,
            'compute_soft_rank': compute_soft_rank,
            'compute_participation_ratio': compute_participation_ratio,
        }
        
        # Register hooks. Hook registration failures are fatal because otherwise
        # a run can complete with no spectral telemetry.
        self.register_hooks(model)
    
    def _handle_metric_error(self, context: str, exc: Exception, *, layer_idx=None, step=None) -> None:
        """Handle telemetry errors according to ``self.error_policy``.

        ``warn`` is the paper-scale default: it logs an explicit missing marker
        and continues. ``raise`` is recommended for debugging because it fails
        fast instead of silently skipping a layer/step. ``nan`` currently behaves
        like ``warn`` plus a missing-marker line in the relevant layer log.
        """
        message = f"{context}: {type(exc).__name__}: {exc}"
        if self.global_rank == 0:
            print(f"Spectral telemetry warning: {message}")
            if hasattr(self, "log_file"):
                self.log_file.write(f"MISSING_METRIC context={context!r} layer={layer_idx} step={step} reason={message!r}\n")
                self.log_file.flush()
            if layer_idx is not None and step is not None and hasattr(self, "layer_files"):
                try:
                    with open(self.layer_files[layer_idx], "a") as f:
                        f.write(f"Step {step}: status=missing reason={message!r}\n")
                except Exception:
                    pass
        if self.error_policy == "raise":
            raise RuntimeError(message) from exc

    def set_input_tokens(self, token_ids: torch.Tensor):
        """
        Set the current batch's token IDs for frequency-bucketed analysis.
        Call this BEFORE each forward pass when track_by_frequency=True.
        
        Args:
            token_ids: [B, S] tensor of input token IDs
        """
        if not self.track_by_frequency or not self.collecting_activations:
            return
            
        self.current_token_ids = token_ids.detach().to(self.storage_device)
    
    def start_accumulation_tracking(self, step, gradient_accumulation_steps):
        """
        Start tracking activation statistics for all micro-batches in an accumulation step.
        
        Args:
            step: The update step number
            gradient_accumulation_steps: Number of micro-batches
        """
        should_track = self.should_log_step(step)
        if should_track:
            if self.global_rank == 0:
                print(f"Beginning activation collection for ALL {gradient_accumulation_steps} micro-batches in step {step}")
                
            self.current_step = step
            self.current_accumulation_step = step
            self.accumulation_size = gradient_accumulation_steps
            self.tracking_full_batch = True
            self.collecting_activations = True
            
            # Clear buffers
            self.layer_pre_acts = {}
            self.layer_post_acts = {}
            self.token_ids_buffer = []
    
    def process_micro_batch(self, micro_batch_idx):
        """
        Process a micro-batch within gradient accumulation.
        
        Args:
            micro_batch_idx: Current micro-batch index (0-based)
        """
        if not self.tracking_full_batch:
            return
        
        # Store token IDs for this micro-batch (if frequency tracking enabled)
        if self.track_by_frequency and self.current_token_ids is not None:
            self.token_ids_buffer.append(self.current_token_ids.clone())
            
        if self.global_rank == 0 and self.collecting_activations:
            if micro_batch_idx == 0:
                print(f"Collected first micro-batch for step {self.current_accumulation_step}")
            elif micro_batch_idx == self.accumulation_size - 1:
                print(f"Collected final micro-batch ({micro_batch_idx+1}/{self.accumulation_size}) for step {self.current_accumulation_step}")
    
    def end_accumulation(self):
        """
        Signal the end of gradient accumulation.
        Called after all micro-batches have been processed.
        """
        if not self.tracking_full_batch:
            return
            
        if self.global_rank == 0:
            print(f"Completing accumulation for step {self.current_accumulation_step}")
    
    def disable_collection(self):
        """Temporarily disable activation collection without removing hooks."""
        self.collecting_activations = False
        if self.global_rank == 0 and hasattr(self, 'log_file'):
            self.log_file.write(f"Temporarily disabled activation collection at step {self.current_step}\n")
            self.log_file.flush()
            print(f"Disabled eigenvalue metrics collection for evaluation")

    def enable_collection(self):
        """Re-enable activation collection if we're in a tracking step."""
        if self.tracking_full_batch:
            self.collecting_activations = True
            if self.global_rank == 0 and hasattr(self, 'log_file'):
                self.log_file.write(f"Re-enabled activation collection at step {self.current_step}\n")
                self.log_file.flush()
                print(f"Re-enabled eigenvalue metrics collection after evaluation")
    
    def register_hooks(self, model):
        """
        Register hooks on the GPT model's MLP layers.
        """
        try:
            if hasattr(model, 'module'):
                if hasattr(model.module, 'transformer'):
                    blocks = model.module.transformer.h
                else:
                    raise ValueError("Could not find transformer in DDP-wrapped model")
            else:
                if hasattr(model, 'transformer'):
                    blocks = model.transformer.h
                else:
                    raise ValueError("Could not find transformer in model")
            
            num_layers = len(blocks)
            if self.global_rank == 0:
                self.log_file.write(f"Found {num_layers} layers in model\n")
                self.log_file.flush()
            
            self.hook_handles = []
            
            for layer_idx in range(num_layers):
                block = blocks[layer_idx]
                
                if not hasattr(block, 'mlp'):
                    if self.global_rank == 0:
                        self.log_file.write(f"Block {layer_idx} does not have 'mlp' attribute\n")
                        self.log_file.flush()
                    continue
                
                if not (hasattr(block.mlp, 'c_fc') and hasattr(block.mlp, 'c_proj')):
                    if self.global_rank == 0:
                        self.log_file.write(f"Block {layer_idx} mlp missing c_fc or c_proj\n")
                        self.log_file.flush()
                    continue
                
                c_fc = block.mlp.c_fc
                c_proj = block.mlp.c_proj
                
                def get_fc_forward_hook(idx):
                    def fc_forward_hook(mod, inp, out):
                        if not self.collecting_activations:
                            return
                        self.capture_pre_acts(idx, out)
                    return fc_forward_hook
                
                def get_proj_pre_hook(idx):
                    def proj_pre_hook(mod, inp):
                        if not self.collecting_activations:
                            return
                        self.capture_post_acts(idx, inp[0])
                    return proj_pre_hook
                
                h1 = c_fc.register_forward_hook(get_fc_forward_hook(layer_idx))
                h2 = c_proj.register_forward_pre_hook(get_proj_pre_hook(layer_idx))
                
                self.hook_handles.extend([h1, h2])
            
            if self.global_rank == 0:
                self.log_file.write(f"Successfully registered hooks for {num_layers} GPT blocks\n")
                self.log_file.flush()
        
        except Exception as e:
            if self.global_rank == 0:
                print(f"Error in register_hooks: {str(e)}")
                if hasattr(self, 'log_file'):
                    self.log_file.write(f"Error in register_hooks: {str(e)}\n")
                    self.log_file.flush()
            raise e
    
    def capture_pre_acts(self, layer_idx, tensor):
        """Capture pre-activations (output of c_fc)."""
        try:
            if layer_idx not in self.layer_pre_acts:
                self.layer_pre_acts[layer_idx] = []
            
            if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
                
            self.layer_pre_acts[layer_idx].append(tensor.detach().to(self.storage_device))
        except Exception as e:
            self._handle_metric_error("capture_pre_acts", e, layer_idx=layer_idx, step=self.current_step)

    def capture_post_acts(self, layer_idx, tensor):
        """Capture post-activations (input to c_proj after ReLU²)."""
        try:
            if layer_idx not in self.layer_post_acts:
                self.layer_post_acts[layer_idx] = []
                    
            if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
                
            self.layer_post_acts[layer_idx].append(tensor.detach().to(self.storage_device))
        except Exception as e:
            self._handle_metric_error("capture_post_acts", e, layer_idx=layer_idx, step=self.current_step)
    
    def on_step_end(self, step, force=False):
        """
        Call at the end of each step to compute and log metrics.
        
        Args:
            step: Current training step
            force: If True, log metrics regardless of step number (used for last step)
        """
        if not force and (not self.should_log_step(step) or step == self.last_logged_step):
            return
        
        # Avoid double-logging the same step
        if step == self.last_logged_step:
            return
        
        if self.layer_pre_acts and self.layer_post_acts:
            if self.global_rank == 0:
                print(f"Computing metrics for step {step} with data from ALL MICRO-BATCHES")
            
            # Compute pooled spectral metrics
            self._compute_metrics(step)
            
            # Compute frequency-bucketed metrics if enabled
            if self.track_by_frequency:
                self._compute_frequency_metrics(step)
        
        # Cleanup
        self.collecting_activations = False
        self.last_logged_step = step
        self.tracking_full_batch = False
        self.layer_pre_acts = {}
        self.layer_post_acts = {}
        self.token_ids_buffer = []
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def should_log_step(self, step):
        """Check if current step should be logged."""
        return step == 1 or step % self.log_steps == 0
    
    def _compute_local_covariance(self, acts_list):
        """
        Compute covariance matrix and mean from local activations.
        
        Args:
            acts_list: List of activation tensors
            
        Returns:
            (covariance_matrix, local_mean, num_samples) tuple
        """
        sample = acts_list[0]
        dim = sample.shape[-1]
        
        if sample.dim() == 3:
            tensor = torch.cat([x.reshape(-1, dim) for x in acts_list], dim=0)
        else:
            tensor = torch.cat(acts_list, dim=0)
        
        n_samples = tensor.size(0)
        tensor = tensor.to(self.compute_device)
        
        local_mean = tensor.mean(dim=0)
        cov = self.eigen_funcs['compute_covariance'](tensor)
        
        del tensor
        return cov, local_mean, n_samples
    
    def _reduce_covariance_across_ranks(self, local_cov, local_mean, local_n_samples, layer_idx, act_type):
        """
        Average covariance matrices across all ranks with between-means correction.
        """
        if not self.gather_statistics or self.world_size <= 1:
            return local_cov, local_n_samples
        
        try:
            device = self.compute_device
            local_cov = local_cov.to(device)
            local_mean = local_mean.to(device)
            
            local_n_tensor = torch.tensor([local_n_samples], dtype=torch.long, device=device)
            all_n_samples = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.world_size)]
            dist.all_gather(all_n_samples, local_n_tensor)
            all_n_samples = [n.item() for n in all_n_samples]
            total_samples = sum(all_n_samples)
            
            weighted_cov = local_cov * (local_n_samples - 1)
            dist.all_reduce(weighted_cov, op=dist.ReduceOp.SUM)
            
            dim = local_mean.shape[0]
            all_means = [torch.zeros(dim, dtype=local_mean.dtype, device=device) 
                        for _ in range(self.world_size)]
            dist.all_gather(all_means, local_mean)
            
            global_mean = torch.zeros_like(local_mean)
            for i, (mean, n) in enumerate(zip(all_means, all_n_samples)):
                global_mean += mean * (n / total_samples)
            
            between_means_correction = torch.zeros_like(local_cov)
            for i, (mean_i, n_i) in enumerate(zip(all_means, all_n_samples)):
                mean_diff = mean_i - global_mean
                outer_prod = torch.outer(mean_diff, mean_diff)
                between_means_correction += outer_prod * n_i
            
            unnormalized_cov = weighted_cov + between_means_correction
            exact_global_cov = unnormalized_cov / (total_samples - 1)
            
            if self.global_rank == 0 and hasattr(self, 'log_file'):
                correction_norm = torch.norm(between_means_correction).item()
                cov_norm = torch.norm(weighted_cov).item()
                relative_correction = correction_norm / (cov_norm + 1e-12)
                self.log_file.write(f"Layer {layer_idx} ({act_type}): Global cov from {total_samples} samples "
                                   f"(between-means correction: {relative_correction:.6f})\n")
                self.log_file.flush()
            
            return exact_global_cov, total_samples
            
        except Exception as e:
            self._handle_metric_error("reduce pooled covariance", e, layer_idx=layer_idx)
            return local_cov, local_n_samples
    
    def _compute_metrics_from_covariance(self, cov, n_samples):
        """Compute paper-facing spectral metrics from a covariance matrix."""
        lam = self.eigen_funcs['compute_sorted_eigs'](cov)
        lam_norm = self.eigen_funcs['normalize_eigs'](lam)
        spectral_entropy = self.eigen_funcs['compute_spectral_entropy'](lam_norm).item()
        soft_rank = self.eigen_funcs['compute_soft_rank'](lam_norm).item()
        hard_rank = self.eigen_funcs['compute_participation_ratio'](lam).item()
        return {
            'soft_rank': soft_rank,
            'hard_rank': hard_rank,
            'spectral_entropy': spectral_entropy,
            'n_samples': n_samples,
        }
    
    def _compute_metrics(self, step):
        """Compute paper-facing spectral-rank metrics for all layers (all tokens pooled)."""
        try:
            for layer_idx in sorted(self.layer_pre_acts.keys()):
                if layer_idx not in self.layer_post_acts:
                    continue
                    
                if not self.layer_pre_acts[layer_idx] or not self.layer_post_acts[layer_idx]:
                    continue
                
                try:
                    pre_acts = self.layer_pre_acts[layer_idx]
                    post_acts = self.layer_post_acts[layer_idx]
                    
                    pre_cov, pre_mean, pre_n_samples = self._compute_local_covariance(pre_acts)
                    post_cov, post_mean, post_n_samples = self._compute_local_covariance(post_acts)
                    
                    if self.gather_statistics and self.world_size > 1:
                        pre_cov, pre_n_samples = self._reduce_covariance_across_ranks(
                            pre_cov, pre_mean, pre_n_samples, layer_idx, "pre")
                        post_cov, post_n_samples = self._reduce_covariance_across_ranks(
                            post_cov, post_mean, post_n_samples, layer_idx, "post")
                    
                    if self.global_rank == 0:
                        pre_metrics = self._compute_metrics_from_covariance(pre_cov, pre_n_samples)
                        post_metrics = self._compute_metrics_from_covariance(post_cov, post_n_samples)
                        
                        
                        # Write paper-facing layer metrics.
                        layer_file = self.layer_files[layer_idx]
                        with open(layer_file, 'a') as f:
                            f.write(f"Step {step}: "
                                    f"soft_rank_pre={pre_metrics['soft_rank']:.2f}, "
                                    f"soft_rank_post={post_metrics['soft_rank']:.2f}, "
                                    f"hard_rank_pre={pre_metrics['hard_rank']:.2f}, "
                                    f"hard_rank_post={post_metrics['hard_rank']:.2f}, "
                                    f"spectral_entropy_pre={pre_metrics['spectral_entropy']:.3f}, "
                                    f"spectral_entropy_post={post_metrics['spectral_entropy']:.3f}\n")
                        
                        del pre_metrics, post_metrics, pre_cov, post_cov
                    
                except Exception as e:
                    self._handle_metric_error("pooled spectral metrics", e, layer_idx=layer_idx, step=step)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            if self.global_rank == 0 and hasattr(self, 'log_file'):
                data_source = "all GPUs" if self.gather_statistics and self.world_size > 1 else "rank 0 only"
                self.log_file.write(f"Completed spectral-rank metrics for step {step} ({data_source})\n")
                self.log_file.flush()
            
        except Exception as e:
            self._handle_metric_error("pooled spectral metrics step", e, step=step)
    
    def _compute_bucket_covariance_stats(self, flat_acts: torch.Tensor, mask: torch.Tensor, dim: int):
        """Compute local covariance sufficient statistics for one token bucket."""
        mask = mask.to(flat_acts.device)
        if mask.numel() != flat_acts.shape[0]:
            raise ValueError(
                f"Bucket mask length {mask.numel()} does not match activations {flat_acts.shape[0]}"
            )
        if mask.any():
            bucket = flat_acts[mask].to(self.compute_device)
        else:
            bucket = torch.empty(0, dim, dtype=flat_acts.dtype, device=self.compute_device)
        stats = covariance_stats_from_2d(bucket)
        return stats

    def _reduce_covariance_stats_across_ranks(self, local_stats, layer_idx, act_type):
        """Reduce covariance sufficient statistics across data-parallel ranks.

        This is used by ``frequency_bucket_reduction='distributed_covariance'``.
        All ranks must enter this method for each layer/bucket/activation type,
        even when their local bucket is empty, to avoid distributed deadlocks.
        """
        local_mean = local_stats["mean"].to(self.compute_device)
        local_scatter = local_stats["scatter"].to(self.compute_device)
        local_n = int(local_stats["n"])

        if not self.gather_statistics or self.world_size <= 1:
            if local_n < 2:
                cov = torch.zeros_like(local_scatter)
            else:
                cov = local_scatter / (local_n - 1)
            return cov, local_n

        try:
            device = self.compute_device
            dim = local_mean.shape[0]

            local_n_tensor = torch.tensor([local_n], dtype=torch.long, device=device)
            all_n_tensors = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.world_size)]
            dist.all_gather(all_n_tensors, local_n_tensor)
            all_n_samples = [int(n.item()) for n in all_n_tensors]
            total_samples = sum(all_n_samples)

            # Sum within-rank centered scatter matrices.
            global_scatter = local_scatter.clone()
            dist.all_reduce(global_scatter, op=dist.ReduceOp.SUM)

            # Gather local means for between-rank mean correction.
            all_means = [torch.zeros(dim, dtype=local_mean.dtype, device=device) for _ in range(self.world_size)]
            dist.all_gather(all_means, local_mean)

            if total_samples == 0:
                return torch.zeros_like(local_scatter), 0

            global_mean = torch.zeros_like(local_mean)
            for mean, n in zip(all_means, all_n_samples):
                if n > 0:
                    global_mean += mean * (n / total_samples)

            between_means_correction = torch.zeros_like(local_scatter)
            for mean_i, n_i in zip(all_means, all_n_samples):
                if n_i == 0:
                    continue
                mean_diff = mean_i - global_mean
                between_means_correction += torch.outer(mean_diff, mean_diff) * n_i

            unnormalized_cov = global_scatter + between_means_correction
            if total_samples < 2:
                exact_global_cov = torch.zeros_like(local_scatter)
            else:
                exact_global_cov = unnormalized_cov / (total_samples - 1)

            return exact_global_cov, total_samples

        except Exception as e:
            self._handle_metric_error(
                f"reduce frequency-bucket covariance ({act_type})", e, layer_idx=layer_idx
            )
            if local_n < 2:
                cov = torch.zeros_like(local_scatter)
            else:
                cov = local_scatter / (local_n - 1)
            return cov, local_n

    def _write_frequency_metrics_for_layer(self, layer_idx, step, tertile_metrics):
        """Write frequency-bucketed metrics for one layer on rank 0."""
        if self.global_rank != 0:
            return
        with open(self.layer_freq_files[layer_idx], 'a') as f:
            f.write(f"Step {step}: frequency_bucket_reduction={self.frequency_bucket_reduction}\n")
            for tname in TokenFrequencyTable.TERTILE_NAMES:
                tm = tertile_metrics.get(tname)
                if tm is None:
                    f.write(f"  {tname.upper()}: insufficient samples\n")
                else:
                    f.write(f"  {tname.upper()} (n={tm['n_samples']}): "
                            f"soft_rank_pre={tm['pre']['soft_rank']:.2f}, "
                            f"soft_rank_post={tm['post']['soft_rank']:.2f}, "
                            f"hard_rank_pre={tm['pre']['hard_rank']:.2f}, "
                            f"hard_rank_post={tm['post']['hard_rank']:.2f}, "
                            f"spectral_entropy_pre={tm['pre']['spectral_entropy']:.3f}, "
                            f"spectral_entropy_post={tm['post']['spectral_entropy']:.3f}\n")

    def _update_tii_from_metrics(self, tertile_metrics, tii_pr_diffs):
        """Append one layer's tail-minus-head hard-rank gap when available."""
        if (tertile_metrics.get('head') is not None and tertile_metrics.get('tail') is not None):
            pr_diff = (
                tertile_metrics['tail']['post']['hard_rank']
                - tertile_metrics['head']['post']['hard_rank']
            )
            tii_pr_diffs.append(pr_diff)

    def _log_tail_integrity_index(self, step, tii_pr_diffs):
        """Log Tail Integrity Index on rank 0."""
        if self.global_rank == 0 and tii_pr_diffs:
            tii = sum(tii_pr_diffs) / len(tii_pr_diffs)
            with open(self.tii_file, 'a') as f:
                f.write(
                    f"Step {step}: TII={tii:.4f} (n_layers={len(tii_pr_diffs)}, "
                    f"frequency_bucket_reduction={self.frequency_bucket_reduction})\n"
                )
            self.log_file.write(f"Step {step}: Tail Integrity Index = {tii:.4f}\n")
            self.log_file.flush()

    def _compute_frequency_metrics(self, step):
        """Compute frequency-bucketed metrics in the configured reduction mode."""
        if self.frequency_bucket_reduction == "rank0_local":
            self._compute_frequency_metrics_rank0_local(step)
        elif self.frequency_bucket_reduction == "distributed_covariance":
            self._compute_frequency_metrics_distributed_covariance(step)
        else:
            raise ValueError(f"Unknown frequency_bucket_reduction={self.frequency_bucket_reduction!r}")

    def _compute_frequency_metrics_rank0_local(self, step):
        """Compute HEAD/MID/TAIL metrics from each rank's local shard.

        Only rank 0 writes its local bucket metrics. This reproduces the
        submitted-paper telemetry path for frequency-bucketed spectra.
        """
        if not self.track_by_frequency or not self.token_ids_buffer:
            return

        try:
            all_token_ids = torch.cat(self.token_ids_buffer, dim=0)
            all_token_ids_flat = all_token_ids.reshape(-1)
            tertiles = self.token_freq_table.get_tertile(all_token_ids_flat)
            tii_pr_diffs = []

            for layer_idx in sorted(self.layer_pre_acts.keys()):
                if layer_idx not in self.layer_post_acts:
                    continue
                if not self.layer_pre_acts[layer_idx] or not self.layer_post_acts[layer_idx]:
                    continue

                try:
                    pre_acts = self.layer_pre_acts[layer_idx]
                    post_acts = self.layer_post_acts[layer_idx]
                    dim = pre_acts[0].shape[-1]
                    pre_flat = torch.cat([x.reshape(-1, dim) for x in pre_acts], dim=0)
                    post_flat = torch.cat([x.reshape(-1, dim) for x in post_acts], dim=0)

                    if pre_flat.shape[0] != all_token_ids_flat.shape[0]:
                        if self.global_rank == 0:
                            self.log_file.write(
                                f"Warning: Layer {layer_idx} activation count ({pre_flat.shape[0]}) "
                                f"!= token count ({all_token_ids_flat.shape[0]})\n")
                            self.log_file.flush()
                        continue

                    tertile_metrics = {}
                    for tertile_idx, tertile_name in enumerate(TokenFrequencyTable.TERTILE_NAMES):
                        mask = tertiles == tertile_idx
                        n_samples = int(mask.sum().item())
                        if n_samples < self.min_samples_per_bucket:
                            tertile_metrics[tertile_name] = None
                            continue

                        pre_bucket = pre_flat[mask].to(self.compute_device)
                        post_bucket = post_flat[mask].to(self.compute_device)

                        pre_cov = self.eigen_funcs['compute_covariance'](pre_bucket)
                        post_cov = self.eigen_funcs['compute_covariance'](post_bucket)
                        pre_m = self._compute_metrics_from_covariance(pre_cov, n_samples)
                        post_m = self._compute_metrics_from_covariance(post_cov, n_samples)

                        tertile_metrics[tertile_name] = {'pre': pre_m, 'post': post_m, 'n_samples': n_samples}
                        del pre_bucket, post_bucket, pre_cov, post_cov

                    self._update_tii_from_metrics(tertile_metrics, tii_pr_diffs)
                    self._write_frequency_metrics_for_layer(layer_idx, step, tertile_metrics)
                    del pre_flat, post_flat

                except Exception as e:
                    self._handle_metric_error("rank0 frequency metrics", e, layer_idx=layer_idx, step=step)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._log_tail_integrity_index(step, tii_pr_diffs)
            if self.global_rank == 0 and hasattr(self, 'log_file'):
                self.log_file.write(
                    f"Completed frequency-bucketed metrics for step {step} "
                    f"(frequency_bucket_reduction=rank0_local)\n"
                )
                self.log_file.flush()

        except Exception as e:
            self._handle_metric_error("rank0 frequency metrics step", e, step=step)

    def _compute_frequency_metrics_distributed_covariance(self, step):
        """Compute HEAD/MID/TAIL metrics from globally reduced bucket covariance statistics."""
        if not self.track_by_frequency or not self.token_ids_buffer:
            return

        try:
            all_token_ids = torch.cat(self.token_ids_buffer, dim=0)
            all_token_ids_flat = all_token_ids.reshape(-1)
            tertiles = self.token_freq_table.get_tertile(all_token_ids_flat)
            tii_pr_diffs = []

            for layer_idx in sorted(self.layer_pre_acts.keys()):
                if layer_idx not in self.layer_post_acts:
                    continue
                if not self.layer_pre_acts[layer_idx] or not self.layer_post_acts[layer_idx]:
                    continue

                try:
                    pre_acts = self.layer_pre_acts[layer_idx]
                    post_acts = self.layer_post_acts[layer_idx]
                    dim = pre_acts[0].shape[-1]
                    pre_flat = torch.cat([x.reshape(-1, dim) for x in pre_acts], dim=0)
                    post_flat = torch.cat([x.reshape(-1, dim) for x in post_acts], dim=0)

                    local_alignment_ok = int(pre_flat.shape[0] == all_token_ids_flat.shape[0])
                    if self.gather_statistics and self.world_size > 1:
                        ok_tensor = torch.tensor([local_alignment_ok], dtype=torch.long, device=self.compute_device)
                        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
                        global_alignment_ok = bool(ok_tensor.item())
                    else:
                        global_alignment_ok = bool(local_alignment_ok)

                    if not global_alignment_ok:
                        if self.global_rank == 0:
                            self.log_file.write(
                                f"Warning: Skipping distributed frequency metrics for layer {layer_idx} "
                                "because at least one rank has token/activation misalignment.\n"
                            )
                            self.log_file.flush()
                        del pre_flat, post_flat
                        continue

                    tertile_metrics = {}
                    for tertile_idx, tertile_name in enumerate(TokenFrequencyTable.TERTILE_NAMES):
                        mask = tertiles == tertile_idx

                        pre_stats = self._compute_bucket_covariance_stats(pre_flat, mask, dim)
                        post_stats = self._compute_bucket_covariance_stats(post_flat, mask, dim)

                        pre_cov, pre_n = self._reduce_covariance_stats_across_ranks(
                            pre_stats, layer_idx, f"pre/{tertile_name}"
                        )
                        post_cov, post_n = self._reduce_covariance_stats_across_ranks(
                            post_stats, layer_idx, f"post/{tertile_name}"
                        )
                        n_samples = min(pre_n, post_n)

                        if n_samples < self.min_samples_per_bucket:
                            tertile_metrics[tertile_name] = None
                        elif self.global_rank == 0:
                            pre_m = self._compute_metrics_from_covariance(pre_cov, pre_n)
                            post_m = self._compute_metrics_from_covariance(post_cov, post_n)
                            tertile_metrics[tertile_name] = {
                                'pre': pre_m,
                                'post': post_m,
                                'n_samples': n_samples,
                            }
                        else:
                            tertile_metrics[tertile_name] = None

                        del pre_cov, post_cov

                    if self.global_rank == 0:
                        self._update_tii_from_metrics(tertile_metrics, tii_pr_diffs)
                        self._write_frequency_metrics_for_layer(layer_idx, step, tertile_metrics)
                    del pre_flat, post_flat

                except Exception as e:
                    self._handle_metric_error("distributed frequency metrics", e, layer_idx=layer_idx, step=step)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._log_tail_integrity_index(step, tii_pr_diffs)
            if self.global_rank == 0 and hasattr(self, 'log_file'):
                self.log_file.write(
                    f"Completed frequency-bucketed metrics for step {step} "
                    f"(frequency_bucket_reduction=distributed_covariance)\n"
                )
                self.log_file.flush()

        except Exception as e:
            self._handle_metric_error("distributed frequency metrics step", e, step=step)

    def cleanup(self):
        """Clean up resources."""
        if self.global_rank == 0:
            print("Cleaning up eigenmetrics tracker - removing hooks")
            if hasattr(self, 'log_file'):
                self.log_file.write("Cleaning up eigenmetrics tracker\n")
                self.log_file.flush()
            
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
        
        self.layer_pre_acts = {}
        self.layer_post_acts = {}
        self.token_ids_buffer = []
        
        if self.global_rank == 0 and hasattr(self, 'log_file'):
            self.log_file.write(f"Eigenmetrics tracker cleaned up successfully\n")
            self.log_file.flush()
            self.log_file.close()
