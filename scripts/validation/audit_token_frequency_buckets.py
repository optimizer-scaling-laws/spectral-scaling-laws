#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from optimizer_ssl.data.binary_shards import load_headered_token_shard
from optimizer_ssl.spectra.frequency_metrics import load_frequency_vector


def sha256_of_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def find_data_files(data_dir=None, data_glob=None):
    if (data_dir is None) == (data_glob is None):
        raise ValueError("Provide exactly one of --data_dir or --data_glob")

    if data_dir is not None:
        tried = []
        for pattern in ["fineweb_train_*.bin", "*train*.bin", "*.bin"]:
            p = os.path.join(data_dir, pattern)
            tried.append(p)
            files = sorted(glob.glob(p))
            if files:
                return files, p, tried
        return [], None, tried

    files = sorted(glob.glob(data_glob))
    return files, data_glob, [data_glob]


def compute_bucket_stats(freq: torch.Tensor):
    sorted_freqs, _ = freq.sort(descending=True)
    cumsum = sorted_freqs.cumsum(0).float()
    total = cumsum[-1].item()

    head_cutoff_idx = (cumsum <= total * 0.33).sum().item()
    mid_cutoff_idx = (cumsum <= total * 0.67).sum().item()

    head_cutoff_idx = max(1, min(head_cutoff_idx, len(sorted_freqs) - 1))
    mid_cutoff_idx = max(head_cutoff_idx + 1, min(mid_cutoff_idx, len(sorted_freqs) - 1))

    head_min_freq = sorted_freqs[head_cutoff_idx - 1].item()
    mid_min_freq = sorted_freqs[mid_cutoff_idx - 1].item()

    head_mask = freq >= head_min_freq
    mid_mask = (freq >= mid_min_freq) & (freq < head_min_freq)
    tail_mask = freq < mid_min_freq

    head_occ = freq[head_mask].sum().item()
    mid_occ = freq[mid_mask].sum().item()
    tail_occ = freq[tail_mask].sum().item()

    head_tokens = int(head_mask.sum().item())
    mid_tokens = int(mid_mask.sum().item())
    tail_tokens = int(tail_mask.sum().item())

    zero_freq_tokens = int((freq == 0).sum().item())

    return {
        "total_tokens_from_table": int(total),
        "head_cutoff_idx": int(head_cutoff_idx),
        "mid_cutoff_idx": int(mid_cutoff_idx),
        "head_min_freq": int(head_min_freq),
        "mid_min_freq": int(mid_min_freq),
        "head_tokens": head_tokens,
        "mid_tokens": mid_tokens,
        "tail_tokens": tail_tokens,
        "head_occ": int(head_occ),
        "mid_occ": int(mid_occ),
        "tail_occ": int(tail_occ),
        "head_occ_pct": 100.0 * head_occ / total if total > 0 else float("nan"),
        "mid_occ_pct": 100.0 * mid_occ / total if total > 0 else float("nan"),
        "tail_occ_pct": 100.0 * tail_occ / total if total > 0 else float("nan"),
        "zero_freq_tokens": zero_freq_tokens,
        "zero_freq_pct_vocab": 100.0 * zero_freq_tokens / len(freq) if len(freq) > 0 else float("nan"),
    }


def scan_and_recount(files, vocab_size, dtype, chunk_size, file_format="headered"):
    recount = torch.zeros(vocab_size, dtype=torch.long)

    total_raw_tokens = 0
    total_in_range_tokens = 0
    total_out_of_range_tokens = 0

    global_min_token = None
    global_max_token = None
    files_with_out_of_range = []

    per_file_summary = []

    for filepath in files:
        tokens = load_headered_token_shard(filepath) if file_format == "headered" else np.memmap(filepath, dtype=dtype, mode="r")
        n = len(tokens)
        total_raw_tokens += n

        file_in_range = 0
        file_out_of_range = 0
        file_min = None
        file_max = None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = np.asarray(tokens[start:end])

            if chunk.size == 0:
                continue

            chunk_min = int(chunk.min())
            chunk_max = int(chunk.max())

            file_min = chunk_min if file_min is None else min(file_min, chunk_min)
            file_max = chunk_max if file_max is None else max(file_max, chunk_max)

            global_min_token = chunk_min if global_min_token is None else min(global_min_token, chunk_min)
            global_max_token = chunk_max if global_max_token is None else max(global_max_token, chunk_max)

            in_range_mask = chunk < vocab_size
            in_range_chunk = chunk[in_range_mask]
            out_count = int((~in_range_mask).sum())

            if in_range_chunk.size > 0:
                counts = np.bincount(in_range_chunk, minlength=vocab_size)
                recount += torch.from_numpy(counts.astype(np.int64))
                in_count = int(in_range_chunk.size)
            else:
                in_count = 0

            file_in_range += in_count
            file_out_of_range += out_count

        total_in_range_tokens += file_in_range
        total_out_of_range_tokens += file_out_of_range

        if file_out_of_range > 0:
            files_with_out_of_range.append(filepath)

        per_file_summary.append({
            "file": filepath,
            "raw_tokens": int(n),
            "in_range_tokens": int(file_in_range),
            "out_of_range_tokens": int(file_out_of_range),
            "min_token": None if file_min is None else int(file_min),
            "max_token": None if file_max is None else int(file_max),
        })

    return {
        "recount_freq": recount,
        "total_raw_tokens": int(total_raw_tokens),
        "total_in_range_tokens": int(total_in_range_tokens),
        "total_out_of_range_tokens": int(total_out_of_range_tokens),
        "global_min_token": None if global_min_token is None else int(global_min_token),
        "global_max_token": None if global_max_token is None else int(global_max_token),
        "files_with_out_of_range": files_with_out_of_range,
        "per_file_summary": per_file_summary,
    }


def compare_tables(saved_freq: torch.Tensor, recount_freq: torch.Tensor):
    diff = saved_freq.to(torch.long) - recount_freq.to(torch.long)
    abs_diff = diff.abs()

    mismatched_ids = torch.nonzero(diff != 0, as_tuple=False).squeeze(-1)
    num_mismatched_ids = int(mismatched_ids.numel())
    max_abs_diff = int(abs_diff.max().item()) if abs_diff.numel() > 0 else 0
    l1_diff = int(abs_diff.sum().item())

    top_k = min(20, num_mismatched_ids)
    top_mismatches = []
    if top_k > 0:
        vals, idxs = torch.topk(abs_diff, k=top_k)
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            if v == 0:
                continue
            top_mismatches.append({
                "token_id": int(idx),
                "saved": int(saved_freq[idx].item()),
                "recount": int(recount_freq[idx].item()),
                "abs_diff": int(v),
            })

    return {
        "saved_sum": int(saved_freq.sum().item()),
        "recount_sum": int(recount_freq.sum().item()),
        "num_mismatched_token_ids": num_mismatched_ids,
        "max_abs_diff": max_abs_diff,
        "l1_diff": l1_diff,
        "top_mismatches": top_mismatches,
        "saved_sha256": sha256_of_tensor(saved_freq),
        "recount_sha256": sha256_of_tensor(recount_freq),
    }


def verdicts(vocab_size, saved_freq, bucket_stats, scan_stats, cmp_stats):
    checks = []

    checks.append((
        "frequency_table_length_matches_vocab",
        len(saved_freq) == vocab_size,
        f"len(freq)={len(saved_freq)}, vocab_size={vocab_size}"
    ))

    checks.append((
        "saved_sum_matches_recount_sum",
        cmp_stats["saved_sum"] == cmp_stats["recount_sum"],
        f"saved_sum={cmp_stats['saved_sum']}, recount_sum={cmp_stats['recount_sum']}"
    ))

    checks.append((
        "saved_sum_matches_in_range_token_count",
        cmp_stats["saved_sum"] == scan_stats["total_in_range_tokens"],
        f"saved_sum={cmp_stats['saved_sum']}, in_range_tokens={scan_stats['total_in_range_tokens']}"
    ))

    checks.append((
        "no_out_of_range_token_ids",
        scan_stats["total_out_of_range_tokens"] == 0,
        f"out_of_range_tokens={scan_stats['total_out_of_range_tokens']}, "
        f"global_max_token={scan_stats['global_max_token']}, vocab_size={vocab_size}"
    ))

    checks.append((
        "exact_frequency_table_match",
        cmp_stats["num_mismatched_token_ids"] == 0,
        f"mismatched_token_ids={cmp_stats['num_mismatched_token_ids']}, "
        f"l1_diff={cmp_stats['l1_diff']}, max_abs_diff={cmp_stats['max_abs_diff']}"
    ))

    target = {"head": 33.0, "mid": 34.0, "tail": 33.0}
    realized = {
        "head": bucket_stats["head_occ_pct"],
        "mid": bucket_stats["mid_occ_pct"],
        "tail": bucket_stats["tail_occ_pct"],
    }
    max_deviation = max(abs(realized[k] - target[k]) for k in target)
    checks.append((
        "bucket_split_close_to_target",
        max_deviation <= 2.0,
        f"HEAD={realized['head']:.2f}%, MID={realized['mid']:.2f}%, "
        f"TAIL={realized['tail']:.2f}%, max_deviation={max_deviation:.2f}%"
    ))

    return checks


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Audit a saved token frequency table against raw tokenized .bin files."
    )
    parser.add_argument("--freq_file", type=str, required=True,
                        help="Path to saved token frequency table (.npy preferred, .pt supported)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data_dir", type=str,
                             help="Directory containing tokenized .bin files")
    input_group.add_argument("--data_glob", type=str,
                             help="Glob pattern for tokenized .bin files")
    parser.add_argument("--vocab_size", type=int, default=50304,
                        help="Vocabulary size used by the model/tokenizer")
    parser.add_argument("--format", type=str, default="headered", choices=["headered", "raw"],
                        help="Token file format. headered = 256 int32 header + uint16 tokens.")
    parser.add_argument("--dtype", type=str, default="uint16",
                        choices=["uint16", "uint32", "int32", "int64"],
                        help="Underlying dtype of token files")
    parser.add_argument("--chunk_size", type=int, default=10_000_000,
                        help="Number of tokens per recount chunk")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Optional path to save audit report as JSON")
    args = parser.parse_args()

    np_dtype = getattr(np, args.dtype)

    print_section("STEP 1: FIND DATA FILES")
    files, matched_pattern, tried_patterns = find_data_files(args.data_dir, args.data_glob)
    print("Tried patterns:")
    for p in tried_patterns:
        print(f"  - {p}")
    if not files:
        print("\nFAIL: No files found.")
        return 1

    print(f"\nMatched pattern: {matched_pattern}")
    print(f"Found {len(files)} files")
    for f in files[:10]:
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    print_section("STEP 2: LOAD SAVED FREQUENCY TABLE")
    try:
        freq = load_frequency_vector(args.freq_file, vocab_size=args.vocab_size)
    except Exception as exc:
        print(f"FAIL: could not load {args.freq_file}: {exc}")
        return 1
    print(f"freq_file        : {args.freq_file}")
    print(f"tensor_shape     : {tuple(freq.shape)}")
    print(f"tensor_dtype     : {freq.dtype}")
    print(f"vocab_size(arg)  : {args.vocab_size}")
    print(f"freq_sum         : {int(freq.sum().item())}")
    print(f"freq_min         : {int(freq.min().item()) if len(freq) > 0 else 'NA'}")
    print(f"freq_max         : {int(freq.max().item()) if len(freq) > 0 else 'NA'}")
    print(f"freq_sha256      : {sha256_of_tensor(freq)}")

    print_section("STEP 3: RECOMPUTE REALIZED HEAD/MID/TAIL SPLIT FROM SAVED TABLE")
    bucket = compute_bucket_stats(freq)
    print(f"head_min_freq    : {bucket['head_min_freq']}")
    print(f"mid_min_freq     : {bucket['mid_min_freq']}")
    print(f"HEAD occurrences : {bucket['head_occ']} ({bucket['head_occ_pct']:.2f}%)")
    print(f"MID occurrences  : {bucket['mid_occ']} ({bucket['mid_occ_pct']:.2f}%)")
    print(f"TAIL occurrences : {bucket['tail_occ']} ({bucket['tail_occ_pct']:.2f}%)")
    print(f"HEAD token IDs   : {bucket['head_tokens']}")
    print(f"MID token IDs    : {bucket['mid_tokens']}")
    print(f"TAIL token IDs   : {bucket['tail_tokens']}")
    print(f"Zero-freq tokens : {bucket['zero_freq_tokens']} ({bucket['zero_freq_pct_vocab']:.2f}% of vocab)")

    print_section("STEP 4: SCAN RAW FILES AND EXACTLY RECOUNT FREQUENCIES")
    print(f"Using dtype={args.dtype}, chunk_size={args.chunk_size}")
    scan = scan_and_recount(files, args.vocab_size, np_dtype, args.chunk_size, args.format)
    recount = scan["recount_freq"]

    print(f"total_raw_tokens        : {scan['total_raw_tokens']}")
    print(f"total_in_range_tokens   : {scan['total_in_range_tokens']}")
    print(f"total_out_of_range      : {scan['total_out_of_range_tokens']}")
    print(f"global_min_token_seen   : {scan['global_min_token']}")
    print(f"global_max_token_seen   : {scan['global_max_token']}")
    print(f"files_with_out_of_range : {len(scan['files_with_out_of_range'])}")
    if scan["files_with_out_of_range"]:
        for f in scan["files_with_out_of_range"][:10]:
            print(f"  - {f}")
        if len(scan["files_with_out_of_range"]) > 10:
            print(f"  ... and {len(scan['files_with_out_of_range']) - 10} more")

    print_section("STEP 5: COMPARE SAVED TABLE VS FULL RECOUNT")
    cmp_stats = compare_tables(freq, recount)
    print(f"saved_sum                : {cmp_stats['saved_sum']}")
    print(f"recount_sum              : {cmp_stats['recount_sum']}")
    print(f"num_mismatched_token_ids : {cmp_stats['num_mismatched_token_ids']}")
    print(f"l1_diff                  : {cmp_stats['l1_diff']}")
    print(f"max_abs_diff             : {cmp_stats['max_abs_diff']}")
    print(f"saved_sha256             : {cmp_stats['saved_sha256']}")
    print(f"recount_sha256           : {cmp_stats['recount_sha256']}")

    if cmp_stats["top_mismatches"]:
        print("\nTop mismatches:")
        for row in cmp_stats["top_mismatches"]:
            print(
                f"  token_id={row['token_id']:<8d} "
                f"saved={row['saved']:<12d} "
                f"recount={row['recount']:<12d} "
                f"abs_diff={row['abs_diff']}"
            )

    print_section("STEP 6: VERDICTS")
    checks = verdicts(args.vocab_size, freq, bucket, scan, cmp_stats)
    all_pass = True
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        all_pass = all_pass and ok

    print("\nOverall result:", "PASS" if all_pass else "FAIL/WARN")

    report = {
        "args": vars(args),
        "matched_pattern": matched_pattern,
        "files_found": files,
        "saved_table": {
            "shape": list(freq.shape),
            "sum": int(freq.sum().item()),
            "sha256": sha256_of_tensor(freq),
        },
        "bucket_stats": bucket,
        "scan_stats": {
            "total_raw_tokens": scan["total_raw_tokens"],
            "total_in_range_tokens": scan["total_in_range_tokens"],
            "total_out_of_range_tokens": scan["total_out_of_range_tokens"],
            "global_min_token": scan["global_min_token"],
            "global_max_token": scan["global_max_token"],
            "files_with_out_of_range": scan["files_with_out_of_range"],
        },
        "comparison": cmp_stats,
        "checks": [
            {"name": name, "ok": ok, "message": msg}
            for (name, ok, msg) in checks
        ],
        "overall_pass": all_pass,
    }

    if args.json_out is not None:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to: {out}")

    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())