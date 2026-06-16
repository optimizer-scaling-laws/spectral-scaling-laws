from pathlib import Path
from optimizer_ssl.data.frequency_table import compute_bucket_stats
from optimizer_ssl.spectra.frequency_metrics import load_frequency_vector


def test_released_token_frequency_table():
    freq_path = Path("results/processed/token_frequencies.npy")
    assert freq_path.exists()
    freq = load_frequency_vector(freq_path)
    stats = compute_bucket_stats(freq)
    assert stats["vocab_size"] == 50304
    assert stats["total_tokens"] == 10255376574
    assert stats["zero_frequency_tokens"] == 203
    total = stats["total_tokens"]
    assert abs(100 * stats["buckets"]["head"]["num_occurrences"] / total - 33.0) < 1.0
    assert abs(100 * stats["buckets"]["mid"]["num_occurrences"] / total - 34.0) < 1.0
    assert abs(100 * stats["buckets"]["tail"]["num_occurrences"] / total - 33.0) < 1.0
