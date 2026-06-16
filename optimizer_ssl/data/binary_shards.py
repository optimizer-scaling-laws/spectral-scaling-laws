from pathlib import Path
import numpy as np

MAGIC = 20240520
VERSION = 1
HEADER_SIZE_INTS = 256
HEADER_SIZE_BYTES = HEADER_SIZE_INTS * 4


def load_headered_token_shard(path: str | Path) -> np.memmap:
    """Load a FineWeb-style binary shard: 256 int32 header + uint16 token IDs."""
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=HEADER_SIZE_INTS)
    if len(header) != HEADER_SIZE_INTS:
        raise ValueError(f"{path} is too small to contain the expected header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"Bad magic number in {path}: got {int(header[0])}, expected {MAGIC}")
    if int(header[1]) != VERSION:
        raise ValueError(f"Unsupported shard version in {path}: got {int(header[1])}, expected {VERSION}")
    ntok = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_SIZE_BYTES, shape=(ntok,))


def read_headered_token_count(path: str | Path) -> int:
    """Return token count from a FineWeb-style binary shard header."""
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=HEADER_SIZE_INTS)
    if len(header) != HEADER_SIZE_INTS:
        raise ValueError(f"{path} is too small to contain the expected header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"Bad magic number in {path}: got {int(header[0])}, expected {MAGIC}")
    if int(header[1]) != VERSION:
        raise ValueError(f"Unsupported shard version in {path}: got {int(header[1])}, expected {VERSION}")
    return int(header[2])
