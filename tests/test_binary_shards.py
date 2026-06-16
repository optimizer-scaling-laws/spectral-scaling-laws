from pathlib import Path

import numpy as np
import pytest

from optimizer_ssl.data.binary_shards import HEADER_SIZE_INTS, MAGIC, VERSION, load_headered_token_shard


def write_test_shard(path: Path, tokens):
    header = np.zeros(HEADER_SIZE_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with path.open("wb") as f:
        header.tofile(f)
        np.asarray(tokens, dtype=np.uint16).tofile(f)


def test_load_headered_token_shard(tmp_path):
    path = tmp_path / "fineweb_train_000001.bin"
    write_test_shard(path, [1, 2, 3, 4])
    arr = load_headered_token_shard(path)
    assert arr.dtype == np.uint16
    assert arr.tolist() == [1, 2, 3, 4]


def test_load_headered_token_shard_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    header = np.zeros(HEADER_SIZE_INTS, dtype=np.int32)
    header[0] = 123
    header[1] = VERSION
    header[2] = 1
    with path.open("wb") as f:
        header.tofile(f)
        np.asarray([1], dtype=np.uint16).tofile(f)
    with pytest.raises(ValueError, match="Bad magic"):
        load_headered_token_shard(path)
