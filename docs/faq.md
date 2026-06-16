# FAQ

## Why are raw FineWeb shards not included?

They are large external data artifacts and should be regenerated or downloaded separately.

## Why is `token_frequencies.npy` included?

It is a small paper artifact that makes HEAD/MID/TAIL bucket assignment reproducible without downloading all training shards.
