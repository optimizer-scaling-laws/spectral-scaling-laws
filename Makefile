install:
	pip install -e ".[dev]"

install-metrics:
	pip install -e ".[metrics]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/

lint:
	python -m ruff check optimizer_ssl tests scripts

format:
	python -m black optimizer_ssl tests scripts

prepare-data:
	bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh

train-160m-example:
	bash scripts/train/train_160m_example.sh

train-350m-example:
	bash scripts/train/train_350m_example.sh

tiny-debug:
	bash scripts/train/train_tiny_debug.sh

figures:
	bash scripts/reproduce/reproduce_main_results_from_processed.sh results/processed results/figures

reproduce: figures

check:
	python -m ruff check optimizer_ssl tests scripts
	python -m compileall -q optimizer_ssl third_party/dion/dion scripts tests examples
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/
	bash -n scripts/train/*.sh scripts/preprocess/*.sh scripts/reproduce/*.sh

probe-synthetic:
	python examples/probe_synthetic_activations.py
