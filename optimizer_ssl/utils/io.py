import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
