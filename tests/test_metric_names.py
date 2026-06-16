from pathlib import Path


def test_removed_metric_names_not_in_active_repo():
    root = Path(__file__).resolve().parents[1]
    active_paths = [root / "optimizer_ssl", root / "configs", root / "scripts", root / "docs"]
    banned = [
        "EEE" + "_pre", "EEE" + "_post", "compute" + "_eee",
        "JS" + "=", "compute" + "_js",
        "Re" + "nyi", "re" + "nyi", "R" + "ényi",
        "enable" + "_" + "alpha" + "_" + "sweep", "alpha" + "_" + "sweep" + "_" + "values",
        "track" + "_symmetry",
        "SE" + "_pre", "PR" + "_pre",
    ]
    offenders = []
    for base in active_paths:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".sh"}:
                if path.name == "test_metric_names.py" or "log_schema" in str(path):
                    continue
                text = path.read_text(errors="ignore")
                for token in banned:
                    if token in text:
                        offenders.append(f"{path.relative_to(root)} contains {token}")
    assert not offenders, "\n".join(offenders[:50])
