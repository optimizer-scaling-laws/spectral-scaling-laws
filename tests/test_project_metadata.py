from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


EXPECTED_AUTHORS = {"Nandan Kumar Jha", "Brandon Reagen"}
EXPECTED_REPO = "https://github.com/optimizer-scaling-laws/spectral-scaling-laws"
EXPECTED_PAPER = "https://arxiv.org/abs/2605.21803"
EXPECTED_PROJECT = "https://optimizer-scaling-laws.github.io/"


def test_pyproject_authors_and_urls_match_paper_metadata():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    authors = {author["name"] for author in data["project"]["authors"]}
    assert authors == EXPECTED_AUTHORS
    urls = data["project"]["urls"]
    assert urls["Paper"] == EXPECTED_PAPER
    assert urls["Homepage"] == EXPECTED_PROJECT
    assert urls["Repository"] == EXPECTED_REPO


def test_license_uses_paper_authors():
    text = Path("LICENSE").read_text()
    assert "Nandan Kumar Jha and Brandon Reagen" in text
    assert "Prem Bhaskar Jha" not in text
