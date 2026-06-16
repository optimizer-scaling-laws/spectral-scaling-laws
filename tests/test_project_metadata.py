from pathlib import Path
import tomllib

import yaml


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


def test_citation_cff_matches_paper_and_repository():
    data = yaml.safe_load(Path("CITATION.cff").read_text())
    authors = {f"{a['given-names']} {a['family-names']}" for a in data["authors"]}
    assert authors == EXPECTED_AUTHORS
    assert data["repository-code"] == EXPECTED_REPO
    assert data["url"] == EXPECTED_PROJECT
    preferred = data["preferred-citation"]
    assert preferred["title"] == "Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws"
    assert preferred["url"] == EXPECTED_PAPER
    ids = {(item["type"], item["value"]) for item in preferred["identifiers"]}
    assert ("doi", "10.48550/arXiv.2605.21803") in ids


def test_license_uses_paper_authors():
    text = Path("LICENSE").read_text()
    assert "Nandan Kumar Jha and Brandon Reagen" in text
    assert "Prem Bhaskar Jha" not in text
