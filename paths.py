"""Paths for data, trained tokenisers, and run outputs.

Defaults live under this repository. Override them with environment variables
or by copying paths.example.env to paths.env in this directory. Existing
environment variables take precedence over paths.env.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file(REPO_ROOT / "paths.env")


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


DATA_ROOT = _path("SCRIBE_DATA_ROOT", REPO_ROOT / "data")
OUH_DIR = _path("SCRIBE_OUH_DIR", DATA_ROOT / "ouh")
MIMIC_DIR = _path("SCRIBE_MIMIC_DIR", DATA_ROOT / "mimic")
CTRATE_DIR = _path("SCRIBE_CTRATE_DIR", DATA_ROOT / "ct-rate")
PUBMED_ABSTRACTS = _path("SCRIBE_PUBMED_ABSTRACTS", DATA_ROOT / "pubmed" / "pubmed_abstracts.jsonl")
TEXT_DIR = _path("SCRIBE_TEXT_DIR", DATA_ROOT / "text")
VOCAB_DIR = _path("SCRIBE_VOCAB_DIR", REPO_ROOT / "vocabs")
RUNS_DIR = _path("SCRIBE_RUNS_DIR", REPO_ROOT / "runs")

# Optional external tools used only by clinical metrics.
CHEXPERT_LABELER = os.environ.get("SCRIBE_CHEXPERT_LABELER", "").strip()
CHEXBERT_WEIGHTS = os.environ.get("SCRIBE_CHEXBERT_WEIGHTS", "").strip()
CHEXPERT_CONDA = os.environ.get("SCRIBE_CHEXPERT_CONDA", "chexpert-label").strip()
CONDA_BIN = os.environ.get("SCRIBE_CONDA", "conda").strip() or "conda"
GREEN_DIR = os.environ.get("SCRIBE_GREEN_DIR", "").strip()
EVAL_DIR = os.environ.get("SCRIBE_EVAL_DIR", "").strip()
CTCLIP_TEXT_CLASSIFIER = os.environ.get("SCRIBE_CTCLIP_TEXT_CLASSIFIER", "").strip()
RADBERT_CHECKPOINT = os.environ.get("SCRIBE_RADBERT_CHECKPOINT", "").strip()
METAMAP_DIR = os.environ.get("SCRIBE_METAMAP_DIR", "").strip()


def report_dir(data_source: str) -> Path:
    mapping = {
        "ouh": OUH_DIR,
        "MIMIC": MIMIC_DIR,
        "ct-rate": CTRATE_DIR,
    }
    try:
        return mapping[data_source]
    except KeyError as exc:
        raise ValueError(
            f"Unknown data_source {data_source!r}. Choose from ouh, MIMIC, ct-rate."
        ) from exc


def generated_reports_csv(run: str) -> Path:
    return RUNS_DIR / run / "generated_reports" / f"generated_reports_{run}.csv"


def domain_text_file(data_source: str, extra_info: str) -> Path:
    """Text dump written by vocab training and read by the combined tokeniser.

    extra_info is the suffix before it is prefixed with the dataset name,
    for example 'fc' or 'fc_train'. The file is named
    '{data_source}{extra_info}report_text.txt'.
    """
    return TEXT_DIR / f"{data_source}{extra_info}report_text.txt"
