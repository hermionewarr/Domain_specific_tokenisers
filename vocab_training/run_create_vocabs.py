"""Train the BPE tokenisers used by the paper experiments.

Prints commands unless --run is passed. Domain training uses the training-split
findings and conclusions. The vocabulary cap is 30,000 and the saved folder
uses the realised size after the frequency cutoff.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DOMAIN = HERE / "CreateVocab.py"
PUBMED = HERE / "CreatePubMedVocab.py"

DATASETS = ("ouh", "MIMIC", "ct-rate")


def _domain_job(data_source, tok_type):
    return (
        f"{tok_type}_{data_source}",
        [
            sys.executable, str(DOMAIN),
            "--data_source", data_source,
            "--tok_type", tok_type,
            "--vocab_size", "30000",
            "--min_freq", "3",
            "--extra_info", "fc_train",
            "--pre_tok", "no_digit",
        ],
    )


def commands():
    jobs = [_domain_job(data_source, "bpe") for data_source in DATASETS]
    # Example WordPiece run. Uncomment to train it the same way as the BPE jobs.
    # jobs.append(_domain_job("MIMIC", "wp"))
    jobs.append((
        "pubmed_bpe_30000",
        [sys.executable, str(PUBMED), "--tok_type", "bpe", "--vocab_size", "30000", "--min_freq", "5"],
    ))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute the commands. Default is to print them.")
    parser.add_argument("--only", default="", help="Substring filter on the job name.")
    args = parser.parse_args()
    selected = [(name, cmd) for name, cmd in commands() if args.only in name]
    if not selected:
        raise SystemExit(f"No vocab jobs matched --only {args.only!r}")
    log_dir = REPO_ROOT / "vocab_logs"
    for index, (name, cmd) in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {name}: {' '.join(cmd)}")
        if not args.run:
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.txt"
        with log_path.open("w") as handle:
            process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            print(f"{name} failed with return code {process.returncode}. See {log_path}")
        else:
            print(f"{name} completed. Log: {log_path}")


if __name__ == "__main__":
    main()
