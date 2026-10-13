"""Train the tokenisers used by the paper experiments.

Prints commands unless --run is passed. Domain training writes a text dump
under the text directory. Combined training reads the full findings-plus-
conclusion dumps ({dataset}fcreport_text.txt), so those dumps are created first.

The domain vocabulary cap is 30,000. The folder name uses the realised size.
Copy or rename a retrained BPE 1-1 folder so its name ends in _1-1_V1 before
the language-model runs that load Combined_1-1.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DOMAIN = HERE / "CreateVocab.py"
PUBMED = HERE / "CreatePubMedVocab.py"
COMBINED = HERE / "CreatePubMedVocabCombinedDomainSpecific.py"

DATASETS = ("ouh", "MIMIC", "ct-rate")
# ratio is the percentage of radiology text. 25, 50, and 75 are the 1-3, 1-1, and 3-1 mixes.
RATIOS = (25, 50, 75)


def commands():
    jobs = []
    for data_source in DATASETS:
        jobs.append((
            f"text_{data_source}",
            [sys.executable, str(DOMAIN), "--data_source", data_source, "--extra_info", "fc", "--dump_text_only"],
        ))
        for tok_type in ("bpe", "wp"):
            jobs.append((
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
            ))
    for tok_type, size, byte in (
        ("bpe", "10000", False),
        ("wp", "10000", False),
        ("bpe", "30000", False),
        ("wp", "30000", False),
        ("bpe", "28896", True),
    ):
        cmd = [sys.executable, str(PUBMED), "--tok_type", tok_type, "--vocab_size", size, "--min_freq", "5"]
        if byte:
            cmd.append("--byte_level")
        jobs.append((f"pubmed_{tok_type}_{size}{'_byte' if byte else ''}", cmd))
    for data_source in DATASETS:
        for tok_type in ("bpe", "wp"):
            for ratio in RATIOS:
                jobs.append((
                    f"{tok_type}_{data_source}_pubmed_{ratio}",
                    [
                        sys.executable, str(COMBINED),
                        "--run", f"{tok_type}_{data_source}_{ratio}",
                        "--tok_type", tok_type,
                        "--vocab_size", "30000",
                        "--min_freq", "5",
                        "--domain_specifc_data_source", data_source,
                        "--ratio", str(ratio),
                        "--data_source", "pubmed",
                    ],
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
