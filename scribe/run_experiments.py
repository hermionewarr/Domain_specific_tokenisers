"""Run the paper's language-model comparisons.

Prints one command per run. Pass --run to execute them.
Pass --only to keep run names that contain that text.

During training, scribe/main.py replaces block size, max length, and batch
size using the training-report length distribution. The values here are the
starting point used for each dataset.

vocab_size for the domain tokenisers is the realised vocabulary after the
frequency cutoff, not the 30,000 cap passed to training. If you retrain, set
it to the size printed by vocab_training/CreateVocab.py. The BPE 1-1 runs load
the folder whose name ends in _1-1_V1.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN = REPO_ROOT / "scribe" / "main.py"

SPLITS = {
    "ouh": {
        "data_source": "ouh",
        "train_data": "reports_sections_train.csv",
        "val_data": "reports_sections_val.csv",
        "test_data": "reports_sections_test.csv",
        "max_length": 512,
        "batch_size": 128,
        "block_size": 512,
    },
    "MIMIC": {
        "data_source": "MIMIC",
        "train_data": "train.csv",
        "val_data": "valid.csv",
        "test_data": "test.csv",
        "max_length": 256,
        "batch_size": 128,
        "block_size": 256,
    },
    "ct-rate": {
        "data_source": "ct-rate",
        "train_data": "train_reports_ctrate.csv",
        "val_data": "val_reports_ctrate.csv",
        "test_data": "test_reports_ctrate.csv",
        "max_length": 1024,
        "batch_size": 32,
        "block_size": 1024,
    },
}

# Realised sizes of the train-split, no-individual-digit domain vocabularies.
DOMAIN_BPE = {"ouh": 10987, "MIMIC": 9061, "ct-rate": 9296}
DOMAIN_WP = {"ouh": 12065, "MIMIC": 10053, "ct-rate": 10043}


def _base(name, tokeniser, dataset):
    slug = dataset.lower()
    config = {
        "run": f"{name}_{slug}",
        "tokeniser": tokeniser,
        "epochs": 50,
        "train_mode": "True",
        "test_mode": "True",
        "eval_mode": "True",
        "wandb_log": "True",
        "extra_tok_info": "fc_train_no_digit",
        **SPLITS[dataset],
    }
    return config


def paper_experiments():
    experiments = []
    for dataset in SPLITS:
        experiments.append({**_base("bpe", "PTCTBPE", dataset), "vocab_size": DOMAIN_BPE[dataset]})
        experiments.append({**_base("wp", "PTCTWP", dataset), "vocab_size": DOMAIN_WP[dataset]})
        experiments.append({**_base("gpt2", "gpt2", dataset), "lr": 1e-5, "vocab_size": 50257})
        experiments.append({**_base("gpt4", "gpt4", dataset), "lr": 1e-5, "vocab_size": 100000})
        experiments.append({**_base("pubmedbert", "pubmedbert", dataset), "vocab_size": 30522})
        experiments.append({**_base("biomedlm", "BioMedLM", dataset), "vocab_size": 28896})
        for size in (10000, 30000, 28896):
            experiments.append({**_base(f"pubmed_bpe_{size}", "PubmedBPE", dataset), "vocab_size": size})
        for size in (10000, 30000):
            experiments.append({**_base(f"pubmed_wp_{size}", "PubmedWP", dataset), "vocab_size": size})
        for tokeniser, label in (
            ("Combined_1-3", "bpe_combined_1-3"),
            ("Combined_1-1", "bpe_combined_1-1"),
            ("Combined_3-1", "bpe_combined_3-1"),
            ("WPCombined_1-3", "wp_combined_1-3"),
            ("WPCombined_1-1", "wp_combined_1-1"),
            ("WPCombined_3-1", "wp_combined_3-1"),
        ):
            experiments.append({**_base(label, tokeniser, dataset), "vocab_size": 30000})
    return experiments


def command(config, wandb_log):
    config = dict(config)
    config["wandb_log"] = wandb_log
    args = [sys.executable, str(MAIN)]
    for key, value in config.items():
        args.extend([f"--{key}", str(value)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute the commands. Default is to print them.")
    parser.add_argument("--only", default="", help="Substring filter on the run name.")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    wandb_log = "False" if args.no_wandb else "True"
    selected = [cfg for cfg in paper_experiments() if args.only in cfg["run"]]
    if not selected:
        raise SystemExit(f"No experiments matched --only {args.only!r}")
    log_dir = REPO_ROOT / "runs" / "logs"
    for index, config in enumerate(selected, start=1):
        cmd = command(config, wandb_log)
        print(f"[{index}/{len(selected)}] {' '.join(cmd)}")
        if not args.run:
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"output_{config['run']}.txt"
        with log_path.open("w") as handle:
            process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            print(f"{config['run']} failed with return code {process.returncode}. See {log_path}")
        else:
            print(f"{config['run']} completed. Log: {log_path}")


if __name__ == "__main__":
    main()
