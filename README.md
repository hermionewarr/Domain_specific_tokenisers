# Specialised or Generic? Effect of Vocabulary and Tokenisation for Radiology Language Models

Code to train the tokenisers and the radiology language-model experiments from the paper. Report text and trained tokeniser files are not in this repository. Tokeniser folders will be released separately; point `SCRIBE_VOCAB_DIR` at that download.

## Setup

```bash
pip install -r requirements.txt
cp paths.example.env paths.env
```

Edit `paths.env` if your reports are not under `data/` in this repository. The language-model script reads that file, then any variables already set in the environment.

## Data

Place CSVs in these directories (or set the matching variables):

| Dataset | Directory | Files |
| --- | --- | --- |
| OUH | `data/ouh/` | `reports_sections_train.csv`, `reports_sections_val.csv`, `reports_sections_test.csv`. For a full-corpus text dump, also `reports_sections_split.csv`. |
| MIMIC-CXR | `data/mimic/` | `train.csv`, `valid.csv`, `test.csv` |
| CT-RATE | `data/ct-rate/` | `train_reports_ctrate.csv`, `val_reports_ctrate.csv`, `test_reports_ctrate.csv` |
| PubMed | `data/pubmed/pubmed_abstracts.jsonl` | JSONL with an `abstract` field |

OUH rows need `Findings`, `Conclusion`, `patient_id`, and `redacted_report`. MIMIC rows need `findings`, `impression`, `subject_id`, and `reference_report`. CT-RATE rows can use `Findings_EN` and `Impressions_EN`; a `patient_id` column is added from `VolumeName` when that is the id column, and `redacted_report` is filled from the findings and conclusion when it is absent.

OUH reports cannot be redistributed. MIMIC-CXR and CT-RATE require their own access agreements.

## Train tokenisers

From this directory:

```bash
python vocab_training/run_create_vocabs.py          # print the jobs
python vocab_training/run_create_vocabs.py --run    # train them
python vocab_training/run_create_vocabs.py --only bpe_MIMIC
```

What that trains:

- A findings-plus-conclusion text dump for each dataset (`data/text/ouhfcreport_text.txt`, `MIMICfcreport_text.txt`, `ct-ratefcreport_text.txt`). Combined tokenisers read these files.
- Domain BPE and WordPiece on the training split, with whitespace and punctuation pre-tokenisation and no per-digit split. The cap is 30,000 and the minimum frequency is 3. The saved folder uses the realised size, for example `vocabs/transformer_tokeniser_bpe9061MIMICfc_train_no_digit`.
- PubMed BPE and WordPiece at 10,000 and 30,000, plus a byte-level PubMed BPE at 28,896 (`transformer_tokeniser_bpe28896pubmed_byte`).
- PubMed mixed with each radiology corpus at 1:3, 1:1, and 3:1, for both BPE and WordPiece, cap 30,000.

The BPE 1-1 language-model runs load a folder whose name ends in `_1-1_V1`. Retraining writes the name without `_V1`. Rename that folder before those runs.

One dataset at a time:

```bash
python vocab_training/CreateVocab.py --data_source MIMIC --tok_type bpe --extra_info fc_train --pre_tok no_digit
python vocab_training/CreatePubMedVocab.py --tok_type bpe --vocab_size 30000
python vocab_training/CreatePubMedVocab.py --tok_type bpe --vocab_size 28896 --byte_level
python vocab_training/CreatePubMedVocabCombinedDomainSpecific.py --tok_type bpe --vocab_size 30000 --domain_specifc_data_source MIMIC --ratio 50
```

## Train and evaluate the language model

The model is a small decoder trained to generate the conclusion from the findings. Each run trains, generates test reports, and scores BLEU, METEOR, and ROUGE. MIMIC then uses the CheXpert labeler, CT-RATE uses the CT-CLIP report classifier, and OUH uses CXR-BERT embedding similarity. Those clinical tools are optional and only imported when that dataset is evaluated.

```bash
python scribe/run_experiments.py                 # print every paper run
python scribe/run_experiments.py --only bpe_mimic
python scribe/run_experiments.py --run --only bpe_mimic --no-wandb
python scribe/main.py --run bpe_mimic --tokeniser PTCTBPE --vocab_size 9061 --data_source MIMIC \
  --train_data train.csv --val_data valid.csv --test_data test.csv
```

While training, `scribe/main.py` sets the context length and batch size from the training reports, so the lengths in the experiment list are only the starting values.

The comparisons, each on OUH, MIMIC, and CT-RATE:

- Domain BPE and WordPiece trained on that dataset's training reports
- GPT-2 and GPT-4 tokenisers
- PubMedBERT and BioMedLM tokenisers
- PubMed BPE (10k, 30k, and the 28,896 byte-level vocabulary) and PubMed WordPiece (10k, 30k)
- PubMed mixed with the domain corpus at 1:3, 1:1, and 3:1, for BPE and WordPiece

Domain `vocab_size` values in `scribe/run_experiments.py` are the realised sizes from the paper's train-split vocabularies (OUH BPE 10987, MIMIC BPE 9061, CT-RATE BPE 9296, and the WordPiece sizes in that file). After retraining, use the size printed by `CreateVocab.py`.

Checkpoints and metric tables go to `runs/<run_name>/`.

## Clinical metrics that need extra installs

Set these in `paths.env` only if you want the dataset-specific clinical scores.

- MIMIC CheXpert labeler: `SCRIBE_CHEXPERT_LABELER`, and `SCRIBE_CHEXPERT_CONDA` if it lives in its own conda environment. `SCRIBE_CONDA` is the conda executable.
- CheXbert, if you call that path: `SCRIBE_EVAL_DIR` (a checkout that contains `CheXbert/`) and `SCRIBE_CHEXBERT_WEIGHTS`.
- CT-RATE: `SCRIBE_CTCLIP_TEXT_CLASSIFIER` and `SCRIBE_RADBERT_CHECKPOINT`.
- GREEN: `SCRIBE_GREEN_DIR`.
- MetaMap and RadGraph are used by `python scribe/ouh_clinical_metrics_test.py RUN_NAME`. MetaMap needs `SCRIBE_METAMAP_DIR`.

BLEU, METEOR, and ROUGE do not need those tools. NLTK will try to download `punkt_tab` the first time the evaluation module is imported.

## Cite

If you used this codebase, please cite our paper:

```bash
@InProceedings{Warr_2026_tokenise,
author="Warr, Hermione and Xu, Wentian and Anthony, Harry and Ibrahim, Yasin and McGowan, Daniel R. and Kamnitsas, Konstantinos",
editor="Nguyen, Hien Van and Awasthi, Akash and Patel, Vishal M. and Le, Ngan and Zhou, Yuyin and Liu, Sheng and Zhou, S. Kevin",
title="Specialised or Generic? Tokenization Choices for Radiology Language Models",
booktitle="Emerging LLM/LMM Applications in Medical Imaging", year="2026",
publisher="Springer Nature Switzerland", address="Cham", pages="62--70", isbn="978-3-032-07502-4"
}
```
