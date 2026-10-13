"""Train a domain BPE or WordPiece tokeniser on radiology reports.

The vocabulary cap is --vocab-size. The saved folder uses the realised size
after the minimum-frequency cutoff, which is what the language-model runs
load. Findings and conclusions from the training split are the default corpus
(--extra-info fc_train). Use --extra-info fc to dump the full findings plus
conclusions text that the combined PubMed tokenisers read.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import re
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Digits, Punctuation, Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import TEXT_DIR, VOCAB_DIR, domain_text_file, report_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_source", choices=["ouh", "MIMIC", "ct-rate"], required=True)
    parser.add_argument("--tok_type", choices=["wp", "bpe", "sp", "gpt_train"], default="bpe")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=3)
    parser.add_argument(
        "--extra_info",
        default="fc_train",
        help="fc uses all loaded reports. fc_train and fc_train50 use the training CSV.",
    )
    parser.add_argument(
        "--pre_tok",
        choices=["no_digit", "digit_indiv"],
        default="no_digit",
        help="no_digit matches the paper tokenisers. digit_indiv splits digits.",
    )
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--all_csv", default=None, help="OUH file used when --extra-info fc.")
    parser.add_argument(
        "--dump_text_only",
        action="store_true",
        help="Write the report text file and exit without training a tokeniser.",
    )
    return parser.parse_args()


def default_names(data_source):
    if data_source == "ouh":
        return "reports_sections_train.csv", "reports_sections_val.csv", "reports_sections_test.csv", "reports_sections_split.csv"
    if data_source == "MIMIC":
        return "train.csv", "valid.csv", "test.csv", None
    return "train_reports_ctrate.csv", "val_reports_ctrate.csv", "test_reports_ctrate.csv", None


def load_frames(args):
    directory = report_dir(args.data_source)
    train_name, val_name, test_name, all_name = default_names(args.data_source)
    train_path = Path(args.train_csv) if args.train_csv else directory / train_name
    train_data = pd.read_csv(train_path)
    data = train_data

    if args.data_source == "ouh" and args.extra_info == "fc":
        all_path = Path(args.all_csv) if args.all_csv else directory / all_name
        data = pd.read_csv(all_path)
    elif args.data_source == "MIMIC":
        val_path = Path(args.val_csv) if args.val_csv else directory / val_name
        test_path = Path(args.test_csv) if args.test_csv else directory / test_name
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
        data = pd.concat([train_data, val_data, test_data])
        rename = {"findings": "Findings", "impression": "Conclusion"}
        train_data = train_data.rename(columns=rename)
        data = data.rename(columns=rename)
    elif args.data_source == "ct-rate":
        if args.extra_info == "fc":
            val_path = Path(args.val_csv) if args.val_csv else directory / val_name
            data = pd.concat([train_data, pd.read_csv(val_path)])
        rename = {"Findings_EN": "Findings", "Impressions_EN": "Conclusion"}
        train_data = train_data.rename(columns=rename)
        data = data.rename(columns=rename)
    return data, train_data


def reports_for(args, data, train_data):
    if args.extra_info == "fc":
        reports = data["Findings"] + " " + data["Conclusion"]
    elif args.extra_info in ("fc_train", "fc_train50"):
        reports = train_data["Findings"] + " " + train_data["Conclusion"]
    else:
        raise ValueError("extra_info must be fc, fc_train, or fc_train50")
    return reports.dropna()


def main():
    args = parse_args()
    data, train_data = load_frames(args)
    reports = reports_for(args, data, train_data)
    tagged_info = args.data_source + args.extra_info
    text_path = domain_text_file(args.data_source, args.extra_info)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    report_list = list(reports)
    text = "\n".join(report_list)
    text_path.write_text(text)
    print(f"wrote {len(report_list)} reports to {text_path}")

    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)
    frequent_words = {word for word, count in word_counts.items() if count > 3}
    print("Total Unique Words:", len(set(words)))
    print("Total Frequent Words:", len(frequent_words))
    if args.dump_text_only:
        return

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[MASK]", "[SEP]"]
    if args.tok_type == "wp":
        model = WordPiece()
        trainer = WordPieceTrainer(
            special_tokens=special_tokens,
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            show_progress=True,
        )
    elif args.tok_type == "bpe":
        model = BPE()
        trainer = BpeTrainer(
            special_tokens=special_tokens,
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            show_progress=True,
        )
    elif args.tok_type == "sp":
        import sentencepiece as spm

        VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=str(text_path),
            model_prefix=str(VOCAB_DIR / f"sp_tokenizer{args.vocab_size}{tagged_info}"),
            vocab_size=args.vocab_size,
            model_type="unigram",
            user_defined_symbols=["[MASK]", "[SEP]"],
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[BOS]",
        )
        return
    elif args.tok_type == "gpt_train":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        def batch_iterator(dataset, batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]

        trained = tokenizer.train_new_from_iterator(batch_iterator(report_list, 1000), vocab_size=args.vocab_size)
        out = VOCAB_DIR / f"transformer_tokeniser_gpt_train{args.vocab_size}{tagged_info}_{args.pre_tok}"
        trained.save_pretrained(out)
        print("saved", out)
        return

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase()]
    )
    if args.pre_tok == "digit_indiv":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [Whitespace(), Digits(individual_digits=True), Punctuation()]
        )
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])

    tokenizer.train(files=[str(text_path)], trainer=trainer)
    realised = len(tokenizer.get_vocab())
    print("Final Vocabulary Size:", realised)

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocab_json = VOCAB_DIR / f"vocab_{args.tok_type}_m{realised}{tagged_info}.json"
    tokenizer.save(str(vocab_json))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(vocab_json),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
        additional_special_tokens=["[Findings]", "[Conclusion]"],
        clean_up_tokenization_spaces=True,
    )
    save_dir = VOCAB_DIR / f"transformer_tokeniser_{args.tok_type}{realised}{tagged_info}_{args.pre_tok}"
    fast_tokenizer.save_pretrained(save_dir)
    print("saved", save_dir)


if __name__ == "__main__":
    main()
