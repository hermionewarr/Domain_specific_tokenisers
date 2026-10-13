"""Train a BPE or WordPiece tokeniser on PubMed abstracts.

--byte-level uses a byte-level pre-tokeniser and saves the folder with a
pubmed_byte suffix. The language-model loader expects that suffix for the
28,896 BioMedLM-sized vocabulary.
"""

import argparse
import sys
from pathlib import Path
from time import time

from tokenizers import Tokenizer
from tokenizers import decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PUBMED_ABSTRACTS, VOCAB_DIR

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[MASK]", "[SEP]"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tok_type", choices=["bpe", "wp"], default="bpe")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--byte_level", action="store_true")
    parser.add_argument("--pubmed_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pubmed_path = args.pubmed_path or str(PUBMED_ABSTRACTS)
    if args.tok_type == "bpe":
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(
            special_tokens=SPECIAL_TOKENS,
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            show_progress=True,
        )
    else:
        tokenizer = Tokenizer(WordPiece())
        trainer = WordPieceTrainer(
            special_tokens=SPECIAL_TOKENS,
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            show_progress=True,
        )

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    if args.byte_level:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])

    start = time()
    tokenizer.train([pubmed_path], trainer=trainer)
    print("total time taken to train the tokeniser:", time() - start)
    realised = len(tokenizer.get_vocab())
    print("Final Vocabulary Size:", realised)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
        additional_special_tokens=["[Findings]", "[Conclusion]"],
        clean_up_tokenization_spaces=True,
    )
    suffix = "pubmed_byte" if args.byte_level else "pubmed"
    # The loader looks up the requested cap (for example 30000 or 28896), not
    # the realised length, for these PubMed folders.
    save_dir = VOCAB_DIR / f"transformer_tokeniser_{args.tok_type}{args.vocab_size}{suffix}"
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(save_dir)
    print("saved", save_dir)


if __name__ == "__main__":
    main()
