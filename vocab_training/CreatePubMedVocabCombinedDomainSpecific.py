import pandas as pd
import subprocess
import torch
print(torch.version.cuda)
import argparse
import transformers
print(transformers.__version__)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE 
#import sentencepiece as spm
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits
from tokenizers import normalizers
from time import time
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset, concatenate_datasets, interleave_datasets

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PUBMED_ABSTRACTS, TEXT_DIR, VOCAB_DIR

pubmed_data_path = str(PUBMED_ABSTRACTS)
vocab_path = str(VOCAB_DIR) + "/"
    

### parameters ###
def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='1', help='the run name.')
    parser.add_argument('--data_source', type=str, default='pubmed', help='the general/medical source of the data.')
    parser.add_argument('--vocab_size', type=int, default=30000, help='the size of the vocabulary')
    
    parser.add_argument('--tok_type', type=str, default='bpe', help='the tokeniser to use bpe or wp')
    parser.add_argument('--max_length', type=int, default=1024, help='the maximum length of the input sequence') #switched from 256 on the 28th nov

    parser.add_argument('--special_tokens', type=list, default=["[UNK]", "[BOS]", "[PAD]", "[SEP]", "[MASK]"], help='the special tokens to use')
    parser.add_argument('--batch_size', type=int, default=100000, help='the number of samples for a batch')
    parser.add_argument('--min_freq', type=int, default=5, help='the minimum frequency of a token to be included in the vocab')
    parser.add_argument('--domain_specifc_data', type=str, default='True', help='whether to use domain specific data')
    parser.add_argument('--domain_specifc_data_source', type=str, default="ct-rate", help='the source of the domain specific data')
    
    parser.add_argument('--ratio', type=int, default=50, help='the ratio of domain specific data to pubmed data. default 50:50, 25 for 1-3 and 75 for 3-1 ')
    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    tok_type = args.tok_type
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    batch_size = args.batch_size
    min_freq = args.min_freq
    domain_specifc_data = args.domain_specifc_data
    domain_specifc_data_source = args.domain_specifc_data_source
    data_source = args.data_source
    run = args.run
    ratio = args.ratio
    #print args in a list
    print(args)
    ### load pubmed data ###
    pubmed_data = load_dataset("json", data_files = pubmed_data_path, split = "train")#, streaming = True
    pubmed_dataset = pubmed_data.rename_column("abstract", "text")


    # if domain specific data to be used
    if domain_specifc_data_source is not None:
        text_name = {
            "ouh": "ouhfcreport_text.txt",
            "MIMIC": "MIMICfcreport_text.txt",
            "ct-rate": "ct-ratefcreport_text.txt",
        }[domain_specifc_data_source]
        print(f"loading {domain_specifc_data_source} data")
        domain_specifc_data_path = str(TEXT_DIR / text_name)
        
        #for any domain specific data
        domain_specifc_data = load_dataset("text", data_files = domain_specifc_data_path, split="train")
        print(len(pubmed_dataset) /(len(domain_specifc_data)))
        print(len(pubmed_dataset))
        print(len(domain_specifc_data))
        oversample_factor = max(1, len(pubmed_dataset) // (10*len(domain_specifc_data))) #10*len(domain_specifc_data) when we use the ful pubmed dataset
        print(f"Oversampling factor: {oversample_factor}")

        balance_dataset = balance_datasets(domain_specifc_data, pubmed_dataset, ratio)

    tokenizer, trainer = tok_intialise(vocab_size, min_freq, tok_type)
    tokenizer = pre_tok(tokenizer)
    first_sample = next(batch_iterator(batch_size=5, dataset=balance_dataset))
    #print("First sample passed to tokenizer:", first_sample[0])
    print("Type:", type(first_sample[0]))  # Should be <class 'str'>

    print(tokenizer.pre_tokenizer.pre_tokenize_str(first_sample[0][:50]))
    tokenizer = train_tok(tokenizer, batch_size, trainer, balance_dataset, domain_specifc_data)
    save_tok(tokenizer, args)

    return

def balance_datasets(domain_specifc_data, pubmed_dataset, ratio):       
    # Repeat the radiology dataset multiple times
    #oversampled_radiology = concatenate_datasets([domain_specifc_data] * oversample_factor).shuffle()
    start_time = time()
    # Interleave with weighted sampling
    #balanced_dataset = interleave_datasets([oversampled_radiology, pubmed_dataset], probabilities=[0.5, 0.5], stopping_stratergy = "first_exhausted")
    if ratio == 50:
        print("50-50")
        probs = [0.5, 0.5]
    elif ratio == 25:
        print("1:3")
        probs = [0.25, 0.75]
    elif ratio == 75:
        print("3:1")
        probs = [0.75, 0.25]
    balanced_dataset = interleave_datasets([domain_specifc_data, pubmed_dataset], probabilities=probs, stopping_strategy = "all_exhausted")
    #balanced_dataset = interleave_datasets([domain_specifc_data, pubmed_dataset], probabilities=[0.25, 0.75], stopping_stratergy = "all_exhausted")
    print(len(balanced_dataset))
    #shuffle the dataset
    balanced_dataset = balanced_dataset.shuffle()
    print(f"Time taken to balance the dataset: {time() - start_time:.2f} s")
    return balanced_dataset


def batch_iterator(batch_size, dataset):
    batch = []
    for sample in dataset:
        batch.append(sample["text"])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch  # Yield remaining data if any


#the tokenizer
def tok_intialise(vocab_size, min_freq, tok_type):
    if tok_type == "bpe":
        print("using bpe tokenizer")
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[MASK]", "[SEP]"], 
                            vocab_size=vocab_size, min_frequency=min_freq, show_progress=True)#initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    elif tok_type == "wp":
        print("using wp tokenizer")
        tokenizer = Tokenizer(WordPiece())
        trainer = WordPieceTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[MASK]", "[SEP]"], 
                                vocab_size=vocab_size, min_frequency=min_freq, show_progress=True)
    return tokenizer, trainer

#pretokenisation
def pre_tok(tokenizer):
    #normalise
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
    #tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True) # biomedLM
    #tokenizer.decoder = decoders.ByteLevel()
    #tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    return tokenizer

#train the tokenizer
def train_tok(tokenizer, batch_size, trainer, balanced_dataset, domain_specifc_data_source):
    #length = len(pubmed_data['train']) # doesnt work with streaming
    #print(length)
    if domain_specifc_data_source is not None:
        print("Training Tokeniser with both pubmed and domain specific data")
        tokenizer.train_from_iterator(batch_iterator(batch_size, balanced_dataset), trainer=trainer, length=len(balanced_dataset))
    else:
        tokenizer.train([pubmed_data_path], trainer=trainer) 

    #print("total time taken to train the tokeniser:", time()-start_time)
    encoding = tokenizer.encode("Testing tokeniser: Blood Glucose levels are normal")
    print(encoding.tokens, encoding.ids)
    decoding = tokenizer.decode(encoding.ids)
    print(decoding)

    vocab_size = len(tokenizer.get_vocab())
    print("Final Vocabulary Size:", vocab_size)
    #tokenizer.save(path+f"DataInfo/vocab/vocab_{tok_type}{vocab_size}combined_pubmed_{domain_specifc_data_source}.json")

    return tokenizer

### Save the Tokeniser ###
def save_tok(tokenizer, args):
    wrapped_tokeniser = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
        additional_special_tokens = ["[Findings]", "[Conclusion]"],
        clean_up_tokenization_spaces=True
    )

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    if args.domain_specifc_data_source is not None:
        print("saving combined tokeniser")
        if args.ratio == 50:
            label = "1-1"
        elif args.ratio == 25:
            label = "1-3"
        elif args.ratio == 75:
            label = "3-1"
        save_path = vocab_path+f"transformer_tokeniser_{args.tok_type}{args.vocab_size}combined_{args.data_source}_{args.domain_specifc_data_source}_{label}"
        print("saving to: ", save_path)
        
        wrapped_tokeniser.save_pretrained(save_path)
         
    else:
        wrapped_tokeniser.save_pretrained(vocab_path+f"transformer_tokeniser_{args.tok_type}{args.vocab_size}{args.data_source}_{args.run}")
    
    return

if __name__ == "__main__":
    main()

### Create Tokeniser from PubMed Abstracts and Articles ###

### Create Tokeniser from PubMed Abstracts, Articles and Domain specific data ###