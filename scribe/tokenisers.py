#tokenisers to compare in this script are:
#1. WordPiece homemade
#2. BPE homemade
#3. big ol gpt2
#4. something like pubmedbert

import sys
from pathlib import Path

from transformers import PreTrainedTokenizerFast, AutoTokenizer, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import VOCAB_DIR

vocab_path = str(VOCAB_DIR) + "/"

#vocab_size = 3000
#data_source = 'ct-rate' #"ouh" # 'MIMIC', 'chexpert', 'ct-rate'
extra_info="fc_train_no_digit" #_train
#1. WordPiece homemade

#2. BPE homemade
""" fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=path+"DataInfo/vocab_bpe_m5000.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
)
fast_tokenizer.save_pretrained(path+"DataInfo/transformer_tokeniser_bpe5000") """


def ScribeTokeniser(args):
    #returns the token user inputed in the argument
    extra_info=args.extra_tok_info
    print("extra info: ", extra_info)
    if args.tokeniser == 'gpt2':
        print("Using gpt2 tokeniser")
        return GPT2Tokenizer.from_pretrained('gpt2', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]') #AutoTokenizer.from_pretrained('gpt2', pad_token = 'BOS') , bos_token ='[BOS]', eos_token = '[BOS]'
    elif args.tokeniser == 'gpt4':
        print("Using gpt4 tokeniser")
        return GPT2Tokenizer.from_pretrained('Xenova/gpt-4', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]') 
    elif args.tokeniser == 'pubmedbert':
        print("Using pubmedbert tokeniser") #uses word piece
        return AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]', sep_token='[BOS]', cls_token='[BOS]')
    elif args.tokeniser == 'cxr-bert':
        print("using microsoft cxr bert tokeniser") #https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general
        return AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-general', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]', sep_token='[BOS]', cls_token='[BOS]')
    elif args.tokeniser == 'BioMedLM':
        print("Using BioMedLM tokeniser. a pubmed one")
        return AutoTokenizer.from_pretrained('stanford-crfm/BioMedLM', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
    elif args.tokeniser == 'PTCTWP':
        print(f"Using word piece tokeniser, info {extra_info}")
        transformer_tokeniser_wp = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_wp{args.vocab_size}{args.data_source}{extra_info}', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_wp
    elif args.tokeniser == 'PTCTBPE':
        print(f"Using BPE tokeniser, info {extra_info}")
        transformer_tokeniser_bpe = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}{args.data_source}{extra_info}', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_bpe
    elif args.tokeniser == 'PubmedBPE':
        print("Using Pubmed BPE tokeniser")
        print(args.vocab_size)
        #vocab size type
        print(type(args.vocab_size))
        if args.vocab_size == 28896:
            print("using the biomedlm esque pubmed tokeniser")
            transformer_tokeniser_pmbpe = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}pubmed_byte', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        else:
            transformer_tokeniser_pmbpe = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}pubmed', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_pmbpe
    elif args.tokeniser == 'PubmedWP':
        print("Using Pubmed WP tokeniser")
        transformer_tokeniser_pmwp = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_wp{args.vocab_size}pubmed', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_pmwp
    elif args.tokeniser == 'Combined_1-3':
        print("Using Combined BPE tokeniser 1-3")
        #transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe9139MIMICfc_combined_pubmed_{args.vocab_size}', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}combined_pubmed_{args.data_source}_1-3', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'Combined_1-1':
        print("Using Combined BPE tokeniser 1-1")
        # The 1-1 BPE runs load the _V1 folder. Retraining writes the name without _V1.
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}combined_pubmed_{args.data_source}_1-1_V1', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'Combined_3-1':
        print("Using Combined BPE tokeniser 3-1")
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_bpe{args.vocab_size}combined_pubmed_{args.data_source}_3-1', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'WPCombined_1-1':
        print("Using WP Combined tokeniser 1-1")
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_wp{args.vocab_size}combined_pubmed_{args.data_source}_1-1', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'WPCombined_1-3':
        print("Using WP Combined tokeniser 1-3")
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_wp{args.vocab_size}combined_pubmed_{args.data_source}_1-3', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'WPCombined_3-1':
        print("Using WP Combined tokeniser 3-1")
        transformer_tokeniser_combined = AutoTokenizer.from_pretrained(vocab_path+f'transformer_tokeniser_wp{args.vocab_size}combined_pubmed_{args.data_source}_3-1', trucation=True, max_length=args.max_length, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        return transformer_tokeniser_combined
    elif args.tokeniser == 'sp':
        print("Using SentencePiece tokeniser") #{tok_type}_tokenizer{vocab_size}
        #transformer_tokeniser_sp = AutoTokenizer.from_pretrained(vocab_path+f'sp_tokeniser_{args.vocab_size}.model', trucation=True, max_length=256)
        transformer_tokeniser_sp = AutoTokenizer.from_pretrained(vocab_path+'sp_transformer_compatible_tok', trucation=True, max_length=args.max_length, use_fast=False, pad_token = '[BOS]', bos_token='[BOS]', eos_token='[BOS]')
        
        return transformer_tokeniser_sp
    else:
        raise ValueError('Invalid tokeniser. Please choose from gpt2, pubmedbert, PTCTWordPiece or PTCTBPE')
