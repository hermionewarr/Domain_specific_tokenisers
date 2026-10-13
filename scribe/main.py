#Script to train scribe the language model.
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from train_scribe import train_model, pretrain_pubmed, wandb_setup
from test_scribe import test, evaluate_model, clinical_evalution, clinical_metrics_ct_rate, clinical_metrics_ouh_embeddings
from tokenisers import ScribeTokeniser
from Scribe import Scribe, ScribeConfig
from ScribeDataset import ScribeDataset
from ScribeCustomCollator import CustomCollator
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RUNS_DIR, generated_reports_csv, report_dir

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='1', help='the run name.')
    #parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--train_data', type=str, default='reports_sections_train.csv', help='the name of the training data file')
    parser.add_argument('--val_data', type=str, default='reports_sections_val.csv', help='the name of the validation data file')
    parser.add_argument('--test_data', type=str, default='reports_sections_test.csv', help='the name of the test data file')
    parser.add_argument('--data_source', type=str, default='ouh', help='the source of the data. Please choose from ouh or MIMIC')

    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='the number of samples for a batch')
    parser.add_argument('--vocab_size', type=int, default=3000, help='the size of the vocabulary')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate for the model')
    parser.add_argument('--block_size', type=int, default=1024, help='the block size for the model') #switched from 512 on the 6th dec.

    parser.add_argument('--tokeniser', type=str, default='PTCTBPE', help='the tokeniser to use. Please choose from gpt2, pubmedbert, PTCTWP or PTCTBPE')
    parser.add_argument('--max_length', type=int, default=1024, help='the maximum length of the input sequence') #switched from 256 on the 28th nov

    parser.add_argument('--pretrained', type=str, default=None, help='the path to the pretrained model')
    parser.add_argument('--train_mode', type=str, default="True", help='whether to train the model')
    parser.add_argument('--test_mode', type=str, default="True", help='whether to test the model')
    parser.add_argument('--eval_mode', type=str, default="True", help='whether to evaluate the model')
    parser.add_argument('--wandb_log', type=str, default="True", help='whether to log to wandb')

    parser.add_argument('--extra_tok_info', type=str, default="fc_train_no_digit", help='extra info to add to the tokeniser name')
    parser.add_argument('--pubmed_pretrain', type=str, default="False", help='whether to pretrain on PubMed data')
    parser.add_argument('--pubmed_epochs', type=int, default=1, help='number of epochs to pretrain on PubMed data')
    parser.add_argument('--pubmed_pretrained_model_path', type=str, default=None, help='the path to the pretrained model')

    args = parser.parse_args()
    return args

def make_run_dir(run_name):
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        overwrite = input(f'run {run_name} already exists. Do you want to overwrite? (y/n)')
        if overwrite == 'n':
            run_name = input('Please enter a new run name: ')
            return make_run_dir(run_name)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "generated_reports").mkdir(parents=True, exist_ok=True)
    return run_name

def find_90th_percentile_and_set_block_size(train_dataset, args):
    """
    Find the token number where 90% of tokenized reports are within that range
    and set args.block_size accordingly.
    """
    print("Calculating token lengths for all reports in training dataset...")
    
    # Calculate token lengths for all reports
    token_lengths = []
    for report in train_dataset.reports:
        token_length = len(train_dataset.tokenizer.encode(report))
        token_lengths.append(token_length)
    
    # Calculate 90th percentile
    percentile_95 = np.percentile(token_lengths, 95)
    percentile_90 = np.percentile(token_lengths, 90)
    percentile_99 = np.percentile(token_lengths, 99)
    print(f"90th percentile token length: {percentile_90:.2f}")
    print(f"99th percentile token length: {percentile_99:.2f}")
    print(f"Min token length: {min(token_lengths)}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Mean token length: {np.mean(token_lengths):.2f}")
    print(f"Median token length: {np.median(token_lengths):.2f}")
    
    # Calculate how many reports are over the 90th percentile
    reports_over_90th = sum(1 for length in token_lengths if length > percentile_90)
    total_reports = len(token_lengths)
    print(f"Reports over 90th percentile ({percentile_90:.2f}): {reports_over_90th} out of {total_reports} ({reports_over_90th/total_reports*100:.1f}%)")
    
    reports_over_95th = sum(1 for length in token_lengths if length > percentile_95)
    print(f"Reports over 95th percentile ({percentile_95:.2f}): {reports_over_95th} out of {total_reports} ({reports_over_95th/total_reports*100:.1f}%)")
    
    reports_over_99th = sum(1 for length in token_lengths if length > percentile_99)
    print(f"Reports over 99th percentile ({percentile_99:.2f}): {reports_over_99th} out of {total_reports} ({reports_over_99th/total_reports*100:.1f}%)")
    
    over_128 = sum(1 for length in token_lengths if length > 128)
    over_256 = sum(1 for length in token_lengths if length > 256)
    over_512 = sum(1 for length in token_lengths if length > 512)
    over_1024 = sum(1 for length in token_lengths if length > 1024)
    print(over_128, over_256, over_512, over_1024)
    # Set block_size based on 90th percentile
    if percentile_90 <= 128:
        new_block_size = 128
        new_batch_size = 512
        print(f"90th percentile ({percentile_90:.2f}) <= 256, setting block_size to {new_block_size}]")
        print(f"number of reports over {new_block_size} = {over_256}")
    elif percentile_90 <= 256:
        new_block_size = 256
        new_batch_size = 256 #128
        print(f"90th percentile ({percentile_90:.2f}) <= 256, setting block_size to 256 wit batch size = {new_batch_size}")
        print(f"number of reports over {new_block_size} = {over_256}")
    elif percentile_90 <= 512:
        new_block_size = 512
        new_batch_size = 128
        print(f"90th percentile ({percentile_90:.2f}) between 256 and 512, setting block_size to 512")
        print(f"number of reports over {new_block_size} = {over_512}")
    elif percentile_90 <= 1024:
        new_block_size = 1024
        new_batch_size = 32
        print(f"90th percentile ({percentile_90:.2f}) between 512 and 1024, setting block_size to 1024")
        print(f"number of reports over {new_block_size} = {over_1024}")
    else:
        print(f"WARNING: 90th percentile ({percentile_90:.2f}) is larger than 1024!")
        new_block_size = 1024  # Keep default but warn user
    
    # Update args
    args.block_size = new_block_size
    args.max_length = new_block_size
    args.batch_size = new_batch_size
    
    return args

def main():
    #just to help keep track of runs
    current_date = datetime.now().strftime("%d-%m-%y")
    print("current date: ", current_date)

    # parse arguments
    args = parse_agrs()
    print("data source: ", args.data_source)
    #configs = vars(args)
    #wandb_setup(configs)
    wandb_log = True if args.wandb_log == "True" else False
    
    # create tokenizer
    tokenizer = ScribeTokeniser(args)
    print("tokenizer details: ", tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token)
    #load datasets
    if args.data_source == 'ouh':
        data_dir = str(report_dir('ouh')) + '/'
    elif args.data_source == 'MIMIC':
        data_dir = str(report_dir('MIMIC')) + '/'
    elif args.data_source == 'ct-rate':
        data_dir = str(report_dir('ct-rate')) + '/'
    else:
        raise ValueError(f"Unknown data_source {args.data_source!r}")
    #train_dataset = ScribeDataset(tiny_data, tokenizer)
    train_dataset = ScribeDataset(data_dir+args.train_data, tokenizer, data_source=args.data_source)
    val_dataset = ScribeDataset(data_dir+args.val_data, tokenizer, data_source=args.data_source)
    test_dataset = ScribeDataset(data_dir+args.test_data, tokenizer, data_source=args.data_source)
    #test_dataset = ScribeDataset(tiny_data, tokenizer)

    # Find 90th percentile and set block_size accordingly
    if args.train_mode == "True":
        print(len(train_dataset))
        #over_256 = train_dataset.no_of_reports_overx_tokens(256)
        #over_512 = train_dataset.no_of_reports_overx_tokens(512)
        #over_1024 = train_dataset.no_of_reports_overx_tokens(1024)
        #print("train reports over 256 tokens: ", over_256, "out of ", len(train_dataset))
        #print("train reports over 512 tokens: ", over_512, "out of ", len(train_dataset))
        #print("train reports over 1024 tokens: ", over_1024, "out of ", len(train_dataset))
        
        args = find_90th_percentile_and_set_block_size(train_dataset, args)
        print(f"Final block_size: {args.block_size}")
        print(f"Final max_length: {args.max_length}")

    print("size of train dataset: ",len(train_dataset))
    print("size of val dataset: ",len(val_dataset))
    print("size of test dataset: ",len(test_dataset))

    #make run directory
    if args.train_mode == "True" and args.pretrained is None:
        args.run = make_run_dir(args.run)
    #print(train_dataset.__getitem__(1))
    #print(train_dataset.__getitem__(23251))

    #load dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=CustomCollator(tokenizer, args),
        shuffle=True,
        )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1, #args.batch_size,
        collate_fn=CustomCollator(tokenizer, args),
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size= 1, #4, #1, #args.batch_size, 
        collate_fn=CustomCollator(tokenizer, args),
        shuffle=False,
        )

    #test_dataloader = test_dataloader[:4]
    
    #load the model
    print("vocab size input: ", args.vocab_size)
    model_config = ScribeConfig(vocab_size = tokenizer.vocab_size + (64 - tokenizer.vocab_size % 64), block_size = args.block_size) #vocab_size=tokenizer.vocab_size 
    #the padding on the vocab size stops me getting an error for gpt (ah i added an extra padding token thats why) but anyway gives a buffer to add extra tokens
    print("vocab size w padding: ", model_config.vocab_size)
    #make model vocab size the smae as the tokeniser padded up to the nearest multiple of 64
    #vocab_size = tokenizer.vocab_size
    #model_config.vocab_size = vocab_size + (64 - vocab_size % 64)
    #model_config = ScribeConfig(vocab_size=args.vocab_size)
    model = Scribe(model_config)
    optimizer = None
    scaler = None
    epoch = 0
    #IF PRETRAINED MODEL
    if args.pretrained:
        print("loading pretrained model")
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print(f"loaded model from epoch {epoch}")
        if args.train_mode == "True":
            optimizer = checkpoint['optimizer']
            scaler = checkpoint['scaler']
    else:
        print("initialising model from scratch")

    for param in model.parameters():
        print(param.dtype)
        break


    #TRAIN MODEL
    if args.train_mode == "True":
        print("training model")
        wandb_setup(vars(args), wandb_log)
        # PubMed pretraining if enabled
        if args.pubmed_pretrain == "True":
            print("Starting PubMed pretraining...")
            model, optimizer, scaler = pretrain_pubmed(args, model, tokenizer, model_config)
            print("PubMed pretraining completed")

        # if pubmed pretrained model path is provided, load it
        if args.pubmed_pretrained_model_path:
            print("Loading PubMed pretrained model...")
            checkpoint = torch.load(args.pubmed_pretrained_model_path)
            model.load_state_dict(checkpoint['model'])
            optimizer = checkpoint['optimizer']
            scaler = checkpoint['scaler']
            print("PubMed pretrained model loaded")
        
        train_model(args, model, train_dataloader, val_dataloader, tokenizer, model_config, optimizer, scaler, epoch)

    #TEST MODEL
    if args.test_mode=="True": #and (args.pretrained or (args.train_mode=="True")):
        print("testing model")
        if args.train_mode=="True":
            model.load_state_dict(torch.load(RUNS_DIR / args.run / "model" / f"final_{args.run}_model.pth")['model'])
        model.eval()
        print("max length of reports: ", args.max_length)
        gen_reports_df = test(model, test_dataloader, tokenizer, args)

    #EVALUATE MODEL
    if args.eval_mode=="True":
        #load in generate reports and evaluate
        print("evaluating model")
        if args.test_mode=="False":
            gen_reports_df = pd.read_csv(generated_reports_csv(args.run))
    
        evaluate_model(gen_reports_df, args)
        evaluate_model(gen_reports_df, args, 'conclusion', 'gen_conclusion', info='con')

        if args.data_source == "MIMIC":
            #use green score of chexbert labeler to evaluate the generated text
            print("evaluating generated reports clinically")
            clinical_evalution(gen_reports_df, args)
        
        elif args.data_source == "ct-rate":
            print("evaluating ct-rate data clinically")
            clinical_metrics_ct_rate(gen_reports_df, args)
            
        elif args.data_source == "ouh":
            print("evaluating ouh data clinically")
            clinical_metrics_ouh_embeddings(gen_reports_df, args)
    


    return

if __name__ == "__main__":
    main()