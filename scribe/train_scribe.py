# training loops for scribe
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from tokenisers import ScribeTokeniser
from utils import model_params_setup
from tqdm import tqdm
import wandb
from time import time
import numpy as np 
from torch.amp import GradScaler, autocast
import math
import os
from test_scribe import test, evaluate_model
import time
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PUBMED_ABSTRACTS, RUNS_DIR

CUDA_LAUNCH_BLOCKING=1
#set the device as torch cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretrain_pubmed(args, model, tokenizer, model_config):
    """
    Pretrain the model on PubMed abstracts
    """
    model.train()
    model.to(device)
    
    # Load PubMed data
    pubmed_data_path = str(PUBMED_ABSTRACTS)
    pubmed_data = load_dataset("json", data_files=pubmed_data_path, split="train")
    pubmed_dataset = pubmed_data.rename_column("abstract", "text")
    
    # Create dataloader for PubMed data
    pubmed_dataloader = DataLoader(
        pubmed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'input_ids': tokenizer([item['text'] for item in x], padding="longest", return_tensors="pt", truncation=True, max_length=args.max_length)['input_ids'],
            'attention_mask': tokenizer([item['text'] for item in x], padding="longest", return_tensors="pt", truncation=True, max_length=args.max_length)['attention_mask']
        }
    )
    
    # Setup optimizer and scaler
    optimizer, _ = model_params_setup(model, args, model_config)
    scaler = GradScaler()
    
    # Calculate log interval (log 200 times per epoch)
    total_batches = len(pubmed_dataloader)
    log_interval = max(1, math.floor(total_batches / 200))
    print(f"Logging PubMed pretraining loss every {log_interval} batches")
    
    # Training loop
    for epoch in range(args.pubmed_epochs):
        total_loss = 0
        batch_loss = 0
        epoch_start_time = time.time()  # Start timing the epoch
        
        for i, batch in enumerate(tqdm(pubmed_dataloader, desc=f"PubMed Pretraining Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(bool).to(device)
            
            with autocast('cuda'):
                _, outputs, loss = model(input_ids, attention_mask, targets=input_ids)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            batch_loss += loss.item()
            
            # Log more frequently
            if (i + 1) % log_interval == 0:
                avg_batch_loss = batch_loss / log_interval
                print(f"PubMed Pretraining Epoch {epoch+1}, Batch {i+1}/{total_batches}, Average Loss: {avg_batch_loss:.4f}")
                if args.wandb_log == "True":
                    wandb.log({
                        'pubmed_pretrain_loss': avg_batch_loss,
                    })
                batch_loss = 0  # Reset batch loss accumulator
                #save model
                save_model(model, args, optimizer, epoch, avg_batch_loss, scaler, pubmed_pretrain=True)
            
            #empty cache
            torch.cuda.empty_cache()
            
        epoch_time = time.time() - epoch_start_time  # Calculate epoch time
        avg_loss = total_loss / len(pubmed_dataloader)
        print(f"PubMed Pretraining Epoch {epoch+1} Complete, Average Loss: {avg_loss:.4f}")
        hours = epoch_time // 3600
        minutes = (epoch_time % 3600) // 60
        print(f"Time taken for epoch {epoch+1}: {hours:.0f} hours and {minutes:.0f} minutes")
        print("saving pubmed pretrain model")
        save_model(model, args, optimizer, epoch, avg_batch_loss, scaler, pubmed_pretrain=True)
    
    return model, optimizer.state_dict(), scaler.state_dict()

def wandb_setup(configs, wandb_log):  
    if wandb_log:
        wandb.init(
            project='Scribe', 
            name = configs['run'],
            group="vocab_size_exp",
            )
        wandb.config.update(configs)

def train_model(args, model, train_dataloader, val_dataloader, tokenizer, model_config, optimizer_cp=None, scaler_cp = None, prev_epochs = None):
    model.train()
    model.to(device)
    print(f"training on device: {device}")
    #wandb_setup(vars(args), wandb_log)
    best_val_loss = 100
    best_bleu1 = 0
    val_loss = best_val_loss #just for initial model saves
    optimizer, scheduler = model_params_setup(model, args, model_config)
    
    scaler = GradScaler()
    if optimizer_cp is not None:
        print("loading optimizer")
        optimizer.load_state_dict(optimizer_cp)
    
    if scaler_cp is not None:
        print("loading scaler")
        scaler.load_state_dict(scaler_cp)
   
    total_batches = len(train_dataloader)  # total number of batches in an epoch
    log_interval = max(1, math.floor(total_batches / 10)) 
    print("log interval: ", log_interval)

    start_epoch = 0
    if prev_epochs is not None:
        print("loading previous epochs")
        start_epoch = prev_epochs
    print("starting from epoch: ", start_epoch)
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Epochs", position=0):
        for i, batch in tqdm(enumerate(train_dataloader), desc="Batches", position=1): #later use gradient accumulation to simulate larger batch size
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(bool).to(device)
            #if epoch == 0 and i == 0:
                #flops = FlopCountAnalysis(model, inputs=(input_ids, attention_mask, input_ids))
                #print(f"FLOPs: {flops.total()/10**9:.2f} G") # causes an error and is also really high!
            #optimizer.zero_grad()
            with autocast('cuda'):
                _,outputs, loss = model(input_ids, attention_mask, targets=input_ids)
            scaler.scale(loss).backward()
            #if grad clip
            #scaler.unscale_(optimizer)
            #clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) #does putting this here instead of before save mem?

            if epoch == 0 and i == 0:
                max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
                print(f"Peak GPU memory usage during the first iteration: {max_memory:.2f} GB")

            if args.wandb_log and (i % log_interval == 0): #so log ten times per epoch
                wandb.log({'train loss': loss.item()})
            #empty cache
            torch.cuda.empty_cache()
        if epoch == 0:
            print("time for one epoch is: ", time.time()-start_time)

        #validate model
        if (epoch % (args.epochs/10) == 0) and (epoch != 0):
            val_loss = val(model, args, val_dataloader)
            print(f'epoch: {epoch}, loss: {loss.item()}')
            print(f'val loss: {val_loss}')
            if args.wandb_log:
                wandb.log({'val loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, args, optimizer, epoch, val_loss, scaler)
            
            #dont validate if pubmedbert tokeniser while we fix
            if args.tokeniser != 'pubmedbert':
                val_metrics = val_generate(model, args, val_dataloader, tokenizer)
                if args.wandb_log:
                    # log the nlp metrics from dataframe to wandb
                    wandb.log(val_metrics.T.to_dict())
                new_bleu1 = val_metrics['bleu1']
                if new_bleu1 > best_bleu1:
                    best_bleu1 = new_bleu1
                    print(f"new best bleu1: {best_bleu1}")
                    save_model(model, args, optimizer, epoch, val_loss, scaler, best_bleu = best_bleu1)

        #save final model as well as best (do it at the end of each poch in case i get bored and cancel it early)
        save_model(model, args, optimizer, epoch, val_loss, scaler, final=True)

    return model

@torch.no_grad() 
def val(model, args, val_dataloader):
    model.eval()
    val_losses = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(bool).to(device)
        with autocast('cuda'):
            _,outputs, loss = model(input_ids, attn_mask, targets=input_ids)
        val_losses.append(loss.item())
        
    return np.mean(val_losses)

def val_generate(model, args, val_dataloader, tokenizer):
    """
    Function to randomly sample a subset of the validation reports and then run inference on them.
    1. get subset of reports (maybe just do x shuffled batches)
    2. generate conclusions for subset
    3. evaluate the conclusions
    4. log the results to wandb
    """
    val_count = 0
    model.eval()
    start_time = time.time()
    gen_reports = test(model, val_dataloader, tokenizer, args, val=True)
    metrics = evaluate_model(gen_reports, args, 'conclusion', 'gen_conclusion', info='con', val=True)
    print(f"Time taken to val reports: {time.time() - start_time}")
    metrics = metrics[["bleu1", "bleu4","meteor", "rougeL"]]
    #calculate the  cross entropy loss between the output logits and input ids. (need to adjust gen and test returns)
    return metrics

def save_model(model, args, optimiser, epoch, val_loss, scaler, best_bleu = None, final=False, pubmed_pretrain=False):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'scaler': scaler.state_dict()
        }
        #make parent dir
        model_dir = RUNS_DIR / args.run / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if final:
            torch.save(checkpoint, model_dir / f"final_{args.run}_model.pth")
        elif best_bleu is not None:
            torch.save(checkpoint, model_dir / f"best_bleu1_{args.run}_model.pth")
        elif pubmed_pretrain:
            torch.save(checkpoint, model_dir / "pubmed_pretrain_model.pth")
        else:
            torch.save(checkpoint, model_dir / f"{args.run}_model.pth")
        
        print("model saved")
        return
    
