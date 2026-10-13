#from pycocoevalcap.bleu.bleu import Bleu
#from pycocoevalcap.meteor.meteor import Meteor
#from pycocoevalcap.rouge.rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from tqdm import tqdm
import torch
from torch.amp import autocast
import pandas as pd
import numpy as np
import time 
import tempfile
import subprocess
#import sys
#from green_score import GREEN

#these imports are for the clinical metrics
import tempfile
import csv
import sys
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (
    CHEXBERT_WEIGHTS,
    CHEXPERT_CONDA,
    CHEXPERT_LABELER,
    CONDA_BIN,
    CTCLIP_TEXT_CLASSIFIER,
    EVAL_DIR,
    GREEN_DIR,
    RADBERT_CHECKPOINT,
    RUNS_DIR,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    nltk.download('punkt_tab', quiet=True)
except Exception as exc:
    print(f"nltk punkt_tab download skipped: {exc}")


def _generated_dir(run):
    path = RUNS_DIR / run / "generated_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _import_chexbert():
    if EVAL_DIR:
        sys.path.append(os.path.abspath(EVAL_DIR))
    from CheXbert.src.label import label, save_preds
    from CheXbert.src.models.bert_labeler import bert_labeler
    return label, save_preds, bert_labeler


def _import_green():
    if GREEN_DIR:
        sys.path.append(os.path.abspath(GREEN_DIR))
    from green_score.green import GREEN
    return GREEN


def _import_ctclip():
    if not CTCLIP_TEXT_CLASSIFIER:
        raise FileNotFoundError(
            "Set SCRIBE_CTCLIP_TEXT_CLASSIFIER to the CT-CLIP text_classifier directory."
        )
    sys.path.append(os.path.abspath(CTCLIP_TEXT_CLASSIFIER))
    from eval import evaluate_predictions, save_confusion_matrices
    from infer import infer_from_dataframe
    return infer_from_dataframe, evaluate_predictions, save_confusion_matrices


def test(model, dataloader, tokenizer, args, val=False):
    model.eval()
    model.to(device)
    print("model testing on device: ", device)
    gen_reports = []
    pat_ids = []
    ref_reports = []
    #gen_con = []
    conclusions = []
    findings = []
    run_name = args.run
    i= 0
    start_time = time.time()
    for i, batch in enumerate(tqdm(dataloader)):
        input_ids = batch['input_ids']
        #findings = batch['findings']
        #take the first token of each input id # for now will replace with bos token
        #print("debugging testing")
        #print(input_ids.size())
        #input_ids = input_ids[:, :10]#.unsqueeze(1) 
        input_ids = batch['findings_ids']#[:, :20] *use this
        attention_masks = batch["findings_msk"]#[:, :20]
        #ref_report = batch["reference_report"]
        findings.append(batch["findings"])
        #print(ref_report)
        #print(len(input_ids))
        #print(attention_masks)
        #print("First 10 tokens of the reports:")
        #print(input_ids)
        #print(tokenizer.batch_decode(input_ids))
        if (args.tokeniser == "pubmedbert") | (args.tokeniser == "cxr-bert"):
            #all but the last token as i think bert tokenisers insert a [cls] token
            #print(input_ids.shape)
            #return to this
            input_ids = input_ids[:,:-1] #this might not work might have to just replace cls with padding
            attention_masks = attention_masks[:,:-1]
            #attention_masks = batch["attention_mask"][:,:-20]
        pat_ids.append(batch['pat_ids'])
        ref_reports.append(batch["reference_report"])
        conclusions.append(batch["conclusion"])
        with autocast('cuda'):
            #start_time = time.time()
            outputs = model.generate(input_ids.to(device), attn_mask = attention_masks.to(device), max_new_tokens=args.max_length, tokenizer=tokenizer) #attention_masks.to(device)
            #print(f"Time taken to generate reports: {time.time() - start_time}")
            outputs = outputs.detach().cpu().numpy()
            #gen = gen.detach().cpu().numpy()
        #decode the outputs
        gen_reports.append(tokenizer.batch_decode(outputs))
        #gen_con.append(tokenizer.batch_decode(gen))
        #empty cache
        torch.cuda.empty_cache()
        #maybe write generated report to file here so that if it crashes i can still get some of the results/ restart from where i left off.
        i=i+1
        #print(i)
        if val:
            if i == 500:
                break

        """else:
            if i == 5:
                print("only testing on first: ", i)
                print(gen_reports[-1])
                print(ref_reports[-1])
                print(tokenizer.decode(input_ids[-1], show_special_tokens=True))
                break"""
        
        
    if val == False: print(f"Time taken to generate reports: {time.time() - start_time}")  
    ref_reports = [item for sublist in ref_reports for item in sublist] 
    gen_reports = [item for sublist in gen_reports for item in sublist]
    pat_ids = [item for sublist in pat_ids for item in sublist] 
    ##gen_con = [item for sublist in gen_con for item in sublist]
    conclusions = [item for sublist in conclusions for item in sublist]

    #extract the conculsions from the generated reports
    #print("debugging")
    #print(gen_reports[0])
    #print(len(gen_reports))
    #gen_con = [report.split("conclusion: ")[1] for report in gen_reports]
    gen_con = []
    tok_conclusion = tokenizer.decode(tokenizer.encode("conclusion: "))
    if args.tokeniser == "pubmedbert":
        tok_conclusion = tokenizer.decode(tokenizer.encode("conclusion: ")[1:-1])
        print("pubmed: ", tok_conclusion)
    for report in gen_reports:
        #print(tokenizer.decode(tokenizer.encode("conclusion: ")))
        #if "conclusion: " in report:
        if tok_conclusion in report:
            gen_con.append(report.split(tok_conclusion)[1])
        else:
            #print(f"Failed to extract conclusion from report: ~")#{report}")
            gen_con.append(np.nan)
    #remove the BOS token from the generated reports and conclusions
    print(tokenizer.special_tokens_map)
    gen_reports = [report.replace("[BOS]", "") for report in gen_reports] #[(report.replace(tokenizer.bos_token, "") and report.replace(tokenizer.pad_token, "")) for report in gen_reports] #gen_reports = [report.replace("[BOS]", "") for report in gen_reports]
    ref_reports = [report.replace("[BOS]", "") for report in ref_reports]
    #gen_con = [(report.replace("[BOS]", "") and report.replace(tokenizer.bos_token, "") and report.replace(tokenizer.pad_token, "")) if isinstance(report, str) else report for report in gen_con] #gen_con = [report.replace("[BOS]", "") if isinstance(report, str) else report for report in gen_con]
    gen_con = [report.replace("[BOS]", "") if isinstance(report, str) else report for report in gen_con]
    conclusions = [report.replace("[BOS]", "") for report in conclusions]
    #try and remove the findings from the generated reports if not then just insert nan
    
    #print(gen_con[0])
    #print(gen_reports[0])
    gen_report_df = pd.DataFrame({'patient_id': pat_ids, 'reference reports': ref_reports,'generated_report': gen_reports, 'conclusion': conclusions, 'gen_conclusion': gen_con})
    print("Number of reports generated: ", len(gen_report_df))
    
    gen_report_df = gen_report_df.dropna(subset=['reference reports', 'generated_report', 'conclusion', 'gen_conclusion']) #basically should remove the ones where the findings were too long so the conclusion prompt got chopped off
    #print(len(gen_report_df))
    if val==False:
        out_dir = _generated_dir(args.run)
        out_csv = out_dir / f"generated_reports_{run_name}.csv"
        gen_report_df.to_csv(out_csv, index=False)
        print("test reports saved to: ", out_csv)
    
    return gen_report_df

def evaluate_model(test_report_df, args, ref='reference reports', gen='generated_report', info='rep', val=False):
    #calulate nlp metrics for the test reports
    #test_report_df = test(model, test_dataloader, tokenizer, args)
    #remove nans
    og_len = len(test_report_df)
    test_report_df = test_report_df.dropna(subset=[ref, gen])
    #if val==False:
    print(f"Removed {og_len - len(test_report_df)} rows with NaNs")
    #remove any unwanted special tokens from generated reports
    #test_report_df[gen] = test_report_df[gen].str.replace("[BOS]", "")
    #test_report_df.to_csv(f'runs/{args.run}/generated_reports/generated_reports_eval_{args.run}.csv', index=False)
    #calculate the metrics
    print("calculating NLP metrics")
    # Calculate scores for each row
    test_report_df[['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rouge2', 'rougeL']] = test_report_df.apply(
        lambda row: calculate_metrics(row[ref], row[gen]),
        axis=1, result_type="expand"
    )
    # Compute average scores
    average_scores = test_report_df[['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rouge2', 'rougeL']].mean()
    if val==False:
        print("\nAverage scores:")
        print(average_scores)
        #save the results
        out_dir = _generated_dir(args.run)
        test_report_df.to_csv(out_dir / f"{args.run}_test_metrics_{info}.csv", index=False)
        average_scores.to_csv(out_dir / f"{args.run}_average_test_metrics_{info}.csv")
    return average_scores

# Helper function to calculate metrics
def calculate_metrics(reference, generated):
    # Simulating Bleu tokenisation
    reference = reference.replace("[BOS]", "")
    generated = generated.replace("[BOS]", "")
    reference = reference.lower()
    generated = generated.lower()
    # BLEU
    #bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=SmoothingFunction().method1) #cummalitive bleu4
    bleu1 = calculate_bleu(reference, generated, 1)
    bleu2 = calculate_bleu(reference, generated, 2)
    bleu3 = calculate_bleu(reference, generated, 3)
    bleu4 = calculate_bleu(reference, generated, 4)
    # METEOR
    meteor = meteor_score([word_tokenize(reference)], word_tokenize(generated)) # idk if its right to tokenise like this but thats what nitk said to do.
    # ROUGE
    # Initialize Rouge scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference, generated)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rougeL

# Helper function to calculate BLEU scores for n-grams
def calculate_bleu(reference, generated, n):
    weights = [1 / n if i < n else 0 for i in range(4)]
    return sentence_bleu(
        [reference.split()],
        generated.split(),
        weights=weights,
        smoothing_function=SmoothingFunction().method1
    )

### CLINICAL EVALUATION MIMIC ONLY ###
def clinical_evalution(gen_reports_df, args):
    """
    Function to evaluate the clinical correctnous of the reports.
    need to extract labels from both impressions and compare them.
    requires chexbert labeller. or green metric 
    """
    reference_text = gen_reports_df['conclusion']
    generated_text = gen_reports_df['gen_conclusion']
    #print(reference_text[0])
    #print(generated_text[0])

    #chexbert scores
    #chexbert = get_chexbert()
    #gen_df, ref_df = get_chexbert_labels_for_gen_and_ref_reports(chexbert, reference_text, generated_text, args)
    
    
    chexpert_metrics(reference_text, generated_text, CHEXPERT_LABELER, args)
    #green scores
    #green_score_list, summary, result_df =green_CE(reference_text, generated_text) #really slow like over an hour for 500 reps
    #results_df.to_csv(f"runs/{args.run}/generated_reports/{args.run}_green_scores.csv", index=False)

    return

### GREEN METRIC ###
def green_CE(reference_text, generated_text):
    "Very slow but interesting metric"
    GREEN = _import_green()
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".")
    mean, std, green_score_list, summary, result_df = green_scorer(reference_text, generated_text)
    print(green_score_list)
    print(summary)
    for index, row in result_df.iterrows():
        print(f"Row {index}:\n")
        for col_name in result_df.columns:
            print(f"{col_name}: {row[col_name]}\n")
        print('-' * 80)
    return green_score_list, summary, result_df

### CHEXBERT METRICS ###
def get_chexbert():
    if not CHEXBERT_WEIGHTS:
        raise FileNotFoundError("Set SCRIBE_CHEXBERT_WEIGHTS to the CheXbert checkpoint.")
    _, _, bert_labeler = _import_chexbert()
    model = bert_labeler()
    checkpoint = torch.load(CHEXBERT_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    return model

def get_chexbert_labels_for_gen_and_ref_reports(chexbert, reference_text, generated_text, args):

    print("size check: ", len(reference_text))
    label, save_preds, _ = _import_chexbert()
    with tempfile.TemporaryDirectory() as temp_dir:
    #if i==1:
        csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
        csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

        header = ["Report Impression"]

        with open(csv_gen_reports_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            #csv_writer.writerows([[gen_report] for gen_report in generated_text])
            #write the generated report in quotes
            csv_writer.writerows([['"' + gen_report + '"'] for gen_report in generated_text])

        with open(csv_ref_reports_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows([[ref_report] for ref_report in reference_text])

        # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
        preds_gen_reports = label(chexbert, csv_gen_reports_file_path, device = device)
        preds_ref_reports = label(chexbert, csv_ref_reports_file_path, device = device)
        #disease order: Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices,No Finding
        #redefines the labels to the standard 1,0,-1,nan format
        save_path = str(_generated_dir(args.run) / "ce_labels") + "/"
        df = save_preds(preds_gen_reports, csv_gen_reports_file_path, save_path+"gen/") #path not used here as currently not saving
        df2 = save_preds(preds_ref_reports, csv_ref_reports_file_path, save_path+"ref/")
    
    #calculate the clinical metrics
    ce_metrics(df2, df)
    
    return df, df2

def ce_metrics(df_targets, df_outputs):
    print(df_outputs.head())
    print(df_targets.head())
    print(df_outputs.shape)
    print(df_targets.shape)
    print(df_outputs.columns)
    #convvert -1 to 1 and nans to 0
    df_outputs = df_outputs.replace(-1, 1)
    df_outputs = df_outputs.replace(np.nan, 0)
    df_targets = df_targets.replace(-1, 1)
    df_targets = df_targets.replace(np.nan, 0)
    #drop report impressions column
    df_outputs = df_outputs.drop(columns=['Report Impression'])
    df_targets = df_targets.drop(columns=['Report Impression'])
    #add more here if want the metrics on a per disease basis
    p_sup,r_sup,f1_sup,_ = precision_recall_fscore_support(df_targets, df_outputs, average='macro')
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(df_targets, df_outputs, average='micro')
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(df_targets, df_outputs, average='weighted')
   
    print(f"Macro Precision: {p_sup}, Recall: {r_sup}, F1: {f1_sup}")
    print(f"Micro Precision: {p_micro}, Recall: {r_micro}, F1: {f1_micro}")
    print(f"Weighted Precision: {p_weighted}, Recall: {r_weighted}, F1: {f1_weighted}")

    #get the clinical metrics for each disease
    print("Disease\tPrecision\tRecall\tF1")
    for i in range(14):
        p,r,f1,_ = precision_recall_fscore_support(df_targets.iloc[:,i], df_outputs.iloc[:,i], average='binary')
        print(f"{df_targets.columns[i]}\t{np.round(p,3)}\t{np.round(r,3)}\t{np.round(f1,3)}")
     
    return

#chexpert
def chexpert_metrics(df_targets, df_outputs, chexpert_path, args):
    if not chexpert_path:
        raise FileNotFoundError(
            "Set SCRIBE_CHEXPERT_LABELER to the CheXpert labeler directory. "
            "Clone from https://github.com/stanfordmlgroup/CheXpert-labeler"
        )
    if not os.path.exists(chexpert_path):
        raise FileNotFoundError(f"CheXpert labeler not found at {chexpert_path}. Clone from https://github.com/stanfordmlgroup/CheXpert-labeler")

    with tempfile.TemporaryDirectory() as temp_dir:
        ref_path = os.path.join(temp_dir, "chexpert_targets.csv")
        gen_path = os.path.join(temp_dir, "chexpert_outputs.csv")
        ref_out = os.path.join(temp_dir, "chexpert_ref_lab_out.csv")
        gen_out = os.path.join(temp_dir, "chexpert_gen_lab_out.csv")

        # Remove nans and clean up data
        print(len(df_targets), len(df_outputs))
        nan_mask = df_targets.isna() | df_outputs.isna()
        df_targets = df_targets[~nan_mask]
        df_outputs = df_outputs[~nan_mask]
        print(len(df_targets), len(df_outputs))
        
        # Strip whitespace
        df_targets = df_targets.apply(lambda x: x.strip() if isinstance(x, str) else x)
        df_outputs = df_outputs.apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Save temporary files
        df_targets.to_csv(ref_path, index=False, header=False)
        df_outputs.to_csv(gen_path, index=False, header=False)

        # Run CheXpert labeler
        env_name = CHEXPERT_CONDA
        def command(env_name, chexpert_path, input_csv, output_csv):
            return f'{CONDA_BIN} run -n {env_name} python {chexpert_path}/label.py --reports_path {input_csv} --output_path {output_csv}'
        
        save_dir = _generated_dir(args.run) / "ce_labels"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / "ce_output.out"
        
        # Run labeler and save output
        with open(output_file, "w") as f:
            subprocess.run(command(env_name, chexpert_path, ref_path, ref_out), shell=True, check=True, cwd=chexpert_path, stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(command(env_name, chexpert_path, gen_path, gen_out), shell=True, check=True, cwd=chexpert_path, stdout=f, stderr=subprocess.STDOUT)
        
        # Load and save CheXpert labels
        ref_labels = pd.read_csv(ref_out).rename(columns={'Reports':'Report Impression'})
        gen_labels = pd.read_csv(gen_out).rename(columns={'Reports':'Report Impression'})
        
        ref_labels.to_csv(save_dir / "ref_chexpert.csv", index=False)
        gen_labels.to_csv(save_dir / "gen_chexpert.csv", index=False)
        
    # Calculate metrics
    ce_metrics(ref_labels, gen_labels)
    return ref_labels, gen_labels

### CLINICAL EVALUATION CT-RATE ONLY ###
def clinical_metrics_ct_rate(gen_reports_df, args):
    reference_text = gen_reports_df[['patient_id','conclusion']]
    generated_text = gen_reports_df[['patient_id','gen_conclusion']]
    reference_text = reference_text.rename(columns={'conclusion': 'report_text'})
    generated_text = generated_text.rename(columns={'gen_conclusion': 'report_text'})

    print(len(reference_text), len(generated_text))
    if not RADBERT_CHECKPOINT:
        raise FileNotFoundError("Set SCRIBE_RADBERT_CHECKPOINT to RadBertClassifier.pth.")
    infer_from_dataframe, evaluate_predictions, save_confusion_matrices = _import_ctclip()
    checkpoint = RADBERT_CHECKPOINT
    reference_labels = infer_from_dataframe(reference_text, checkpoint)
    generated_labels = infer_from_dataframe(generated_text, checkpoint)

    save_dir = _generated_dir(args.run)
    
    metrics_df, eval = evaluate_predictions(reference_labels, generated_labels, save_dir=save_dir)
    metrics_df.to_csv(os.path.join(save_dir, f"{args.run}_ce_metrics.csv"), index=False)
    print(metrics_df)
    
    save_path = os.path.join(save_dir, f"{args.run}_cm.png")
    fig = save_confusion_matrices(eval["confusion_matrix"], save_path)
    return


### CLINICAL EVALUATION OUH ONLY ###
def clinical_metrics_ouh_embeddings(gen_reports_df, args):
    """
    Function to evaluate the clinical similarity of OUH reports using embeddings.
    Uses BiomedNLP-PubMedBERT model to generate embeddings and calculate cosine similarity.
    """
    reference_text = gen_reports_df[['patient_id','conclusion']]
    generated_text = gen_reports_df[['patient_id','gen_conclusion']]
    ref_texts = reference_text['conclusion'].values.tolist()
    gen_texts = generated_text['gen_conclusion'].values.tolist()

    try:
        # Initialize the model and tokenizer
        #model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        model_name = "microsoft/BiomedVLP-CXR-BERT-general"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        def get_embeddings(texts):
            embeddings = []
            for text in texts:
                # Tokenize and prepare input
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get model output
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Use CLS token embedding as text representation
                embeddings.append(outputs.last_hidden_state[0, 0, :].cpu().numpy())
            
            return np.array(embeddings)
        
        # Get embeddings for reference and generated texts
        ref_embeddings = get_embeddings(ref_texts)
        gen_embeddings = get_embeddings(gen_texts)
        
        # Calculate cosine similarity between reference and generated embeddings
        similarities = []
        for ref_emb, gen_emb in zip(ref_embeddings, gen_embeddings):
            similarity = np.dot(ref_emb, gen_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb))
            similarities.append(similarity)
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        print(f"\nClinical Evaluation Results:")
        print(f"Mean Cosine Similarity: {mean_similarity:.4f}")
        print(f"Standard Deviation: {std_similarity:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'patient_id': reference_text['patient_id'],
            'similarity_score': similarities
        })
        
        # Use the correct path for saving metrics
        save_dir = os.path.join('runs', args.run, 'generated_reports')
        print(f"save_dir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'CE_OUH.csv')
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
        
        return similarities, mean_similarity, std_similarity
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    
   
