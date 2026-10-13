# Optional extra OUH clinical metrics: embedding similarity, RadGraph, MetaMap, and GREEN.
# External packages are imported inside the functions that use them.
import pandas as pd
import os
import sys
from pathlib import Path
from time import sleep
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GREEN_DIR, METAMAP_DIR, RUNS_DIR

radgraph_metrics = True
cs = True
model_type = 'radgraph-xl'

def clinical_metrics_ouh_embed(gen_reports_df):
    results_df = gen_reports_df[['patient_id', 'conclusion', 'gen_conclusion']]
    results_df = results_df[results_df['conclusion'].notna() & results_df['gen_conclusion'].notna()]
    print(len(results_df))
    #print(results_df.columns)
    
    # Filter out reports that are too short (less than 3 words)
    results_df = results_df[
        (results_df['conclusion'].str.split().str.len() > 2) & 
        (results_df['gen_conclusion'].str.split().str.len() > 2)
    ]
    ref_texts = results_df['conclusion'].values.tolist()
    gen_texts = results_df['gen_conclusion'].values.tolist()

    #quick experiment
    #gen_texts = ref_texts.copy()
    #random.shuffle(gen_texts)
    #test_size = 1000

    print(f"Processing {len(ref_texts)} reports...")
    print("First reference text:", ref_texts[0] if ref_texts else "Empty")
    print("First generated text:", gen_texts[0] if gen_texts else "Empty")

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
        print(f"Mean cosine similarity: {mean_similarity}")
        print(f"std:{np.std(similarities)}")
        return similarities, mean_similarity, ref_embeddings, gen_embeddings
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

#radgraph metric attempt 3 for ouh metrics
#via radgraph
def clinical_metrics_ouh_rad(gen_reports_df, save_path):

    print(f"Starting F1RadGraph computation for {len(gen_reports_df)} samples...")
    from radgraph import F1RadGraph
    #join gen and ref texts together and remove any row where either is empty
    results_df = gen_reports_df[['patient_id', 'conclusion', 'gen_conclusion']]
    results_df = results_df[results_df['conclusion'].notna() & results_df['gen_conclusion'].notna()]
    print(len(results_df))
    print(results_df.columns)
    
    # Filter out reports that are too short (less than 3 words)
    results_df = results_df[
        (results_df['conclusion'].str.split().str.len() > 2) & 
        (results_df['gen_conclusion'].str.split().str.len() > 2)
    ]
    ref_texts = results_df['conclusion'].values.tolist()
    gen_texts = results_df['gen_conclusion'].values.tolist()
    print(len(gen_texts), len(ref_texts))
    #gen_texts = ref_texts.copy()
    #random.shuffle(gen_texts)

    start_time = time.time()
    f1radgraph = F1RadGraph(reward_level="all", model_type=model_type)
    
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=gen_texts, refs=ref_texts)
    
    end_time = time.time()
    duration = end_time - start_time
    #print(f"F1RadGraph computation completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Create DataFrame with individual scores
    results_df = pd.DataFrame({
        'patient_id': results_df['patient_id'].values,
        'reference_text': ref_texts,
        'generated_text': gen_texts,
        'entities_f1': reward_list[0],
        'relations_f1': reward_list[1],
        'overall_f1': reward_list[2]
    })
    
    # Save results DataFrame
    results_save_path = save_path.replace('.csv', '_scores.csv')
    results_df.to_csv(results_save_path, index=False)
    print(f"Saved individual scores to {results_save_path}")
    
    #save annotations
    annotations_ref_df = pd.DataFrame(reference_annotation_lists)
    annotations_gen_df = pd.DataFrame(hypothesis_annotation_lists)

    # Save separately
    ref_save_path = save_path.replace('.csv', '_reference.csv')
    gen_save_path = save_path.replace('.csv', '_generated.csv')

    annotations_ref_df.to_csv(ref_save_path, index=False)
    annotations_gen_df.to_csv(gen_save_path, index=False)

    print(f"Saved reference annotations to {ref_save_path}")
    print(f"Saved generated annotations to {gen_save_path}")

    entities_f1, relations_f1, overall_f1 = mean_reward
    
    print(f"Entities F1: {entities_f1}, Relations F1: {relations_f1}, Overall F1: {overall_f1}")

    return overall_f1, results_df

def _load_generated(run):
    return pd.read_csv(RUNS_DIR / run / "generated_reports" / f"generated_reports_{run}.csv")


def main(runs):
    for run in runs:
        print(run, model_type)
        gen_reports_df = _load_generated(run)
        print(len(gen_reports_df))
        if cs:
            try:
                print("running clinical metrics ouh embed")
                clinical_metrics_ouh_embed(gen_reports_df)
            except Exception as e:
                print(f"Failed to run analysis: {str(e)}")
        if radgraph_metrics:
            try:
                save_path = str(RUNS_DIR / run / "generated_reports" / f"annotations_{run}_test.csv")
                mean_reward, _ = clinical_metrics_ouh_rad(gen_reports_df, save_path)
                print(f"Mean RadGraph F1 score: {mean_reward}")
            except Exception as e:
                print(f"Failed to run radgraph analysis: {str(e)}")
                import traceback
                traceback.print_exc()

# ouh clinical metric attempt number 2
def extract_medical_terms(text, mm):
        try:
            # Extract concepts using MetaMap
            concepts, error = mm.extract_concepts(text)

            if error:
                print(f"MetaMap error: {error}")
                return set()
            
            # Debug: Print all concepts before filtering
            #print("\nAll concepts before filtering:")
            #print(concepts)
            # Extract unique CUIs with their preferred names
            cuis_with_names = set()
            for concept in concepts:
                if concept.cui:  # Only include concepts with CUIs
                    # Get the semantic type of the concept
                    sem_type = concept.semtypes[0] if concept.semtypes else ""
                    
                    # Only include specific semantic types
                    if sem_type in ['dsyn',    # Disease or Syndrome
                                   'fndg',    # Finding
                                   'neop',    # Neoplastic Process
                                   'sosy',    # Sign or Symptom
                                   'anst',    # Anatomical Structure
                                   'bpoc']:   # Body Part, Organ, or Organ Component
                        cuis_with_names.add((concept.cui, concept.preferred_name))
            
            # Debug: Print filtered concepts
            print("\nFiltered concepts:")
            for cui, name in cuis_with_names:
                print(f"CUI: {cui}, Name: {name}")
            
            return cuis_with_names
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return set()
        
def calculate_overlap(set1, set2):
        if not set1 or not set2:
            return 0.0
        
        # Extract just the CUIs for overlap calculation
        set1_cuis = {cui for cui, _ in set1}
        set2_cuis = {cui for cui, _ in set2}
        
        intersection = len(set1_cuis.intersection(set2_cuis))
        union = len(set1_cuis.union(set2_cuis))
           
        return intersection / union if union > 0 else 0.0
        
def clinical_metrics_ouh_metamap(gen_reports_df):
    from pymetamap import MetaMap
    if not METAMAP_DIR:
        raise FileNotFoundError("Set SCRIBE_METAMAP_DIR to the MetaMap public_mm directory.")
    metamap_path = METAMAP_DIR
    mm = MetaMap.get_instance(metamap_path+ "/bin/metamap20")
    
    reference_text = gen_reports_df[['patient_id','conclusion']]
    generated_text = gen_reports_df[['patient_id','gen_conclusion']]
    ref_texts = reference_text['conclusion'].values.tolist()
    gen_texts = generated_text['gen_conclusion'].values.tolist()
    test_size = 10 #len(ref_texts)  # Process all reports
    ref_texts = ref_texts[:test_size]
    gen_texts = gen_texts[:test_size] 
   
    print(f"Processing {test_size} reports with MetaMap...")
    # Create a copy of the input DataFrame
    ce_df = gen_reports_df.copy()
    
    # Initialize new columns
    ce_df['medical_terms_ref'] = None
    ce_df['medical_terms_gen'] = None
    ce_df['medical_terms_overlap'] = 0.0
    
    # Process texts and calculate similarities
    similarities = []
    for idx, (ref_text, gen_text) in enumerate(tqdm(zip(ref_texts, gen_texts), 
                                                  total=test_size, 
                                                  desc="Processing reports")):
        ref_terms = extract_medical_terms(ref_text,mm)
        gen_terms = extract_medical_terms(gen_text,mm)
        
        # Store medical terms in DataFrame (as list of tuples)
        ce_df.at[idx, 'medical_terms_ref'] = list(ref_terms)
        ce_df.at[idx, 'medical_terms_gen'] = list(gen_terms)
        
        # Calculate Jaccard similarity between sets of CUIs
        similarity = calculate_overlap(ref_terms, gen_terms)
        ce_df.at[idx, 'medical_terms_overlap'] = similarity
        similarities.append(similarity)
        
        # Print first few examples for debugging
        if idx < 2:
            print(f"\nExample {idx + 1}:")
            print(f"Reference text: {ref_text[:100]}...")
            print(f"Generated text: {gen_text[:100]}...")
            print(f"Number of medical terms in reference: {len(ref_terms)}")
            print(f"Number of medical terms in generated: {len(gen_terms)}")
            print(f"Medical terms in reference: {[name for _, name in ref_terms]}")
            print(f"Medical terms in generated: {[name for _, name in gen_terms]}")
            print(f"Similarity score: {similarity:.4f}")

    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    print(f"\nResults:")
    print(f"Mean medical terms similarity: {mean_similarity:.4f}")
    print(f"Standard deviation: {std_similarity:.4f}")
    
    # Save the DataFrame
    output_path = 'ce_df_text.csv'
    ce_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    
    return similarities, mean_similarity, std_similarity, ce_df

# Run the new MetaMap-based analysis
""" try:
    similarities, mean_similarity, std_similarity, ce_df = clinical_metrics_ouh_metamap(gen_reports_df)
except Exception as e:
    print(f"Failed to run MetaMap analysis: {str(e)}") """

#ouh ce attempt 2 meta map attempt part 2
def get_pretty_labels(text, mm):
    try:
        # Extract concepts for the single text
        concepts, err = mm.extract_concepts(text)
        if err:
            print(f"MetaMap error: {err}")
            return set()
        
        # Relevant semantic types for filtering (translated from short format to T-codes)
        relevant_types = {
            "T047",  # Disease or Syndrome (was 'dsyn')
            "T033",  # Finding (was 'fndg')
            "T191",  # Neoplastic Process (was 'neop')
            "T184",  # Sign or Symptom (was 'sosy')
            "T017",  # Anatomical Structure (was 'anst')
            "T023"   # Body Part, Organ, or Organ Component (was 'bpoc')
        }
        
        # Extract unique CUIs with their preferred names
        cuis_with_names = set()
        for concept in concepts:
            if concept.cui:  # Only include concepts with CUIs
                # Filter for disease-related terms
                if any(sem_type in relevant_types for sem_type in concept.semtypes):
                    cuis_with_names.add((concept.cui, concept.preferred_name))
        
        return cuis_with_names
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return set()

def clinical_metrics_ouh_metamap2(gen_reports_df):
    from pymetamap import MetaMap
    if not METAMAP_DIR:
        raise FileNotFoundError("Set SCRIBE_METAMAP_DIR to the MetaMap public_mm directory.")
    metamap_path = METAMAP_DIR
    mm_pos_server_dir = metamap_path + "/bin/skrmedpostctl"
    mm_wsd_server_dir = metamap_path + "/bin/wsdserverctl"

    # Start servers
    os.system(f"{mm_pos_server_dir} start")
    os.system(f"{mm_wsd_server_dir} start")
    sleep(10) # Wait for servers to start
    reference_text = gen_reports_df[['patient_id','conclusion']]
    generated_text = gen_reports_df[['patient_id','gen_conclusion']]
    
    try:
        # Initialize MetaMap
        mm = MetaMap.get_instance(metamap_path + "/bin/metamap20")
        
        # Convert to lists
        ref_texts = reference_text['conclusion'].values.tolist()
        gen_texts = generated_text['gen_conclusion'].values.tolist()
        test_size = 2
        ref_texts = ref_texts[:test_size]
        gen_texts = gen_texts[:test_size]
        
        # Extract concepts
        print("extracting labels")
        similarities = []
        for idx, (ref_text, gen_text) in enumerate(tqdm(zip(ref_texts, gen_texts), 
                                                  total=test_size, 
                                                  desc="Processing reports")):
            ref_terms = get_pretty_labels(ref_text, mm)
            gen_terms = get_pretty_labels(gen_text, mm)

            similarity = calculate_overlap(ref_terms, gen_terms)
            similarities.append(similarity)
            
            # Print first few examples for debugging
            if idx < 2:
                print(f"\nExample {idx + 1}:")
                print(f"Reference text: {ref_text[:100]}...")
                print(f"Generated text: {gen_text[:100]}...")
                print(f"Number of medical terms in reference: {len(ref_terms)}")
                print(f"Number of medical terms in generated: {len(gen_terms)}")
                print(f"Medical terms in reference: {[name for _, name in ref_terms]}")
                print(f"Medical terms in generated: {[name for _, name in gen_terms]}")
                print(f"Similarity score: {similarity:.4f}")
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        print(f"\nResults:")
        print(f"Mean medical terms similarity: {mean_similarity:.4f}")
        print(f"Standard deviation: {std_similarity:.4f}")
        return similarities, mean_similarity, std_similarity
        
    finally:
        # Stop servers
        os.system(f"{mm_pos_server_dir} stop")
        os.system(f"{mm_wsd_server_dir} stop")

# Run analysis
#similarities, mean_similarity, std_similarity = clinical_metrics_ouh_metamap2(gen_reports_df)



#green metric  - attempt 4 for ouh metric
def green_CE(reference_text, generated_text):
    "Very slow but interesting metric"
    if GREEN_DIR:
        sys.path.append(os.path.abspath(GREEN_DIR))
    from green_score.green import GREEN
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scribe/ouh_clinical_metrics_test.py RUN [RUN ...]")
        sys.exit(1)
    main(sys.argv[1:])
