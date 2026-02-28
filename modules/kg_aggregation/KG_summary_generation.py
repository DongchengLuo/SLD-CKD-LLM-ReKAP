# -*- coding: utf-8 -*-
import pandas as pd
import json
import pathlib
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# --- 1. Global Configuration Area ---
# ==============================================================================

# --- Local Model Configuration ---
# [IMPORTANT] URL for your local inference server (e.g., vLLM, Ollama, or FastAPI)
LOCAL_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# [IMPORTANT] Name of your local model
MODEL_NAME = "Kimi-K2"
MODEL_TEMPERATURE = 0.2

# Number of concurrent requests sent to the local server.
# Adjust this carefully based on your VRAM, as processing long texts consumes significant memory.
MAX_CONCURRENT_REQUESTS = 2  

# --- Data Path Configuration ---
# TODO: Update these paths to point to your actual dataset files before running
# Point this to the directory containing your PLAIN TEXT (.txt) literature or individual patient files
INPUT_TXT_DIR = './data/txt_files' 
INPUT_SUMMARY_CSV = './data/analysis_summary.csv'
REPORT_OUTPUT_DIR = './output/report_KG'

# --- CSV Column Name Configuration ---
# Change these to match the exact column headers in your CSV file.
# This makes the script agnostic to whether it is processing clusters or individual patients.
# Example for individuals: 'Patient_ID' and 'Patient_Fields'
# Example for clusters: 'Cluster_ID' and 'Cluster_Fields'
CSV_ID_COLUMN = 'Target_ID' 
CSV_FIELD_COLUMN = 'Target_Fields' 

# --- Base Prompt Template ---
BASE_PROMPT_TEMPLATE = '''
Part 1: Role, Mission, and Final Objective  
    • [Role]  
        ◦ You are a top-tier clinical epidemiologist and risk-modelling methodology expert.  
    • [Core Mission]  
        ◦ Your task is to generate a comprehensive methodological report on **“how to assess future chronic kidney disease (CKD) risk in people with non-alcoholic fatty liver disease (NAFLD)”**, based solely on the provided scientific literature text (knowledge base) and a provided list of patient data fields.  
    • [Key Output Requirements]  
        ◦ Nature of Output:  
            ▪ An evidence-based clinical decision-support report.  
            ▪ It must simulate how a clinical expert would systematically synthesise, weigh and interpret a large body of literature to formulate a professional risk-assessment approach.  
            ▪ The focus is on **“how to perform a composite assessment”**, not on “how to build a prediction model”.  
        ◦ Core Content:  
            ▪ Systematically answer: under the current body of evidence, which of the supplied patient fields should be identified, quantified and jointly used to estimate CKD risk.  
        ◦ Target Audience:  
            ▪ Clinical researchers and data scientists.  
            ▪ The report must combine clinical insight with methodological rigour.  

Part 2: Inputs & Knowledge-Base Structure  
        ◦ Target Patient Fields: a list of patient-level variables {instance_field}.  
    • [Framework for Understanding the Knowledge Base]  
        ◦ While analysing the text, internally build a structured knowledge map containing:  
            ▪ Document Metadata: title, year, journal, impact factor, etc. (impact factor is one metric of evidence quality).  
            ▪ Study Context: design (e.g. cohort, RCT), objective, sample size, baseline population characteristics; study design is the primary basis for grading evidence.  
            ▪ Extracted Entities: biomarkers, diseases, risk factors, drugs mentioned.  
            ▪ Extracted Relationships: demonstrated associations (e.g. “A causes B” or “A positively correlates with B”)—your main evidence source.  
            ▪ Systemic Interactions: crucial—pay special attention to discussions/conclusions on multi-factor interactions, synergies or antagonisms.  

Part 3: Required Report Structure  
Your final report must be clearly structured and contain at least the following sections:  
    1. Introduction  
        ◦ Briefly state why CKD-risk assessment in NAFLD matters.  
        ◦ Clarify that this report aims to build a methodological framework based on the supplied literature.  
    2. Relevant Risk-Factor Identification & Mapping  
        ◦ List which fields in the “target list” map to CKD risk factors documented in the literature.  
        ◦ Apply a two-stage mapping:  
            ▪ A. Primary Risk Factors – Direct & Quantifiable Links:  
                • Identify fields proven in the knowledge base as independent, quantifiable CKD risk factors.  
                • List them and note their risk nature (demographic, metabolic, liver-severity, etc.).  
            ▪ B. Contextual & Associated Factors – Indirect or Conceptual Links:  
                • For all remaining fields, perform a second “contextual” search.  
                • Guidance:  
                    ◦ Broaden the Concept: if a specific field (e.g. apple_intake) is absent, search its broader parent (e.g. “diet”, “lifestyle”).  
                    ◦ Identify Indirect Links: check whether the field relates to management/background of a primary factor (e.g. socioeconomic status affects diabetes adherence).  
                    ◦ Summarise the Context:  
                        – First, state explicitly: “Based on the current knowledge base these fields are not independent, quantifiable CKD predictors.”  
                        – Then add any background mention (e.g. “However, the knowledge base highlights ‘healthy diet’ as part of overall lifestyle intervention…”).  
            ▪ Handle Truly Irrelevant Fields:  
                • Only if a field plus all parent concepts are completely absent may you classify it as “no relevance found in current knowledge base”.  
    3. Risk Quantification & Evidence Appraisal  
        ◦ For each identified factor provide quantitative estimates (HR, OR) and statistical significance (p-value).  
        ◦ Grade the quality of each supporting piece of evidence by integrating:  
            ▪ Study design (RCT > prospective cohort > retrospective).  
            ▪ Study quality (journal impact factor, sample size).  
            ▪ Effect size magnitude and consistency (confidence intervals).  
    4. Multi-Factor Interactions & Conflict Resolution  
        ◦ Interaction analysis: explore synergistic or antagonistic effects among factors.  
        ◦ Cross-study synthesis:  
            ▪ Corroboration: if multiple high-quality studies agree, highlight as strong evidence.  
            ▪ Conflict Resolution: when studies disagree:  
                • Contrast designs, populations, methods to explain discordance.  
                • Give a directional judgement based on evidence hierarchy (e.g. favour the more rigorous design).  
    5. Recommendations for a Composite Risk Model  
        ◦ Propose an integrative risk-assessment strategy based on the above.  
        ◦ Discuss statistical issues when combining factors: independence, multicollinearity, interaction terms.  
        ◦ Recommend prioritising factors with higher evidence grade, stronger effect size and cross-study consistency.  
    6. Practical Considerations: Guiding Principles for Missing Data  
        ◦ This section is essential. Provide explicit guidance on how to adapt the assessment when key variables are missing in a real-case scenario. Include at least:  
            ▪ Tier-Based Downgrade: refer to the report’s “tiered assessment strategy”. If data for a higher tier (e.g. liver-fibrosis score) are missing, downgrade certainty and rely on the highest satisfied tier (e.g. basic metabolic markers).  
            ▪ Acknowledge Limitations: warn that absence of key metrics (especially high-weight factors like fibrosis) markedly reduces accuracy; label any conclusion “preliminary” or “incomplete”.  
            ▪ Use of Proxies: suggest usable surrogates—e.g. if FIB-4 cannot be calculated but history shows “cirrhosis”, imaging proves fibrosis, or platelets are clearly low, incorporate these indirect signs.  
            ▪ Recommend Data Collection: emphasise that for high-weight missing data (especially basic labs needed for FIB-4/NFS) the first action should be “order tests to obtain the key values”, not accept an incomplete assessment.  
    7. Limitations  
        ◦ State limitations of the current knowledge base and any research areas not covered.  

Part 4: Your Thinking Process & Workflow  
Follow this sequence exactly:  
    • [Primary Workflow]  
        1. Step 1 – Deconstruct & Map: precisely match the “target field list” to key entities in the knowledge base.  
        2. Step 2 – Find & Extract Evidence: retrieve the core relationship between each mapped entity and “CKD”.  
        3. Step 3 – Assess Evidence Strength: grade quality, recording source, design, p-value and risk metrics.  
        4. Step 4 – Investigate Interactions: revisit source texts, focusing on any discussion of systemic interactions.  
        5. Step 5 – Synthesize & Report: integrate all findings into the final report following Part 3 structure.  
    • [Advanced Research Patterns]  
        1. Pattern 1 – Global Exploration  
            ▪ When: initial clues scarce or important factors possibly missing.  
            ▪ Action: rapidly scan abstracts/conclusions of all texts to build a global NAFLD-CKD risk map, then zoom in.  
        2. Pattern 2 – Pivoting & Deep Dive  
            ▪ When: a pivotal paper (e.g. high-quality review) or recurrent key entity (e.g. “insulin resistance”) appears.  
            ▪ Action: use that paper/entity as a pivot to excavate all internal entities, relationships and context.  
        3. Pattern 3 – Comparative Analysis  
            ▪ When: need to verify consistency of a specific association (e.g. “hypertension → CKD risk”) across papers.  
            ▪ Action: horizontally contrast findings across all papers to address corroboration vs conflict.  
        4. Pattern 4 – Contextual Inquiry  
            ▪ When: a key relationship is found but its conditions are unclear.  
            ▪ Action: analyse the study’s “design & context” and “population characteristics” to define the applicable population.
'''


# ==============================================================================
# --- 2. Helper Functions ---
# ==============================================================================

def ensure_dir_exists(directory_path: str):
    """Ensures that a directory exists, creates it if not."""
    path = pathlib.Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)


def get_target_id_from_filename(filename):
    """Extracts the target ID (cluster or individual) from filenames."""
    # Returning the stem directly as a string to support both numeric and alphanumeric IDs
    return pathlib.Path(filename).stem


def process_single_text_file(txt_path, target_id, instance_field, output_dir):
    """Processes a single plain text file using the local LLM."""
    print(f"[Thread] Starting processing for Target ID={target_id}, file={txt_path.name}")
    output_path = output_dir / f"{target_id}.txt"
    
    try:
        # 1. Read the plain text content
        source_text = txt_path.read_text(encoding='utf-8')
        
        # 2. Format the base prompt with target fields
        base_prompt = BASE_PROMPT_TEMPLATE.format(
            instance_field=json.dumps(instance_field, indent=2, ensure_ascii=False)
        )
        
        # 3. Combine prompt with the source literature text
        final_prompt = f"{base_prompt}\n\n--- Source Literature Text ---\n{source_text}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": MODEL_TEMPERATURE,
        }
        
        # 4. Request the local API with basic retry logic
        max_retries = 3
        result_text = ""
        for attempt in range(max_retries):
            try:
                # Set a high timeout since analyzing large chunks of text takes time
                response = requests.post(LOCAL_API_URL, headers=headers, json=payload, timeout=600)
                response.raise_for_status()
                response_data = response.json()
                result_text = response_data["choices"][0]["message"]["content"].strip()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Local API request failed after {max_retries} attempts. Error: {e}")
                print(f"[Warning] Local API error for ID={target_id}, retrying in {2**attempt}s... ({e})")
                time.sleep(2 ** attempt)

        # 5. Write the response to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"[SUCCESS] Target ID={target_id} report saved to {output_path}")

    except Exception as e:
        error_msg = f"[ERROR] Failed to process Target ID={target_id}: {str(e)}"
        print(error_msg)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(error_msg)


# ==============================================================================
# --- 3. Main Logic ---
# ==============================================================================

def main():
    start_time = time.time()
    print("======================================================")
    print("=== Local KG Summary Generation (Plain Text Mode) ===")
    print("======================================================")

    # Load summary CSV
    try:
        summary_df = pd.read_csv(INPUT_SUMMARY_CSV)
        
        # Convert the ID column to string to ensure correct matching with filename stems
        summary_df[CSV_ID_COLUMN] = summary_df[CSV_ID_COLUMN].astype(str)
        
        # Create a dictionary mapping the target ID to the target fields
        id_to_instance_map = dict(zip(summary_df[CSV_ID_COLUMN], summary_df[CSV_FIELD_COLUMN]))
    except Exception as e:
        print(f"[Fatal Error] Failed to read CSV file at {INPUT_SUMMARY_CSV}: {e}")
        return

    txt_dir = pathlib.Path(INPUT_TXT_DIR)
    output_dir = pathlib.Path(REPORT_OUTPUT_DIR)
    ensure_dir_exists(str(output_dir))

    # Gather tasks
    tasks = []
    for file in txt_dir.glob("*.txt"):
        target_id = get_target_id_from_filename(file.name)
        if not target_id:
            print(f"[WARN] Ignoring invalid filename: {file.name}")
            continue

        # Check if result already exists to allow safe resuming
        output_txt_path = output_dir / f"{target_id}.txt"
        if output_txt_path.exists():
            print(f"[SKIP] Result for Target ID={target_id} already exists, skipping.")
            continue

        instance_field = id_to_instance_map.get(target_id)
        if instance_field is None:
            print(f"[WARN] Target ID={target_id} not found in CSV map, skipping.")
            continue

        tasks.append((file, target_id, instance_field))

    if not tasks:
        print("\n[INFO] No pending tasks found. Exiting.")
        return

    print(f"\n--- Found {len(tasks)} valid tasks to process ---")
    print(f"--- Starting {MAX_CONCURRENT_REQUESTS} concurrent local threads ---")

    # Process tasks concurrently
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = [
            executor.submit(process_single_text_file, file_path, target_id, instance_field, output_dir)
            for file_path, target_id, instance_field in tasks
        ]
        
        for future in as_completed(futures):
            # The exception handling is done inside process_single_text_file
            # so we just wait for all threads to finish here.
            pass

    end_time = time.time()
    print("\n==================================================")
    print(f"✅ All processing completed. Total time: {end_time - start_time:.2f} seconds.")
    print(f"💾 Reports saved to: {REPORT_OUTPUT_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()