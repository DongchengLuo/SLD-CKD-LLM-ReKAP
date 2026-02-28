import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder
import shap
import functools  # Used to "freeze" function parameters
import re
import itertools
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
import numpy as np

# --- 1. CONFIGURATION ---

# -- API and Model Configuration --
# IMPORTANT: For security, use environment variables instead of hardcoding in public repositories.
API_KEY = "YOUR_API_KEY_HERE"
# Replace with your actual local or remote API endpoint
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "kimi-k2-instruct-0905"

# -- SHAP Configuration --
MAX_WORKERS = 16
RETRY_ATTEMPTS = 5
TARGET_SCORE_KEY = "5_year_risk_percent"

# --- NEW: PartitionExplainer and k-means sampling configuration ---
# !! CRITICAL: Set the "sweet spot" nsamples value determined in your convergence testing
MANUAL_NSAMPLES = 5
# !! CRITICAL: Set the number of mean sampling iterations (k)
MEAN_SAMPLING_K = 3

# -- Batch Processing Configuration --
START_INDEX = 0
END_INDEX = 5

# -- File Paths --
# TODO: Update these paths to point to your actual dataset files before running
# !! CRITICAL: Path to the CSV file containing 'key' and 'mapped_category_name' columns
CLUSTER_MAPPING_CSV = "./data/cluster_mapping.csv" 
ENCODING_MAP_PATH = "./data/encoding_map_shap.json"
REPORT_1_DIR = "./data/report_1_dir/"
REPORT_2_PATH = "./data/report_2_general_population.txt"
CLUSTER_SAMPLES_CSV = "./data/cluster_samples.csv"
PARTICIPANT_DATA_JSON = "./data/participant_data.json"
PREVIOUS_REASONING_JSON = "./data/previous_reasoning.json"

# !! MODIFICATION: Use a new output file to avoid overwriting old results
SHAP_OUTPUT_JSON = "./output/shap_partition_results_down1.json"


# --- 2. PROMPT TEMPLATE (Unchanged) ---
PROMPT_TEMPLATE = """
You are a top-tier clinical AI reasoning expert, specializing in the intersection of nephrology and hepatology. Your core mission is to synthesize evidence from multiple independent sources, including two expert reports and a previous reasoning log, to generate an authoritative, de novo risk assessment for Chronic Kidney Disease (CKD) for a specific UKBB participant with fatty liver disease.

You will receive four distinct sources of information:
1.  **A methodological report on CKD risk for a specific population ("Report 1").**
2.  **A report on a well-established CKD risk model for the general population ("Report 2").**
3.  **The participant's individual health data.**

Your analysis must be a deep synthesis of all sources.

--- Evidence Base and Background ---

**Evidence Source One: Specific Population Risk Perspective (e.g., a report on CKD risk in a fatty liver disease population)**
{report_1_content}
* **Role and Value**: This report provides in-depth insights into CKD risk within a specific disease context (such as fatty liver disease). It may contain detailed analysis of risk phenotypes unique to this population, key biomarkers, and distinct risk interactions.

**Evidence Source Two: General Population Risk Perspective (e.g., a report on a general risk model like CKD-PC)**
{report_2_content}
* **Role and Value**: This report offers a robust risk assessment framework validated in large-scale, multi-ethnic cohorts. It includes core risk factors with widely confirmed efficacy and their quantitative weights, serving as a solid benchmark and providing supplementary evidence for factors that may not be quantitatively detailed in 'Report 1' (such as the Urine Albumin-to-Creatinine Ratio, UACR).


--- Core Task: Evidence Synthesis and Critical Thinking Guide ---

Your core task is to **reconcile, integrate, and elevate** all information sources. You cannot simply choose one report and ignore the other; both reports are of equal importance. You must follow these critical thinking guidelines:

1.  **Identify Consensus, Establish Strong Evidence**:
    * When "Report 1," "Report 2," and the previous reasoning log align on a risk factor (e.g., age, diabetes), it should be treated as the highest level of evidence.

2.  **Analyze Differences, Conduct Critical Weighing**:
    * When sources offer unique or seemingly contradictory viewpoints (e.g., one report emphasizes BMI while another focuses on the "lean fatty liver" paradox), you must engage in deep critical thinking. Your reasoning process must clearly articulate:
        * **Contextual Specificity**: Does "Report 1" (specific population) offer a viewpoint that is more explanatory in a specific pathophysiological context than a general rule?
        * **Evidence Strength**: Does "Report 2" (general model) include a key variable validated in larger-scale studies that is not included in "Report 1"?
        * **Decision-Making**: How do you ultimately decide to adopt, merge, or adjust this information? For example, you might decide to use the variables from "Report 2" as the primary stratification tool while using the special phenotypes from "Report 1" to fine-tune the risk within that stratification (or vice versa).

3.  **Construct a Comprehensive, Multidimensional Argument**:
    * Your final assessment should not be a simple average or a choice between viewpoints but must be a new, coherent, and synthesized conclusion. You need to clearly argue why certain pieces of evidence were prioritized in the final risk calculation. Use evidence from one source to **validate, supplement, or challenge** evidence from another, thereby constructing a logically rigorous and sound risk profile.

4.  **Maintain Flexibility, No Pre-set Starting Point**:
    * Both reports are solid foundations for your analysis. You can start from the strengths of either report and use the insights from the other to supplement and refine it, or vice versa. The key is to build the most comprehensive and reasonable risk assessment.

--- Guiding Principles for Reasoning (Few-Shot Scoring Principles) ---
Disclaimer: The following are concise guiding principles designed to demonstrate core reasoning patterns and are not exhaustive. For any new case, you must conduct a more complex and comprehensive, individualized reasoning process based on the totality and uniqueness of the evidence presented.

Principle 1: Synergistic Risk - Assign High Weight
Scenario: The patient presents with multiple metabolic abnormalities (e.g., hyperlipidemia, hyperuricemia, fatty liver indicators).
Reasoning: The synergistic effect of this cluster of abnormalities poses a risk far greater than the sum of its individual parts. This cluster should be assigned a disproportionately high positive weight.

Principle 2: Strong Protective Factor - Assign Negative Weight
Scenario: The patient exhibits one or more powerful protective factors, such as a long-term, consistent level of physical activity far exceeding recommended guidelines.
Reasoning: An exceptionally high level of a positive behavior provides a potent risk mitigation effect and should be assigned a strong negative (protective) weight.

Principle 3: Risk Correlation & Avoiding Double-Counting
Scenario: The patient has numerous negative health indicators that are highly correlated (e.g., severe obesity, hypertension, dyslipidemia).
Reasoning: Avoid "double-counting" risk. The primary drivers (e.g., severe obesity) should receive the majority of the risk weight, while the correlated secondary markers contribute only a minor additional weight.

Principle 4: Confirming Absence of Major Accelerators
Scenario: The patient's clinical profile suggests a potential for a major, uncaptured risk accelerator.
Reasoning: If supplementary information clearly rules out the presence of this major risk accelerator, it increases confidence that the risk is not being driven by this specific high-impact pathway.

Principle 5: Strong Independent Factor - Assign Major Weight
Scenario: The patient has a strong risk (or protective) factor that has a low degree of biological pathway correlation with other observed factors.
Reasoning: When a factor's mechanism is relatively independent, the risk (or benefit) it confers is a purely additive increment. The weight assigned to it should be higher.

Principle 6: Data Quality Scrutiny - Modulate Confidence & Weight
Scenario: A key piece of information has quality issues (e.g., a crucial lab test is from several years ago).
Reasoning: The weight assigned to decisions based on old, vague, or low-quality evidence must be significantly dampened, and the confidence_score must reflect this uncertainty.

Principle 7: Competing Risks Consideration
Scenario: A patient has an extremely severe, non-renal comorbidity with a high short-to-medium term mortality risk.
Reasoning: The long-term risk of developing CKD is influenced by the "competing risk" of death from another cause. For very long-term predictions (10-15 years), a modest downward dampening factor should be applied.


--- End of Principles ---

Required JSON Output Format:
```json
{{
    "evidence_summary": "<A concise, neutral summary of the key findings from the participant's information, noting the primary source of evidence for each key point (which report, or a consensus of multiple reports).>",
    "reasoning_process": "<**You must follow these steps, citing the evidence source (Report 1, Report 2, or consensus) for your evidence-based quantitative reasoning:** 1. **Foundational Risk Factor Inventory (Synthesized View):** Systematically identify and categorize all relevant risk and protective factors from the participant's data. 2. **Integrative Risk Calculation and Pathway Analysis:** Construct the risk score through a weighted synthesis of all evidence. a. **Identify Intersections and Differences in Evidence:** Clearly state which risk assessments are based on multi-source consensus and which are based on the unique insights of a specific report. b. **Critical Weight Allocation:** Assign a quantitative risk contribution to each factor. When information conflicts, clearly articulate which viewpoint you are adopting and why, and explain how you will handle it to avoid double-counting (Principle 3). c. **Identify and Quantify Synergies:** Actively seek out and quantify synergistic (1+1>2) or antagonistic effects between factors. 3. **Time Horizon and Competing Risk Analysis:** Project the synthesized risk over 3, 5, 10, and 15-year intervals, explaining the evolution of the risk trajectory and and considering competing risks (Principle 7). 4. **Final Score Synthesis and Sanity Check:** Aggregate all weighted contributions into the final risk scores for each time point and perform a final 'Clinical Plausibility Sanity Check,' explaining any differences between your synthesized assessment and the prediction a single model might produce, and justifying the rationale for these differences.>",
    "risk_scores": {{
        "3_year_risk_percent": <float>,                
        "5_year_risk_percent": <float>,
        "10_year_risk_percent": <float>,
        "15_year_risk_percent": <float>
    }},
    "confidence": {{
        "confidence_score": <float, from 0.0 to 1.0>,
        "confidence_reasoning": "<Justify the confidence score based on the quality, completeness, and consistency of the available data. Specifically mention the availability of core metrics emphasized in the key reports (e.g., the components for calculating liver fibrosis scores).>"
    }}
}}
```

Now, analyze the following new data:

Participant Information:
{patient_data_json}

Begin your analysis, and return your prediction strictly following the JSON format and the evidence synthesis and critical thinking guide provided above.

"""


# --- 3. Data Loading (Modified: Includes clustering map) ---
def load_all_data_sources():
    """Load all necessary data files into memory and slice based on index."""
    print("[INFO] Loading all data files...")
    try:
        with open(REPORT_2_PATH, 'r', encoding='utf-8') as f:
            report_2_content = f.read()
        cluster_df = pd.read_csv(CLUSTER_SAMPLES_CSV)
        cluster_df['sample_id'] = cluster_df['sample_id'].astype(str)
        sample_to_cluster_map = cluster_df.set_index('sample_id')['cluster'].to_dict()
        with open(PARTICIPANT_DATA_JSON, 'r', encoding='utf-8') as f:
            participant_data_all = json.load(f)

        with open(PREVIOUS_REASONING_JSON, 'r', encoding='utf-8') as f:
            previous_reasoning_data_all = json.load(f)

        # --- Load Clustering Mapping File ---
        try:
            cluster_map_df = pd.read_csv(CLUSTER_MAPPING_CSV)
            # Make sure your CSV contains columns named 'key' and 'mapped_category_name'
            feature_to_cluster_map = cluster_map_df.set_index('key')['mapped_category_name'].to_dict()
            cluster_to_features_map = {}
            for feature, cluster in feature_to_cluster_map.items():
                if cluster not in cluster_to_features_map:
                    cluster_to_features_map[cluster] = []
                cluster_to_features_map[cluster].append(feature)
            print(f"[INFO] Successfully loaded cluster mapping: {len(feature_to_cluster_map)} features -> {len(cluster_to_features_map)} clusters.")
        except FileNotFoundError as e:
            print(f"[ERROR] Cluster mapping file not found - {e}. Please check the CLUSTER_MAPPING_CSV path.")
            exit(1)
        except KeyError as e:
            print(f"[ERROR] Cluster mapping CSV file '{CLUSTER_MAPPING_CSV}' must contain 'key' and 'mapped_category_name' columns. Not found {e}.")
            exit(1)

        # --- !! CRITICAL MODIFICATION: Load Encoding Mapping ---
        try:
            with open(ENCODING_MAP_PATH, 'r', encoding='utf-8') as f:
                encoding_map = json.load(f)
            print(f"[INFO] Successfully loaded encoding map: {len(encoding_map)} fields.")
        except FileNotFoundError:
            print(f"[WARNING] {ENCODING_MAP_PATH} not found, using an empty mapping.")
            encoding_map = {}

        # --- !! CRITICAL MODIFICATION: Build Reverse Decoding Mapping (Numeric -> Original Value) ---
        reverse_encoding_map = {}
        for feature, value_map in encoding_map.items():
            reverse_encoding_map[feature] = {int(v): k for k, v in value_map.items()}
        print(f"[INFO] Successfully built reverse encoding map: {len(reverse_encoding_map)} fields.")

        # --- !! CRITICAL MODIFICATION: Build Global Background Dataset (Using all tmp samples) ---
        print(f"[INFO] Building global background dataset (using all {len(participant_data_all)} samples)...")

        # Helper function: Convert samples to numeric values
        def preprocess_to_numeric(sample_dict, encoding_map):
            numeric_dict = {}
            for key, value in sample_dict.items():
                if pd.isna(value) or value is None:
                    numeric_dict[key] = np.nan
                elif isinstance(value, (int, float)):
                    numeric_dict[key] = float(value)
                elif key in encoding_map and str(value) in encoding_map[key]:
                    numeric_dict[key] = float(encoding_map[key][str(value)])
                else:
                    try:
                        num_val = pd.to_numeric(value, errors='coerce')
                        numeric_dict[key] = num_val if pd.notna(num_val) else np.nan
                    except:
                        numeric_dict[key] = np.nan
            return numeric_dict

        # Convert all samples
        all_samples_numeric = []
        for sample_id, sample_dict in participant_data_all.items():
            all_samples_numeric.append(preprocess_to_numeric(sample_dict, encoding_map))

        # Get all possible feature names (union of sets)
        all_feature_names = sorted(list(set().union(*[set(s.keys()) for s in all_samples_numeric])))

        # Construct DataFrame
        background_df = pd.DataFrame(all_samples_numeric)[all_feature_names]
        background_df = background_df.fillna(np.nan).astype(float)
        print(f"[INFO] Global background dataset build completed: shape={background_df.shape}")

        # --- Slice Target Samples ---
        print(f"[INFO] Loaded {len(participant_data_all)} samples from {PARTICIPANT_DATA_JSON}.")
        print(f"[INFO] Slicing, index range: [{START_INDEX}:{END_INDEX}]")
        sliced_items = itertools.islice(participant_data_all.items(), START_INDEX, END_INDEX)
        participant_data_sliced = dict(sliced_items)
        print(f"[INFO] Data loading complete. Will process {len(participant_data_sliced)} participant records.")

        return (
            report_2_content, sample_to_cluster_map, participant_data_sliced,
            previous_reasoning_data_all, feature_to_cluster_map, cluster_to_features_map,
            encoding_map, reverse_encoding_map, background_df  # !! Newly returned
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found - {e}. Please check the paths in the script.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during data loading: {e}")
        exit(1)


# --- 4. SHAP Integration Core Logic ---

def load_static_context_for_sample(sample_id, report_2_content, sample_to_cluster_map, previous_reasoning_data_all):
    """Load all 'static' context for a *single* target sample."""
    print(f"  [INFO] Loading static context for sample {sample_id}...")
    cluster_id = sample_to_cluster_map.get(sample_id)
    if cluster_id is None:
        raise ValueError(f"[ERROR] Sample ID {sample_id} not found in cluster_samples.csv.")
    
    report_1_path = os.path.join(REPORT_1_DIR, f"{cluster_id}.txt")
    try:
        with open(report_1_path, 'r', encoding='utf-8') as f:
            report_1_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Cluster report {report_1_path} not found.")
    
    previous_reasoning = previous_reasoning_data_all.get(sample_id, {})
    evidence_summary = previous_reasoning.get("evidence_summary", "Not available.")
    reasoning_process_text = previous_reasoning.get("reasoning_process", "Not available.")
    previous_reasoning_log_str = (
        f"--- Previous Reasoning Log ---\n"
        f"Evidence Summary:\n{evidence_summary}\n\n"
        f"Reasoning Process:\n{reasoning_process_text}\n"
        f"--- End of Previous Reasoning Log ---"
    )
    
    static_context = {
        "report_1_content": report_1_content,
        "report_2_content": report_2_content,
        "previous_reasoning_log": previous_reasoning_log_str
    }
    print(f"  [INFO] Static context loaded (Sample {sample_id}).")
    return static_context


def format_prompt_for_sample(participant_dict_masked, static_context):
    """Format the final Prompt using 'masked' participant data and 'static' context."""
    participant_data_to_serialize = {
        key: value
        for key, value in participant_dict_masked.items()
        if value is not None
    }
    patient_data_json_str = json.dumps(participant_data_to_serialize)
    final_prompt = PROMPT_TEMPLATE.format(
        report_1_content=static_context["report_1_content"],
        report_2_content=static_context["report_2_content"],
        previous_reasoning_log=static_context["previous_reasoning_log"],
        patient_data_json=patient_data_json_str
    )
    return final_prompt


def call_llm_with_retry(api_client, prompt, attempt=1):
    """Call LLM API with retry logic."""
    try:
        completion = api_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32768,
            temperature=0.0,
            stream=False
        )
        content = completion.choices[0].message.content
        return content
    except Exception as e:
        print(f"    [LLM Call Failure - Attempt {attempt}]: {e}")
        if attempt < RETRY_ATTEMPTS:
            time.sleep(2)
            return call_llm_with_retry(api_client, prompt, attempt + 1)
        else:
            print(f"    [LLM Critical Failure]: Giving up on this attempt.")
            return None


def parse_llm_response(response_content, target_key):
    """Safely parse JSON from LLM's raw text output and extract the target score."""
    if response_content is None:
        return None
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_content
        data = json.loads(json_str)
        score = data.get("risk_scores", {}).get(target_key)
        if score is None:
            print(f"    [Parsing WARNING]: 'risk_scores.{target_key}' not found in JSON.")
            return None
        return float(score)
    except json.JSONDecodeError:
        print(f"    [Parsing ERROR]: Unable to decode JSON from LLM output.")
        return None
    except Exception as e:
        print(f"    [Parsing CRITICAL ERROR]: {e}")
        return None


def worker_fn_for_shap(participant_dict_masked, static_context, api_client, target_key):
    """
    Task executed by a single `ThreadPoolExecutor` worker thread.
    This function will now retry infinitely until it successfully retrieves and parses a valid score.
    """
    final_prompt = format_prompt_for_sample(participant_dict_masked, static_context)

    attempt = 1
    while True:  # Infinite loop until success
        print(f"    [Worker] Attempting SHAP sample for the {attempt}th time...")
        response_content = call_llm_with_retry(api_client, final_prompt)
        if response_content is None:
            print(f"    [Worker] LLM API call completely failed. Waiting 10 seconds before retrying the entire Worker...")
            time.sleep(10)
            attempt += 1
            continue
        score = parse_llm_response(response_content, target_key)
        if score is None:
            print(f"    [Worker] JSON parsing failed. Waiting 5 seconds before retrying the entire Worker...")
            time.sleep(5)
            attempt += 1
            continue
        return score


# --- NEW: Functions for k-means sampling ---
def get_mean_prediction(sample_dict, static_context, api_client, target_key, k_samples):
    """
    Get the average of k predictions for a single sample.
    """
    scores = []
    print(f"  [k-Sampling] Getting base prediction for k={k_samples} iterations...")

    # Note: For small numbers like k=3, parallelization overhead might outweigh benefits.
    # However, to remain consistent with the functions below, we use a thread pool.
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, k_samples)) as executor:
        futures = [
            executor.submit(
                worker_fn_for_shap,
                sample_dict,
                static_context,
                api_client,
                target_key
            ) for _ in range(k_samples)
        ]

        for future in as_completed(futures):
            try:
                score = future.result()  # worker_fn_for_shap will retry infinitely
                scores.append(score)
            except Exception as e:
                # Theoretically, worker_fn_for_shap won't throw exceptions, but as a safeguard
                print(f"  [k-Sampling] A worker failed critically during base prediction: {e}")

    if not scores:
        raise Exception("All workers for k-means sampling failed; unable to obtain base prediction.")

    mean_score = np.mean(scores)
    print(f"  [k-Sampling] Base prediction complete. Scores: {scores} -> Mean: {mean_score:.4f}")
    return mean_score


# --- MODIFICATION: Parallel prediction function for k-means sampling ---
def partition_predict_fn_with_sampling(
        masked_data_array,
        feature_names,
        static_context,
        api_client,
        target_key,
        k_samples,
        original_sample_dict,
        reverse_encoding_map  # !! NEW: Reverse decoding mapping
):
    """
    The *main function* that the SHAP Explainer will call.
    !! FIX: Correctly handle perturbed values passed by SHAP, instead of always using the original values.
    """
    print(f"\n  [SHAP Predict] Received {masked_data_array.shape[0]} perturbed samples (numpy array)...")

    # !! CRITICAL FIX: Reverse decoding function
    def decode_numeric_value(feature, numeric_value):
        """Reverse decode numeric masked_value back to original semantic value"""
        if pd.isna(numeric_value):
            return None

        # If this feature has a reverse mapping (categorical)
        if feature in reverse_encoding_map:
            int_val = int(round(numeric_value))
            return reverse_encoding_map[feature].get(int_val, numeric_value)
        else:
            # Numeric feature, return directly
            return numeric_value

    list_of_masked_dicts = []
    for row in masked_data_array:
        row_dict = {}
        for i, (feature, masked_value) in enumerate(zip(feature_names, row)):
            if pd.isna(masked_value):  # SHAP mask position
                row_dict[feature] = None
            else:
                # !! CRITICAL FIX: Use the perturbed value from SHAP, not the original value!
                row_dict[feature] = decode_numeric_value(feature, masked_value)
        list_of_masked_dicts.append(row_dict)

    # Print validation (first 10 fields of the first 3 perturbations)
    if len(list_of_masked_dicts) > 0:
        for i in range(min(3, len(list_of_masked_dicts))):
            sample = list_of_masked_dicts[i]
            sample_preview = dict(list(sample.items())[:10])
            print(f"  [Validation] Perturbation {i} sample data (first 10 fields): {sample_preview}")

    print(f"  [SHAP Predict] Converted into {len(list_of_masked_dicts)} dictionaries.")
    print(f"  [SHAP Predict] Initiating k={k_samples} mean sampling for {len(list_of_masked_dicts)} perturbations...")
    print(
        f"  [SHAP Predict] Total calls: {len(list_of_masked_dicts)} * {k_samples} = {len(list_of_masked_dicts) * k_samples}")

    final_mean_scores = [None] * len(list_of_masked_dicts)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {}
        for i, masked_dict in enumerate(list_of_masked_dicts):
            for k in range(k_samples):
                future = executor.submit(
                    worker_fn_for_shap,
                    masked_dict,
                    static_context,
                    api_client,
                    target_key
                )
                future_to_index[future] = (i, k)

        scores_collector = [[] for _ in range(len(list_of_masked_dicts))]
        processed_count = 0
        total_jobs = len(list_of_masked_dicts) * k_samples

        for future in as_completed(future_to_index):
            list_index, k_num = future_to_index[future]
            processed_count += 1
            try:
                score = future.result()
                scores_collector[list_index].append(score)
                print(
                    f"    [SHAP Worker {list_index}/{len(list_of_masked_dicts)}, k={k_num + 1}/{k_samples}] Success, Score: {score:.4f} (Global Progress {processed_count}/{total_jobs})")
            except Exception as e:
                print(f"    [SHAP Worker {list_index}, k={k_num + 1}] Critical Crash: {e}. Terminating SHAP calculation for this sample.")
                raise e

    # Calculate means
    for i, scores in enumerate(scores_collector):
        if len(scores) != k_samples:
            print(f"    [SHAP Predict WARNING]: Perturbation {i} only received {len(scores)}/{k_samples} scores.")
            if not scores:
                raise Exception(f"Perturbation {i} received no scores.")
        final_mean_scores[i] = np.mean(scores)

    print(f"  [SHAP Predict] Parallel batch processing and mean sampling completed.")
    return np.array(final_mean_scores)


# --- 5. REFACTORING: PartitionExplainer calculation function for a single sample ---
def calculate_shap_for_sample(
        target_sample_id,
        base_sample_dict,
        report_2_content,
        sample_to_cluster_map,
        previous_reasoning_data_all,
        api_client,
        feature_to_cluster_map,
        encoding_map,
        reverse_encoding_map,
        background_df
):
    """
    Execute full PartitionExplainer SHAP value calculation for a single sample.
    """
    static_context = load_static_context_for_sample(
        target_sample_id,
        report_2_content,
        sample_to_cluster_map,
        previous_reasoning_data_all
    )

    # Use feature names from the background dataset (ensures consistency)
    feature_names = list(background_df.columns)
    num_features = len(feature_names)
    print(f"  Total features (M): {num_features}")

    # Helper function
    def preprocess_to_numeric(sample_dict, encoding_map):
        numeric_dict = {}
        for key, value in sample_dict.items():
            if pd.isna(value) or value is None:
                numeric_dict[key] = np.nan
            elif isinstance(value, (int, float)):
                numeric_dict[key] = float(value)
            elif key in encoding_map and str(value) in encoding_map[key]:
                numeric_dict[key] = float(encoding_map[key][str(value)])
            else:
                try:
                    num_val = pd.to_numeric(value, errors='coerce')
                    numeric_dict[key] = num_val if pd.notna(num_val) else np.nan
                except:
                    numeric_dict[key] = np.nan
        return numeric_dict

    # Build clustering array
    unique_clusters = sorted(list(set(feature_to_cluster_map.values())))
    cluster_name_to_id = {name: i for i, name in enumerate(unique_clusters)}
    unclustered_id = len(unique_clusters)
    id_to_cluster_name = {i: name for name, i in cluster_name_to_id.items()}
    id_to_cluster_name[unclustered_id] = "UNCLUSTERED_FEATURES"

    clustering_array = []
    unclustered_count = 0
    for fname in feature_names:
        cluster_name = feature_to_cluster_map.get(fname)
        if cluster_name is None:
            clustering_array.append(unclustered_id)
            unclustered_count += 1
        else:
            clustering_array.append(cluster_name_to_id[cluster_name])

    num_effective_clusters = len(unique_clusters) + (1 if unclustered_count > 0 else 0)
    print(f"  Features mapped to {num_effective_clusters} effective clusters (M').")
    if unclustered_count > 0:
        print(f"  [WARNING] {unclustered_count} features not found in the cluster mapping file.")

    clustering_numpy_array = np.array(clustering_array)

    # Get base prediction
    print(f"\n  --- Fetching k-means base prediction for sample {target_sample_id} ---")
    base_prediction_score = get_mean_prediction(
        base_sample_dict,
        static_context,
        api_client,
        TARGET_SCORE_KEY,
        MEAN_SAMPLING_K
    )
    print(f"  --- Base prediction mean: {base_prediction_score} ---")

    # !! CRITICAL FIX: Create a numeric DataFrame for the target sample (aligned with background dataset features)
    base_sample_dict_numeric = preprocess_to_numeric(base_sample_dict, encoding_map)

    # !! Fix KeyError: First create an empty DataFrame containing all features, filled with NaN
    base_sample_df = pd.DataFrame(columns=feature_names, dtype=float)

    # Fill in the data for the target sample (only populate existing features)
    for feature in feature_names:
        if feature in base_sample_dict_numeric:
            base_sample_df.loc[0, feature] = base_sample_dict_numeric[feature]
        else:
            base_sample_df.loc[0, feature] = np.nan  # Missing features are NaN

    # Ensure data type is float
    base_sample_df = base_sample_df.astype(float)

    # Tally missing features
    missing_features = [f for f in feature_names if f not in base_sample_dict_numeric]
    if missing_features:
        print(f"  [WARNING] Target sample is missing {len(missing_features)}/{num_features} features, filled with NaN.")
        print(f"  Sample missing features: {missing_features[:5]}")

    print(f"  Target sample DataFrame shape: {base_sample_df.shape}")
    print(f"  Background dataset DataFrame shape: {background_df.shape}")
    print(f"  Number of non-NaN features in target sample: {base_sample_df.notna().sum(axis=1).values[0]}")

    # Create distance matrix
    n_features = len(feature_names)
    dist_matrix = np.ones((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if clustering_array[i] == clustering_array[j]:
                dist_matrix[i, j] = dist_matrix[j, i] = 0

    condensed_dist = pdist(dist_matrix, metric='euclidean')
    linkage_matrix = linkage(condensed_dist, method='average')

    # Create Masker using the global background dataset
    masker = shap.maskers.Partition(background_df, clustering=linkage_matrix, max_samples=10)
    print(f"  Created Masker using global background dataset: {background_df.shape[0]} samples, {background_df.shape[1]} features")

    # Create prediction function (pass reverse_encoding_map)
    shap_predict_function = functools.partial(
        partition_predict_fn_with_sampling,
        feature_names=feature_names,
        static_context=static_context,
        api_client=api_client,
        target_key=TARGET_SCORE_KEY,
        k_samples=MEAN_SAMPLING_K,
        original_sample_dict=base_sample_dict,
        reverse_encoding_map=reverse_encoding_map
    )

    # Run SHAP
    print(f"\n  --- Initializing SHAP PartitionExplainer (Sample {target_sample_id}) ---")
    print(f"  M' = {num_effective_clusters} clusters, M = {num_features} features")
    print(f"  Manual nsamples = {MANUAL_NSAMPLES}")
    print(f"  Mean sampling k = {MEAN_SAMPLING_K}")
    print(f"  Estimated total API calls: {MANUAL_NSAMPLES} * {MEAN_SAMPLING_K} = {MANUAL_NSAMPLES * MEAN_SAMPLING_K}")

    explainer = shap.PartitionExplainer(shap_predict_function, masker)

    print(f"\n  --- Calculating SHAP values (Sample {target_sample_id}) ---")
    start_shap_time = time.time()
    shap_values_explanation = explainer(
        base_sample_df,
        max_evals=MANUAL_NSAMPLES
    )
    duration = time.time() - start_shap_time
    print(f"  --- SHAP calculation completed (Sample {target_sample_id}, Duration: {duration:.2f} seconds) ---")

    # Aggregate results
    print("  ... Aggregating feature SHAP values into clusters...")
    feature_shap_values = shap_values_explanation.values[0]
    cluster_shap_values = {}
    for i, feature_name in enumerate(feature_names):
        cluster_id = clustering_array[i]
        cluster_name = id_to_cluster_name[cluster_id]
        feature_shap = feature_shap_values[i]
        if cluster_name not in cluster_shap_values:
            cluster_shap_values[cluster_name] = 0.0
        cluster_shap_values[cluster_name] += feature_shap

    sorted_cluster_impacts = sorted(
        cluster_shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    result_data = {
        "base_value": shap_values_explanation.base_values[0],
        "prediction_score": base_prediction_score,
        "shap_values_sum": np.sum(feature_shap_values),
        "num_clusters": num_effective_clusters,
        "num_features": num_features,
        "nsamples_run": MANUAL_NSAMPLES,
        "k_mean_sampling": MEAN_SAMPLING_K,
        "cluster_impacts": {cluster: impact for cluster, impact in sorted_cluster_impacts}
    }

    return result_data


# --- 6. Modified Main Workflow (Now a Loop) ---

def run_batch_shap_analysis():
    """
    Main function, iterates through all selected samples and calculates SHAP values for each.
    """
    total_start_time = time.time()

    # !! MODIFICATION: Receive new return values
    try:
        (report_2_content,
         sample_to_cluster_map,
         participant_data_sliced,
         previous_reasoning_data_all,
         feature_to_cluster_map,
         cluster_to_features_map,
         encoding_map,
         reverse_encoding_map,  # !! NEW
         background_df  # !! NEW
         ) = load_all_data_sources()
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return

    api_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    all_shap_results = {}
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(SHAP_OUTPUT_JSON), exist_ok=True)
    
    if os.path.exists(SHAP_OUTPUT_JSON):
        print(f"[INFO] Existing results file detected: {SHAP_OUTPUT_JSON}. Loading...")
        try:
            with open(SHAP_OUTPUT_JSON, 'r', encoding='utf-8') as f:
                all_shap_results = json.load(f)
            print(f"[INFO] Loaded {len(all_shap_results)} existing results.")
        except Exception as e:
            print(f"[WARNING] Failed to load existing results file: {e}. Starting with empty results.")
            all_shap_results = {}

    total_samples = len(participant_data_sliced)
    print(f"\n--- Commencing Batch SHAP Analysis (PartitionExplainer), Total {total_samples} samples ---")

    for i, (target_sample_id, base_sample_dict) in enumerate(participant_data_sliced.items()):
        print(f"\n=======================================================")
        print(f"=== Processing Sample {i + 1}/{total_samples} (ID: {target_sample_id}) ===")
        print(f"=======================================================")

        if target_sample_id in all_shap_results and "error" not in all_shap_results[target_sample_id]:
            print(f"[INFO] Sample {target_sample_id} already exists in the results file. Skipping.")
            continue

        sample_start_time = time.time()
        try:
            # !! MODIFICATION: Pass new parameters
            shap_result = calculate_shap_for_sample(
                target_sample_id,
                base_sample_dict,
                report_2_content,
                sample_to_cluster_map,
                previous_reasoning_data_all,
                api_client,
                feature_to_cluster_map,
                encoding_map,
                reverse_encoding_map,  # !! NEW
                background_df  # !! NEW
            )

            all_shap_results[target_sample_id] = shap_result
            print(f"--- Sample {target_sample_id} successfully processed (Duration: {time.time() - sample_start_time:.2f} seconds) ---")
        except Exception as e:
            print(f"[!! ERROR] Sample {target_sample_id} processing failed: {e}")
            import traceback
            traceback.print_exc()
            all_shap_results[target_sample_id] = {"error": str(e)}

        print(f"  ... Saving {len(all_shap_results)} accumulated results in real-time to {SHAP_OUTPUT_JSON} ...")
        try:
            with open(SHAP_OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_shap_results, f, ensure_ascii=False, indent=4)
            print(f"  ... Real-time save successful.")
        except Exception as e:
            print(f"  [!! ERROR] Real-time save failed: {e}")

    print(f"\n=======================================================")
    print(f"--- All samples processed (Total Duration: {time.time() - total_start_time:.2f} seconds) ---")
    print(f"[INFO] Final results file located at: {SHAP_OUTPUT_JSON}")


if __name__ == "__main__":
    run_batch_shap_analysis()