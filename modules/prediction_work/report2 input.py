import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from openai import OpenAI

# --- Configuration ---

# -- API and Model Configuration --
# IMPORTANT: For security, it is recommended to use environment variables instead of hardcoding your API key.
API_KEY = "YOUR_API_KEY_HERE"  
# Replace the base URL with your actual API endpoint (e.g., local vLLM, Ollama, or custom server)
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "oss"  # Or any other model you need to use


# -- Concurrency and Task Range --
# Set the maximum number of concurrent worker threads (API calls)
MAX_WORKERS = 10
# Define the range of report files to process (based on the sorted file list).
# This is a Python slice, so the end index is exclusive.
# Process the first 10 reports: START_REPORT_INDEX = 500, END_REPORT_INDEX = 600
# Process all reports: START_REPORT_INDEX = 500, END_REPORT_INDEX = None
START_REPORT_INDEX = 0
END_REPORT_INDEX = 2500 # Set to an integer to limit the range, or set to None to process all files

# -- File Paths --
# NOTE: Please verify these paths and update them to point to your actual dataset files before running.
REPORT_1_DIR = "./data/report/"
REPORT_2_PATH = "./data/A_Comprehensive_Risk_Assessment_Framework.txt"
CLUSTER_SAMPLES_CSV = "./data/cluster_samples_glm.csv"
PARTICIPANT_DATA_JSON = "./data/tmpp.json"
OUTPUT_DIR = "./output/oss-reportone"  # Change output directory to avoid overwriting old results

# --- Main Prompt Template ---
# This is the final English prompt, now including a placeholder for previous reasoning logs.
PROMPT_TEMPLATE = """
You are a top-tier clinical AI reasoning expert, specializing in the intersection of nephrology and hepatology. Your core mission is to synthesize evidence from multiple independent sources, including two expert reports and a previous reasoning log, to generate an authoritative, de novo risk assessment for Chronic Kidney Disease (CKD) for a specific UKBB participant with fatty liver disease.

You will receive four distinct sources of information:
1.  **A methodological report on CKD risk for a specific population ("Report 1").**
2.  **The participant's individual health data.**

Your analysis must be a deep synthesis of all sources.

--- Evidence Base and Background ---

**Evidence Source One: Specific Population Risk Perspective (e.g., a report on CKD risk in a fatty liver disease population)**
{report_1_content}
* **Role and Value**: This report provides in-depth insights into CKD risk within a specific disease context (such as fatty liver disease). It may contain detailed analysis of risk phenotypes unique to this population, key biomarkers, and distinct risk interactions.



--- Core Task: Evidence Synthesis and Critical Thinking Guide ---

Your core task is to **reconcile, integrate, and elevate** all information sources. You cannot simply choose one report and ignore the other; both reports are of equal importance. You must follow these critical thinking guidelines:
**Construct a Comprehensive, Multidimensional Argument**:
    * Your final assessment should not be a simple average or a choice between viewpoints but must be a new, coherent, and synthesized conclusion. You need to clearly argue why certain pieces of evidence were prioritized in the final risk calculation. Use evidence from one source to **validate, supplement, or challenge** evidence from another, thereby constructing a logically rigorous and sound risk profile.
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


def load_data():
    """Load all necessary data files into memory."""
    print("[INFO] Loading data files...")
    try:
        with open(REPORT_2_PATH, 'r', encoding='utf-8') as f:
            report_2_content = f.read()
        cluster_df = pd.read_csv(CLUSTER_SAMPLES_CSV)
        cluster_map = cluster_df.groupby('cluster')['sample_id'].apply(list).to_dict()
        with open(PARTICIPANT_DATA_JSON, 'r', encoding='utf-8') as f:
            participant_data = json.load(f)
        print("[INFO] Data loading completed.")
        return report_2_content, cluster_map, participant_data
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}. Please check the paths in the script.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during data loading: {e}")
        exit(1)


def process_cluster(cluster_id, sample_ids, report_2_content, all_participant_data, all_previous_reasoning, api_client):
    """
    Process all samples of a single cluster via API calls and save the results.
    """
    start_time = time.time()
    print(f"[Cluster {cluster_id}] Starting to process {len(sample_ids)} samples.")

    cluster_results = {}
    report_1_path = os.path.join(REPORT_1_DIR, f"{cluster_id}.txt")

    try:
        with open(report_1_path, 'r', encoding='utf-8') as f:
            report_1_content = f.read()
    except FileNotFoundError:
        print(f"[Cluster {cluster_id}] ERROR: Report 1 file not found at {report_1_path}. Skipping this cluster.")
        return cluster_id, 0, len(sample_ids)

    successful_inferences = 0
    for i, sample_id in enumerate(sample_ids):
        sample_id_str = str(sample_id)
        print(f"[Cluster {cluster_id}] Processing sample {i + 1}/{len(sample_ids)} (ID: {sample_id_str})...")

        participant_info = all_participant_data.get(sample_id_str)
        if not participant_info:
            print(f"[Cluster {cluster_id}] WARNING: Participant data not found for sample ID {sample_id_str}. Skipping.")
            continue

        patient_data_json_str = json.dumps(participant_info)

        previous_reasoning = all_previous_reasoning.get(sample_id_str, {})
        evidence_summary = previous_reasoning.get("evidence_summary", "Not available.")
        reasoning_process_text = previous_reasoning.get("reasoning_process", "Not available.")
        previous_reasoning_log_str = (
            f"--- Previous Reasoning Log ---\n"
            f"Evidence Summary:\n{evidence_summary}\n\n"
            f"Reasoning Process:\n{reasoning_process_text}\n"
            f"--- End of Previous Reasoning Log ---"
        )

        final_prompt = PROMPT_TEMPLATE.format(
            report_1_content=report_1_content,
            report_2_content=report_2_content,
            previous_reasoning_log=previous_reasoning_log_str,
            patient_data_json=patient_data_json_str
        )

        try:
            # ** NEW: Use the NVIDIA client for API call **
            completion = api_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=16384 * 2,
                temperature=0.2,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=False  # CRITICAL: Set to False to get a single response object
            )

            # Extract the reasoning process and the final response content
            response_dict = completion.model_dump()
            reasoning = "No reasoning process found in the response."
            # Reasoning content might be nested in different places depending on the API version.
            # Attempt to find it in a common location here.
            if response_dict.get("choices") and response_dict["choices"][0].get("reasoning_content"):
                reasoning = response_dict["choices"][0]["reasoning_content"]

            content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "No content found.")

            cluster_results[sample_id_str] = {
                "reasoning": reasoning,
                "final_response": content
            }
            successful_inferences += 1

        except Exception as e:
            print(f"[Cluster {cluster_id}] Error for sample {sample_id_str}: {e}")
            cluster_results[sample_id_str] = {"error": str(e)}

    # Save results for the entire cluster
    output_path = os.path.join(OUTPUT_DIR, f"{cluster_id}.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_results, f, ensure_ascii=False, indent=4)
        duration = time.time() - start_time
        print(
            f"[Cluster {cluster_id}] Processing completed. Results for {len(cluster_results)} samples saved to {output_path}. "
            f"Time elapsed: {duration:.2f} seconds."
        )
    except Exception as e:
        print(f"[Cluster {cluster_id}] ERROR: Failed to save results to {output_path}. Error: {e}")

    return cluster_id, successful_inferences, len(sample_ids) - successful_inferences


def main():
    """Main function to coordinate concurrent processing."""
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] API_KEY is not set. Please edit the script and add your API key.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_2_content, cluster_map, participant_data = load_data()

    try:
        report_files = [f for f in os.listdir(REPORT_1_DIR) if f.endswith('.txt')]
        # Sort cluster IDs numerically rather than alphabetically (e.g., 2 before 10)
        all_cluster_ids = sorted([int(os.path.splitext(f)[0]) for f in report_files])
    except Exception as e:
        print(f"[ERROR] Error reading cluster reports from {REPORT_1_DIR}: {e}")
        return

    # ** NEW: Slice the cluster list to process a specific range **
    if END_REPORT_INDEX is None:
        # If there is no end index, process to the end of the list
        selected_cluster_ids = all_cluster_ids[START_REPORT_INDEX:]
    else:
        selected_cluster_ids = all_cluster_ids[START_REPORT_INDEX:END_REPORT_INDEX]

    print(f"[INFO] Found a total of {len(all_cluster_ids)} reports.")
    print(
        f"[INFO] Processing range from index {START_REPORT_INDEX} to {END_REPORT_INDEX if END_REPORT_INDEX is not None else len(all_cluster_ids)}, total {len(selected_cluster_ids)} reports."
    )

    # Initialize OpenAI client for API
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    # Initialize empty dictionary for previous reasoning data to prevent thread execution errors
    previous_reasoning_data = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_cluster,
                cluster_id,
                cluster_map.get(cluster_id, []),
                report_2_content,
                participant_data,
                previous_reasoning_data,
                client  # Pass a single, thread-safe client instance
            )
            for cluster_id in selected_cluster_ids if cluster_map.get(cluster_id)
        ]

        total_success = 0
        total_fail = 0
        for future in as_completed(futures):
            try:
                cluster_id, success, fail = future.result()
                total_success += success
                total_fail += fail
            except Exception as e:
                print(f"[ERROR] A worker thread generated an unexpected exception: {e}")

    print("\n--- All clusters have been processed ---")
    print(f"Total successful inferences: {total_success}")
    print(f"Total failed inferences: {total_fail}")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()