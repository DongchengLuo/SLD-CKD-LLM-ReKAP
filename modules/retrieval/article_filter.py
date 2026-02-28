# -*- coding: utf-8 -*-
import os
import json
import traceback
import time
import threading
import requests
import re
import itertools
from queue import Queue

# ==============================================================================
# --- 1. Global Configuration Area ---
# ==============================================================================

# --- Local Model Configuration ---
# [IMPORTANT] URL for your local inference server (e.g., vLLM, llama.cpp, or Ollama)
LOCAL_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# [IMPORTANT] Name of your local open-source model
MODEL_NAME = "glm4.5air" 
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 8192  # Adjust based on your local model's context window

# --- Concurrency Configuration ---
# Number of concurrent requests sent to the local server.
MAX_CONCURRENT_REQUESTS = 4

# --- Input File Paths ---
# TODO: Update these paths to point to your actual dataset files before running
PATH_PATIENT_SAMPLES = "./data/tmp.json"
PATH_ABSTRACT_SAMPLES = "./data/abstract435.json"

# --- Output Directory Path ---
# Directory where the prediction results will be saved
OUTPUT_DIR = "./output/article_retrieval_results"


# ==============================================================================
# --- 2. Helper Functions ---
# ==============================================================================

def save_individual_result(output_dir, thread_id, item_key, result_data, save_lock):
    """
    Saves the result of a single sample immediately to a file associated with the thread.
    Uses a lock to ensure thread-safe writes.
    """
    output_filename = f"final_prediction_results_thread_{thread_id}.json"
    output_file_path = os.path.join(output_dir, output_filename)

    with save_lock:
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_results = {}

        all_results[item_key] = result_data

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                sorted_results = dict(sorted(all_results.items()))
                json.dump(sorted_results, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"  [Error] Failed to write to result file {output_file_path}: {e}")


# ==============================================================================
# --- 3. Core Worker Thread ---
# ==============================================================================

def inference_worker(
        task_queue,
        thread_id,
        output_dir,
        abstracts_half1_json,
        abstracts_half2_json,
        save_lock
):
    """
    Core worker thread for local inference:
    - Fetches a sample from the global queue.
    - Performs two local LLM calls (one for each half of the literature data).
    - Asks the LLM to return the top 25 most relevant articles per call.
    - Merges the two results, sorts them, and extracts the top 50 overall.
    - Saves the merged results immediately.
    """
    thread_identifier = f"Thread-{thread_id}"

    prompt_template = """
You are a highly precise medical literature analyst AI. Your task is to rank a list of scientific articles based on their relevance to a specific patient's clinical data and the article's journal impact factor (IF). After a comprehensive evaluation, you will select and output **only the top 25 strongest articles** from the list you are given.

You will be given two pieces of information:

1.  **Patient Information**: A JSON object containing the clinical data of a patient.
2.  **Article Data**: A JSON object containing the articles for evaluation.

---
**How to Understand the 'Article Data' Structure:**

The 'Article Data' is a JSON object that functions as a dictionary. **The DOI number of each article is the TOP-LEVEL KEY for each entry.** The value associated with that key is another object containing the 'paper_title', 'abstract', 'journal', and 'IF'. You must use this top-level key (the DOI) for your output.

Here is a small example of the data structure:
```json
{{
    "10.1016/j.jhep.2017.08.024.": {{
        "paper_title": "A fatty liver leads to decreased kidney function?",
        "abstract": "BACKGROUND & AIMS: Non-alcoholic fatty liver disease (NAFLD) has been associated...",
        "journal": "J Hepatol",
        "IF": 26.8
    }},
    "10.24546/0100489395.": {{
        "paper_title": "A High Fibrosis-4 Index is Associated with a Reduction in...",
        "abstract": "Liver fibrosis is associated with non-alcoholic fatty liver disease (NAFLD), and...",
        "journal": "Kobe J Med Sci",
        "IF": 1.1
    }}
}}
```
In this example, "10.1016/j.jhep.2017.08.024." is a DOI. You will extract this exact string for your output.

---
**Your Goal and Ranking Rules:**

1.  **Goal:** Internally evaluate ALL articles in the provided **Article Data**. Based on this evaluation, create a ranked list of **only the top 25 strongest articles**.
2.  **Scoring (0-100):** Generate a single numerical score for each article.
3.  **Contribution (50/50):** The score is determined equally by:
    * **Relevance:** How related the 'abstract' is to the patient's data.
    * **Impact Factor:** The 'IF' value.
4.  **Sorting:** The final output must be sorted by score in descending order (highest to lowest).

---
**Required Output Format and CRITICAL REQUIREMENTS:**

1.  **Format:** The output MUST be a single JSON object containing a dictionary.
2.  **Content:** This dictionary must contain **ONLY the top 25 articles** with the highest scores.
3.  **Structure:** The structure must be `"DOI number": score`.

**CRITICAL REQUIREMENTS:**
* The keys of your output dictionary MUST be an **EXACT, character-for-character match** to the top-level keys (the DOIs) from the input **Article Data**.
* **You are strictly forbidden from including anything else in the output.** DO NOT output the paper title, abstract, journal name, or the IF value. The final JSON object must contain ONLY the DOI numbers and their corresponding numerical scores.

---
Now, analyze the following new data:

**Patient Information:**
{patient_data_json}

**Article Data:**
{abstracts_data_json}

Begin your analysis. Use the top-level keys from the **Article Data** as the DOI numbers. Evaluate all articles, select the **top 25**, and return ONLY the final JSON object for those 25 articles, formatted and sorted exactly as specified.
    """

    def call_llm_for_chunk(patient_data_json_str, abstracts_chunk_json):
        """Helper function to perform a single local LLM call and parse the result."""
        final_prompt = prompt_template.format(
            patient_data_json=patient_data_json_str,
            abstracts_data_json=abstracts_chunk_json
        )
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": MODEL_TEMPERATURE,
            "max_tokens": MODEL_MAX_TOKENS,
        }

        # Basic retry logic for local server connection drops or heavy load
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(LOCAL_API_URL, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"].strip()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Local API request failed after {max_retries} attempts. Error: {e}")
                time.sleep(2 ** attempt)

        # --- Robust JSON Parsing ---
        json_str_to_parse = None
        matches = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if matches:
            json_str_to_parse = matches[-1]
        else:
            last_brace_close = content.rfind('}')
            if last_brace_close != -1:
                last_brace_open = content.rfind('{', 0, last_brace_close)
                if last_brace_open != -1:
                    json_str_to_parse = content[last_brace_open: last_brace_close + 1]

        if not json_str_to_parse:
            raise ValueError(f"No valid JSON object found in LLM output. Raw content: {content}")

        parsed_json = json.loads(json_str_to_parse)
        if not isinstance(parsed_json, dict):
            raise TypeError("LLM output is not a dictionary as expected.")
        return parsed_json

    while not task_queue.empty():
        try:
            sample_key, patient_data_dict = task_queue.get_nowait()
        except Exception:
            break

        print(f"[{thread_identifier}] Started processing sample '{sample_key}'...")

        try:
            patient_data_json_str = json.dumps(patient_data_dict, ensure_ascii=False, indent=2)

            # First local call
            print(f"    -> [{thread_identifier}] Processing part 1 for '{sample_key}'...")
            result1 = call_llm_for_chunk(patient_data_json_str, abstracts_half1_json)

            # Second local call
            print(f"    -> [{thread_identifier}] Processing part 2 for '{sample_key}'...")
            result2 = call_llm_for_chunk(patient_data_json_str, abstracts_half2_json)

            # Merge, sort, and select top 50
            combined_results = {**result1, **result2}
            sorted_items = sorted(combined_results.items(), key=lambda item: float(item[1]), reverse=True)
            top_50_results = dict(sorted_items[:50])

            print(f"  └──> [{thread_identifier}] Sample '{sample_key}' processed successfully!")
            save_individual_result(output_dir, thread_id, sample_key, top_50_results, save_lock)

        except Exception as e:
            print(f"  └──> [{thread_identifier}] Failed to process sample '{sample_key}'! Error: {e}")
            traceback.print_exc()
            
            # Save error state
            final_error_result = {"error": f"An unexpected error occurred: {str(e)}"}
            save_individual_result(output_dir, thread_id, sample_key, final_error_result, save_lock)
            
            # Put the task back into the queue for a retry
            task_queue.put((sample_key, patient_data_dict))
            time.sleep(2)

        finally:
            task_queue.task_done()

    print(f"[{thread_identifier}] Task queue is empty. Thread exiting.")


# ==============================================================================
# --- 4. Main Execution Logic ---
# ==============================================================================

def main():
    start_time = time.time()
    print("======================================================")
    print("=== Local LLM Literature Ranking Batch Inference ===")
    print("======================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load and Split Literature Data ---
    try:
        with open(PATH_ABSTRACT_SAMPLES, 'r', encoding='utf-8') as f:
            abstracts_data = json.load(f)

        # Split the abstract dictionary into two halves
        abstract_items = list(abstracts_data.items())
        midpoint = len(abstract_items) // 2
        abstracts_half1 = dict(abstract_items[:midpoint])
        abstracts_half2 = dict(abstract_items[midpoint:])

        # Convert split data to JSON strings for thread passing
        abstracts_half1_json = json.dumps(abstracts_half1, ensure_ascii=False)
        abstracts_half2_json = json.dumps(abstracts_half2, ensure_ascii=False)

        print(f"--- Successfully loaded and split abstract data: {PATH_ABSTRACT_SAMPLES} ---")
        print(f"    - Part 1 contains {len(abstracts_half1)} articles.")
        print(f"    - Part 2 contains {len(abstracts_half2)} articles.")
    except Exception as e:
        print(f"[Fatal Error] Could not load or split abstract file {PATH_ABSTRACT_SAMPLES}: {e}. Exiting.")
        return

    # --- Load Patient Data ---
    task_queue = Queue()
    try:
        print("\n--- Building global task pool... ---")
        with open(PATH_PATIENT_SAMPLES, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
            
            # Retaining the specific slice/testing logic from the original script
            all_samples = [dict(list(all_samples.items())[i: i + (len(all_samples) + 4) // 5]) for i in
                           range(0, len(all_samples), (len(all_samples) + 4) // 5)][0]
            all_samples = {k: v for k, v in itertools.islice(all_samples.items(), 0, 5000)}

        for sample_key, patient_data_dict in all_samples.items():
            if not isinstance(patient_data_dict, dict):
                print(f"[Warning] Data for sample {sample_key} is not a valid dictionary, skipping.")
                continue
            task_queue.put((sample_key, patient_data_dict))
            
        print(f"--- Task pool built successfully. Total valid tasks: {task_queue.qsize()} ---")
    except Exception as e:
        print(f"[Fatal Error] Error loading patient data: {e}. Exiting.")
        return

    # --- Start Worker Threads ---
    print("\n==================================================")
    print(f"=== ▶️ Starting {MAX_CONCURRENT_REQUESTS} concurrent local inference threads ===")
    print("==================================================")

    save_lock = threading.Lock()
    threads = []
    
    for i in range(MAX_CONCURRENT_REQUESTS):
        thread = threading.Thread(
            target=inference_worker,
            args=(
                task_queue, 
                i + 1, 
                OUTPUT_DIR,
                abstracts_half1_json,  
                abstracts_half2_json, 
                save_lock
            )
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\n--- All local inference threads have finished executing ---")
    
    if not task_queue.empty():
        print(f"\n[Warning] Queue is not empty. {task_queue.qsize()} tasks were left unprocessed due to errors.")

    end_time = time.time()
    print("\n==================================================")
    print(f"✅ All processing completed. Total time taken: {end_time - start_time:.2f} seconds.")
    print(f"💾 Prediction results saved to directory: {OUTPUT_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()