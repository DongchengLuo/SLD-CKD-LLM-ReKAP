# -*- coding: utf-8 -*-
import os
import json
import time
import re
import threading
import requests
from queue import Queue

# --- 1. Global Configuration Area ---

# --- Local Model Configuration ---
# [IMPORTANT] URL for your local inference server (e.g., vLLM, llama.cpp, or Ollama)
LOCAL_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# [IMPORTANT] Name of your local open-source model
MODEL_NAME = "glm4.5air" 
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 8192  # Adjust based on your local model's context window capabilities

# --- Concurrency Configuration ---
# Number of concurrent requests sent to the local server.
# Adjust this based on your GPU VRAM and local server batching settings (e.g., vLLM max_num_seqs).
MAX_CONCURRENT_REQUESTS = 4  

# --- Selection Configuration ---
MAX_ENTITIES_TO_SELECT = 40  # Final number of entities to select (must be an even number)

# --- Input File Paths (Modify according to your actual paths) ---
# TODO: Update these paths to point to your actual dataset files before running
PATH_PATIENT_SAMPLES = "./data/main_instance0(threshold40).json"
PATH_ENTITIES_JSON = "./data/entities_summary_simp.json"

# --- Output Directory Path ---
# Directory where the prediction results will be saved
OUTPUT_DIR = "./output/retrieval_results"


# --- 2. Helper Functions ---

def save_individual_result(output_dir, thread_id, item_key, result_data, save_lock):
    """
    Saves the result of a single sample immediately to a file associated with the thread.
    Uses a lock to ensure thread-safe writes if multiple threads access the same file structure.
    """
    output_filename = f"retrieval_results_thread_{thread_id}.json"
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


# --- 3. Core Worker Thread ---

def inference_worker(
        task_queue,
        thread_id,
        output_dir,
        all_entities_data,
        save_lock
):
    """
    Core worker thread for local inference:
    - Fetches a sample from the global queue.
    - Splits the entity list in half and calls the local LLM for each half.
    - Merges the two results and saves them immediately.
    """
    thread_identifier = f"Thread-{thread_id}"

    # Split all entities into two exact halves
    all_entity_keys = sorted(list(all_entities_data.keys()))
    mid_point = len(all_entity_keys) // 2
    keys_part1 = all_entity_keys[:mid_point]
    keys_part2 = all_entity_keys[mid_point:]
    entities_part1 = {k: all_entities_data[k] for k in keys_part1}
    entities_part2 = {k: all_entities_data[k] for k in keys_part2}

    def _perform_llm_selection(patient_data_dict, entities_for_prompt, num_to_select):
        """Helper function to build prompt, call local API via requests, and parse the JSON result."""
        patient_data_str = json.dumps(patient_data_dict, ensure_ascii=False, indent=2)
        entities_prompt_str = json.dumps(entities_for_prompt, ensure_ascii=False, indent=2)

        prompt = f"""You are an expert in biomedical knowledge graphs and clinical data analysis. Your task is to select a curated list of exactly {num_to_select} entities from the "Candidate Entity List" based on the provided "Patient Profile".

        Your selection process must intelligently balance the following three core principles:

        1.  **High Relevance:** Prioritize entities that are most relevant to the "Patient Profile". Although you will not output the scores, this relevance assessment is the primary basis for your selection.
        2.  **Comprehensive Coverage:** A key objective is to maximize the coverage of different `Semantic Group` and `Functional Group` categories in your final list. For instance, if entities A and B are both highly relevant and in the same group, but entity C is slightly less relevant but belongs to a different group, you should prefer selecting A and C over A and B. This ensures the final selection has a broader medical and biological scope.
        3.  **Evidence Value:** Consider the `'evidence_source_count'`. A higher count indicates the entity is mentioned more frequently in literature, which adds to its value and should be an important factor in your selection.

        **How to Identify the Entity Name:**
        The `Candidate Entity List` is a JSON object where each **top-level key** IS the official entity name you must return. The value associated with each key contains descriptive details but is not the name itself.

        For example, given this structure:
        ```json
        {{
            "Hyperglycaemia": {{
                "consolidated_description": "Hyperglycemia, defined as fasting plasma glucose...",
                "evidence_source_count": 3,
                "Semantic Group": "Metabolic Dysfunction States",
                "Functional Group": "Metabolic Risk Factors"
            }},
            "Genetic polymorphisms (PNPLA3, HSD17B13, TM6SF2, MBOAT7, GCKR)": {{ ... }}
        }}
        ```

        The exact, original entity name to use for the first entry is the key itself: "Hyperglycaemia". For the second, it is "Genetic polymorphisms (PNPLA3, HSD17B13, TM6SF2, MBOAT7, GCKR)".

        Crucial Instruction: The entity names in your output list MUST EXACTLY MATCH these top-level keys from the 'Candidate Entity List'. Do not alter, abbreviate, rephrase, or derive names from the descriptions. Every character, including parentheses, commas, and casing, must be identical.

        Your output MUST be a single, valid JSON object containing only the key "selected_entities". Do not include any text, notes, or explanations outside of the JSON object.

        {{
        "selected_entities": [
        "Entity Name A",
        "Entity Name C",
        ...
        ]
        }}

        The selected_entities list must contain EXACTLY {num_to_select} entity names.

        Patient Profile:

        {patient_data_str}

        Candidate Entity List (in JSON format):

        {entities_prompt_str}
        """

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": MODEL_TEMPERATURE,
            "max_tokens": MODEL_MAX_TOKENS,
        }

        # Add basic retry logic for local server connection drops or temporary overloads
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
                time.sleep(2 ** attempt)  # Exponential backoff

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

        if json_str_to_parse:
            try:
                parsed_json = json.loads(json_str_to_parse)
                if "selected_entities" in parsed_json and isinstance(parsed_json["selected_entities"], list):
                    return parsed_json["selected_entities"]
                else:
                    raise ValueError(f"LLM output JSON is missing the 'selected_entities' key or it is not a list. Raw content: {content}")
            except json.JSONDecodeError:
                raise ValueError(f"LLM output could not be parsed as JSON. Raw content: {content}")
        else:
            raise ValueError(f"No valid JSON object found in LLM output. Raw content: {content}")

    # Process tasks from queue
    while not task_queue.empty():
        try:
            sample_key, patient_data_dict = task_queue.get_nowait()
        except Exception:
            break

        print(f"[{thread_identifier}] Started processing sample '{sample_key}'...")

        try:
            # --- First Selection (Part 1) ---
            selected_part1 = _perform_llm_selection(patient_data_dict, entities_part1, MAX_ENTITIES_TO_SELECT // 2)

            # --- Second Selection (Part 2) ---
            selected_part2 = _perform_llm_selection(patient_data_dict, entities_part2, MAX_ENTITIES_TO_SELECT // 2)

            # --- Merge and Save Results ---
            combined_list = selected_part1 + selected_part2
            final_result_for_saving = {"selected_entities": combined_list}

            # Check final quantity and add a warning if necessary
            final_count = len(combined_list)
            if final_count != MAX_ENTITIES_TO_SELECT:
                print(f"  [Warning] Local LLM returned {final_count} entities, instead of the requested {MAX_ENTITIES_TO_SELECT}.")
                final_result_for_saving["warning"] = f"Expected {MAX_ENTITIES_TO_SELECT} entities, but got {final_count}"

            print(f"  └──> [{thread_identifier}] Sample '{sample_key}' processed successfully!")
            save_individual_result(output_dir, thread_id, sample_key, final_result_for_saving, save_lock)

        except Exception as e:
            print(f"  └──> [{thread_identifier}] Failed to process sample '{sample_key}'! Error: {e}")
            # Put the task back into the queue so another thread can try it later
            task_queue.put((sample_key, patient_data_dict))  
            time.sleep(2) # Brief pause before picking up next task

        finally:
            task_queue.task_done()

    print(f"[{thread_identifier}] Task queue is empty. Thread exiting.")


# --- 4. Main Execution Logic ---

def main():
    start_time = time.time()
    print("==========================================================")
    print("=== Knowledge Graph Local LLM Batch Inference System ===")
    print("==========================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Data Once ---
    try:
        with open(PATH_ENTITIES_JSON, 'r', encoding='utf-8') as f:
            all_entities = json.load(f)
        print(f"Successfully loaded entities file, found {len(all_entities)} entities in total.")

        with open(PATH_PATIENT_SAMPLES, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        print(f"Successfully loaded patient samples file, found {len(all_samples)} samples in total.")
    except FileNotFoundError as e:
        print(f"[Fatal Error] Input file not found: {e}. Exiting application.")
        return
    except json.JSONDecodeError as e:
        print(f"[Fatal Error] Invalid JSON format in input file: {e}. Exiting application.")
        return

    # --- Process All Samples ---
    selected_keys = list(all_samples.keys())
    print(f"Defaulting to process all {len(selected_keys)} samples.")

    # --- Populate Task Queue ---
    task_queue = Queue()
    print("\n--- Building global task pool... ---")
    for sample_key in selected_keys:
        task_queue.put((sample_key, all_samples[sample_key]))
    print(f"--- Task pool built successfully. Total tasks: {task_queue.qsize()} ---")

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
                all_entities,
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