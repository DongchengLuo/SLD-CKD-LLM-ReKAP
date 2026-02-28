import pandas as pd
import json
import os
import pathlib
from glob import glob
import time

# Import the specified Google GenAI library
from google import genai

# from google.genai import types # Since we no longer process PDF bytes, this import can be omitted

# --- 1. Configuration Section ---

# API Key Configuration
# IMPORTANT: For security, it is recommended to use environment variables or a secure key management service instead of hardcoding.
API_KEY = 'YOUR_API_KEY_HERE'  # <--- Replace your API Key here

# Model Configuration
MODEL_NAME = 'gemini-2.5-pro-preview-06-05'  # Or your specified 'gemini-2.5-pro-preview-05-06'

# File and Directory Path Configuration
# TODO: Update these paths to point to your actual dataset files before running
# Input files
DEDUPLICATED_EXTRACTION_PATH = './data/deduplicated_extraction.json'
ENTITY_LIST_PATH = './data/entity_list.json'  # Updated to .json

# Output file
CONSOLIDATED_ENTITIES_OUTPUT_PATH = './output/consolidated_entities_summary.json'

# Prompt instructions for entity summary generation (as a template)
# **Note: This new version of the prompt is more general while retaining your required structured output**
# **Note: This is the revised version which will be used as part of the final prompt**
PROMPT_INSTRUCTIONS = '''
You are a biomedical knowledge synthesis expert. Your primary task is to create a comprehensive, consolidated summary for a single biomedical entity based on the information provided below. The summary must naturally reflect the primary context of the source evidence, which is **predicting the risk of Chronic Kidney Disease (CKD) in individuals with NAFLD, MAFLD, or MASLD**.

**Your instructions are as follows:**
1.  Analyze all the provided `evidence_snippets`.
2.  Synthesize this information into a single, comprehensive, and coherent summary.
3.  If the provided descriptions are contradictory, please try to resolve them or note the discrepancy in your summary.
4.  Ensure your descriptions are written in a neutral, third-person scientific tone.
5.  Generate a JSON output that strictly follows the `REQUIRED OUTPUT JSON SCHEMA` provided below.

--- REQUIRED OUTPUT JSON SCHEMA ---
{
  "synonyms": ["Based on the evidence, list any synonyms or different terminologies used for this entity, e.g., 'high blood pressure', 'HTN', 'hypertensive disease', etc."],
  "consolidated_description": "In a narrative paragraph, synthesize all provided information to create a comprehensive description of this entity. Focus on its definition, importance, and general role as presented in the evidence.",
  "key_attributes_and_roles_summary": [
      "Summarize its key roles in a list. e.g., 'Identified as a major independent risk factor for CKD incidence.'",
      "Summarize its common definitions or measurement criteria. e.g., 'Commonly defined as blood pressure >= 140/90 mmHg or use of antihypertensive medications.'",
      "Summarize its typical interaction with other factors if mentioned. e.g., 'Often co-exists with other metabolic risk factors like diabetes.'"
  ],
  "prevalence_in_nafld_ckd_context": "Summarize any prevalence information, capturing ranges if possible. If no information is provided, return null. e.g., 'Reported prevalence in NAFLD cohorts ranges from 40% to over 70%.'",
  "evidence_source_count": "The total number of unique papers provided in the input.",
  "paper_titles": ["List all unique paper titles from the input evidence snippets."]
}
'''


# --- Configuration End ---

def generate_summary_via_api(
        client: genai.Client,
        model_name: str,
        final_prompt: str  # Now receives only one final string containing all content
) -> dict | None:
    """
    Send a plain text prompt using your specified API call method and get the entity summary.
    """
    print("  [INFO] Calling genai API with text prompt...")
    try:
        # For plain text/JSON input, the content is the prompt itself
        contents_for_api = [final_prompt]

        response = client.models.generate_content(
            model=model_name,
            contents=contents_for_api
        )

        # Clean and parse the returned JSON
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        summary_json = json.loads(response_text)
        print("  [SUCCESS] Successfully received and parsed summary from API.")
        return summary_json

    except Exception as e:
        print(f"  [ERROR] An error occurred during API call or JSON parsing: {type(e).__name__} - {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(
                f"  [DEBUG] Raw response text that failed parsing (first 500 chars): \n---\n{response.text[:500]}...\n---")
        return None


def main_entity_summarizer():
    """
    Main function for batch processing entities and generating summaries.
    """
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("[ERROR] API_KEY is not configured. Please set your API_KEY at the top of the script.")
        return
    print("[INFO] Using in-script variable to configure API key.")

    print("[INFO] Initializing genai.Client...")
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize genai.Client: {e}")
        return

    # --- Data Loading ---
    print("\n[INFO] Loading data files...")
    try:
        with open(ENTITY_LIST_PATH, 'r', encoding='utf-8') as f:
            unique_entities = json.load(f)
        print(f"  - Loaded {len(unique_entities)} deduplicated entities from '{ENTITY_LIST_PATH}'")

        with open(DEDUPLICATED_EXTRACTION_PATH, 'r', encoding='utf-8') as f:
            deduplicated_data = json.load(f)
        print(f"  - Loaded extraction results from {len(deduplicated_data)} documents from '{DEDUPLICATED_EXTRACTION_PATH}'")

    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}")
        return
    except Exception as e:
        print(f"[ERROR] An error occurred while loading data: {e}")
        return

    # --- Main Process ---
    consolidated_entities = {}
    print(f"\n[INFO] Starting to generate summaries for {len(unique_entities)} entities...")
    print("-" * 50)

    for i, entity_name in enumerate(unique_entities):
        print(f"\n[INFO] Processing entity {i + 1}/{len(unique_entities)}: '{entity_name}'")

        evidence_snippets = []
        source_documents = {}

        # 1. Collect all evidence snippets related to the current entity
        for doc_extraction in deduplicated_data:
            doc_metadata = doc_extraction.get('document_metadata', {})
            paper_title = doc_metadata.get('paper_title')
            pubmed_id = doc_metadata.get('pubmed_id')

            # **Core logic fix: Strictly match entity name**
            for entity_obj in doc_extraction.get('extracted_entities', []):
                if entity_name == entity_obj.get('entity_name_as_in_text'):
                    evidence_snippets.append({
                        "source_paper_title": paper_title,
                        "source_pubmed_id": pubmed_id,
                        "contextual_description": entity_obj.get(
                            'general_description_from_text') or "No specific description available.",
                        "direct_quote": (entity_obj.get('overall_supporting_evidence_quotes_for_entity') or [
                            "No quote available."])[0]
                    })
                    if paper_title and paper_title not in source_documents:
                        source_documents[paper_title] = {'pubmed_id': pubmed_id}

            for rel_obj in doc_extraction.get('extracted_relationships', []):
                # Check if source entity and target entity match the current entity
                is_source = entity_name == rel_obj.get('source_entity_text')
                is_target = entity_name == rel_obj.get('target_entity_text')
                if is_source or is_target:
                    evidence_snippets.append({
                        "source_paper_title": paper_title,
                        "source_pubmed_id": pubmed_id,
                        "contextual_description": rel_obj.get('detailed_description_from_text_of_relationship'),
                        "direct_quote":
                            (rel_obj.get('supporting_evidence_quotes_for_relationship') or ["No quote available."])[0]
                    })
                    if paper_title and paper_title not in source_documents:
                        source_documents[paper_title] = {'pubmed_id': pubmed_id}

        if not evidence_snippets:
            print(f"  - WARNING: No context evidence found for '{entity_name}', skipped.")
            continue

        print(f"  - Found {len(evidence_snippets)} evidence snippets for '{entity_name}' from {len(source_documents)} documents.")

        # 2. Construct input data to submit to the API
        entity_type_for_prompt = "Unknown"  # Default value, try to get it from the first found entity object
        for doc in deduplicated_data:
            for entity_obj in doc.get('extracted_entities', []):
                if entity_name == entity_obj.get('entity_name_as_in_text'):
                    entity_type_for_prompt = entity_obj.get('entity_type', 'Unknown')
                    break
            if entity_type_for_prompt != "Unknown":
                break

        input_data_for_prompt = {
            "unique_entity_name": entity_name,
            "entity_type": entity_type_for_prompt,
            "evidence_snippets": evidence_snippets
        }

        # Combine instructions and data into the final prompt string
        input_data_string = json.dumps(input_data_for_prompt, indent=2, ensure_ascii=False)
        final_prompt_string = f"{PROMPT_INSTRUCTIONS}\n\n--- INPUT DATA ---\n{input_data_string}"

        # 3. Call the API
        summary_result = generate_summary_via_api(client, MODEL_NAME, final_prompt_string)

        # 4. Store the results
        if summary_result:
            # Pack the LLM generation result and the input evidence snippets into a list for storage
            consolidated_entities[entity_name] = [summary_result, evidence_snippets]
            print(f"  - Successfully generated summary for '{entity_name}'.")
        else:
            print(f"  - Failed to generate summary for '{entity_name}', skipping this entity.")

    # --- 5. Save consolidated summaries for all entities ---
    print("-" * 50)
    print("\n[INFO] Entity summarization process finished.")

    output_path_obj = pathlib.Path(CONSOLIDATED_ENTITIES_OUTPUT_PATH)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(consolidated_entities, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] All consolidated entity summaries have been saved to: '{output_path_obj}'")


if __name__ == '__main__':
    main_entity_summarizer()