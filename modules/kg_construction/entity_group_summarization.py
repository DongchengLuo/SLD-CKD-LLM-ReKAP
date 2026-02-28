import pandas as pd
import json
import os
import pathlib
import time

# Import the specified Google GenAI library
from google import genai

# --- 1. Configuration Section ---

# API Key Configuration
# IMPORTANT: For security, it is recommended to use environment variables or a secure key management service instead of hardcoding.
API_KEY = 'YOUR_API_KEY_HERE'  # <--- Replace your API Key here

# Model Configuration
MODEL_NAME = 'gemini-2.5-pro-preview-06-05'  # Ensure this model matches your API key permissions

# File and Directory Path Configuration
# TODO: Update these paths to point to your actual dataset files before running
DEDUPLICATED_EXTRACTION_PATH = './data/deduplicated_extraction.json'
ENTITY_GROUPS_CSV_PATH = './data/entity_groups.csv'

# Output File
CONSOLIDATED_GROUPS_OUTPUT_PATH = './output/consolidated_groups_summary.json'

# Prompt: Combines entity group definition and overall summarization task
PROMPT_TEMPLATE_GROUP = '''
You are an expert biomedical knowledge synthesis assistant.
Your primary and ONLY task is to create a comprehensive, consolidated summary for the following specific biomedical entity group, which is defined as a single conceptual unit below:
**{entity_group_definition_str}**

All of your analysis and summarization must be strictly centered on this entity group as a whole. Use the provided `evidence_snippets` to understand the collective role, definition, and attributes of this group (conceptually represented by **"{group_name}"**), especially in the context of predicting Chronic Kidney Disease (CKD) in individuals with NAFLD, MAFLD, or MASLD, which is the primary theme of the evidence. Your goal is to synthesize information about the individual entities into a coherent summary about the group's conceptual role.

**Your instructions are as follows:**
1.  Analyze all the provided `evidence_snippets` to find information specifically about ANY of the entities belonging to the **"{group_name}"** group.
2.  Synthesize this information into a single, comprehensive, and coherent summary that describes the entity group **"{group_name}"** as a conceptual unit.
3.  If the evidence is contradictory regarding the group's role, note the discrepancy in your summary.
4.  Generate a JSON output that strictly follows the `REQUIRED OUTPUT JSON SCHEMA` provided below. The content of the JSON must be about the entity group **"{group_name}"**.

--- INPUT DATA (for context) ---
{input_data_json}

--- REQUIRED OUTPUT JSON SCHEMA (for entity group: "{group_name}") as following---
'''

PROMPT_OUTPUT_INSTRUCT = '''
{
  "consolidated_description": "In a narrative paragraph, synthesize all provided information to create a comprehensive description of this entity GROUP. This description must start with a general definition of the group's theme and then focus on its importance and collective role as presented in the evidence.",
  "key_attributes_and_roles_summary": [
      "Summarize the group's key roles in a list. e.g., 'Represents a cluster of major independent risk factors for CKD incidence.'",
      "Summarize common definitions or measurement criteria for entities within this group. e.g., 'This group includes factors defined by metrics like blood pressure >= 140/90 mmHg or specific lab value thresholds.'",
      "Summarize typical interactions among factors within the group or with other factors, if mentioned. e.g., 'These factors often co-exist and have synergistic effects on disease progression.'"
  ],
  "prevalence_in_nafld_ckd_context": "Summarize any prevalence information about the entities in this group, capturing ranges if possible. If no relevant information is found, return null. e.g., 'Entities in this group are highly prevalent in NAFLD cohorts, with individual factor prevalence ranging from 30% to over 70.'",
  "evidence_source_count": "The total number of unique papers that provided the evidence for this group.",
  "paper_titles": ["List all unique paper titles from the input evidence snippets for this group."]
}
'''


# --- Configuration End ---

def generate_summary_via_api(
        client: genai.Client,
        model_name: str,
        final_prompt: str
) -> dict | None:
    """
    Send a plain text prompt using your specified API call method and get the entity summary.
    """
    print(f"  [INFO] Calling genai API for group summary...")
    try:
        # Put the prompt into a list to match the calling format of the original script
        contents_for_api = [final_prompt]

        # Use the client.models.generate_content method
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


def main_group_summarizer():
    """
    Main function for batch processing entity groups and generating summaries.
    """
    if 'YOUR_API_KEY_HERE' in API_KEY or not API_KEY:
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
        entity_groups_df = pd.read_csv(ENTITY_GROUPS_CSV_PATH, encoding='utf-8')
        print(f"  - Loaded {len(entity_groups_df)} entity mapping relationships from '{ENTITY_GROUPS_CSV_PATH}'")

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
    consolidated_summaries = {}
    grouping_columns = ['Semantic Group', 'Functional Group']

    for group_col in grouping_columns:
        print("\n" + "=" * 60)
        print(f"[INFO] Processing groups based on column: '{group_col}'")
        print("=" * 60)

        valid_groups = entity_groups_df.dropna(subset=[group_col])
        groups = valid_groups.groupby(group_col)

        total_groups = len(groups)
        group_counter = 0

        for group_name, group_df in groups:
            group_counter += 1
            if group_name == "None":
                print(f"\n[INFO] Skipping group '{group_name}' as per instructions.")
                continue

            entities_in_group = group_df['Original Entity'].tolist()
            print(
                f"\n[INFO] Processing group {group_counter}/{total_groups} ('{group_name}') with {len(entities_in_group)} entities...")

            # 1. Collect and deduplicate evidence snippets for the entire group
            group_evidence_snippets = []
            found_paper_titles = set()

            for entity_name in entities_in_group:
                for doc_extraction in deduplicated_data:
                    paper_title = doc_extraction.get('document_metadata', {}).get('paper_title')
                    if paper_title in found_paper_titles:
                        continue

                    found_in_doc = False
                    for entity_obj in doc_extraction.get('extracted_entities', []):
                        if entity_name == entity_obj.get('entity_name_as_in_text'):
                            found_in_doc = True
                            break

                    if not found_in_doc:
                        for rel_obj in doc_extraction.get('extracted_relationships', []):
                            if entity_name == rel_obj.get('source_entity_text') or entity_name == rel_obj.get(
                                    'target_entity_text'):
                                found_in_doc = True
                                break

                    if found_in_doc:
                        group_evidence_snippets.append(doc_extraction)
                        if paper_title:
                            found_paper_titles.add(paper_title)

            if not group_evidence_snippets:
                print(f"  - WARNING: No context evidence found for the entire group '{group_name}'. Skipping.")
                continue

            print(
                f"  - Found {len(group_evidence_snippets)} evidence snippets for group '{group_name}' from {len(found_paper_titles)} unique papers.")

            # 2. Construct input data to submit to the API
            input_data_for_prompt = {
                "entity_group": {
                    "group_name": group_name,
                    "group_type": group_col,
                    "entities": entities_in_group
                },
                "evidence_snippets": group_evidence_snippets
            }

            # 3. Construct the final prompt string
            input_data_string = json.dumps(input_data_for_prompt, indent=2, ensure_ascii=False)
            entity_group_definition_str = f'"{group_name}": {json.dumps(entities_in_group, ensure_ascii=False)}'

            final_prompt_string = PROMPT_TEMPLATE_GROUP.format(
                entity_group_definition_str=entity_group_definition_str,
                group_name=group_name,
                input_data_json=input_data_string
            ) + '\n' + PROMPT_OUTPUT_INSTRUCT

            # 4. Call the API
            summary_result = generate_summary_via_api(client, MODEL_NAME, final_prompt_string)

            # 5. Store the result
            if summary_result:
                summary_key = f"{group_col}: {group_name}"
                consolidated_summaries[summary_key] = summary_result
                print(f"  - Successfully generated summary for group '{summary_key}'.")
            else:
                print(f"  - Failed to generate summary for group '{group_name}'. Skipping.")

    # --- 6. Save consolidated summaries for all entity groups ---
    print("-" * 60)
    print("\n[INFO] Group summarization process finished.")

    output_path_obj = pathlib.Path(CONSOLIDATED_GROUPS_OUTPUT_PATH)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(consolidated_summaries, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] All consolidated group summaries have been saved to: '{output_path_obj}'")


if __name__ == '__main__':
    main_group_summarizer()