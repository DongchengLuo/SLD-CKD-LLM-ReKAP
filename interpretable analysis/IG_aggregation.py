import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# === 1. Configuration Paths ===
# TODO: Update these paths to point to your actual dataset and output directories before running
INPUT_DIR = "./data/IG_up"
ATTR_FILE = "./data/field_attributes.xlsx"  # Attribute mapping file if needed
OUTPUT_REPORT = "./output/Final_Binned_Analysis_Report.csv"


# === 2. Helper Functions ===
def decode_token(token_str):
    """Safely decode token strings containing unicode escape sequences."""
    try:
        return bytes(token_str, "utf-8").decode("unicode_escape")
    except:
        return token_str


def extract_json_block_smart(full_text):
    """Smartly extract the JSON block based on specific anchor text."""
    anchor = "Participant Information"
    anchor_idx = full_text.find(anchor)
    if anchor_idx == -1: return None, None, None
    
    start_brace = full_text.find('{', anchor_idx)
    if start_brace == -1: return None, None, None
    
    brace_count = 0
    end_brace = -1
    for i in range(start_brace, len(full_text)):
        char = full_text[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        
        if brace_count == 0:
            end_brace = i
            break
            
    if end_brace == -1: return None, None, None
    return full_text[start_brace: end_brace + 1], start_brace, end_brace + 1


# === 3. Data Extraction ===
def get_extracted_data():
    """Scan and parse JSON files to extract features, values, and IG scores."""
    print("[INFO] Scanning JSON files...")
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    all_data = []

    for f_path in tqdm(json_files, desc="Extraction"):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        items = []
        if "prompt_analysis" in data:
            items.append(("Single", data))
        else:
            for k, v in data.items():
                if isinstance(v, dict) and "prompt_analysis" in v:
                    items.append((k, v))

        for sid, content in items:
            prompt_analysis = content["prompt_analysis"]
            full_text = ""
            char_map = []
            
            # Reconstruct the full text and build a character-to-token index map
            for i, item in enumerate(prompt_analysis):
                t = decode_token(item.get('token', ''))
                full_text += t
                char_map.extend([i] * len(t))

            json_str, r_start, r_end = extract_json_block_smart(full_text)
            if not json_str: continue
            
            try:
                patient_data = json.loads(json_str)
            except:
                continue

            cursor = r_start
            for key, val in patient_data.items():
                k_str = str(key)
                v_str = "null" if val is None else str(val).lower() if isinstance(val, bool) else str(val)

                # Locate the key in the text
                k_start = full_text.find(f'"{k_str}"', cursor, r_end)
                if k_start == -1: k_start = full_text.find(k_str, cursor, r_end)
                if k_start == -1: continue

                # Locate the value in the text
                v_start = full_text.find(v_str, k_start, r_end)
                if v_start == -1:
                    seg_end = k_start + len(k_str)
                else:
                    seg_end = v_start + len(v_str)
                    cursor = seg_end

                # Map character indices back to token indices
                indices = set(char_map[k_start:seg_end])
                scores = [prompt_analysis[idx]['score'] for idx in indices if idx < len(prompt_analysis)]

                if scores:
                    all_data.append({
                        "Feature": k_str,
                        "Value_Raw": val,
                        "IG_Sum": sum(scores),
                        "IG_Abs_Sum": sum([abs(s) for s in scores])
                    })
                    
    return pd.DataFrame(all_data)


# === 4. Core Analysis Logic (V8.0 Binning Version) ===
def analyze_binned(df):
    """Analyze the extracted data using a binning strategy for continuous variables."""
    report_rows = []

    # Preprocess ICD and Date features
    df['Analyzed_Feature_Name'] = df['Feature'].apply(
        lambda x: "Merged_ICD_Codes" if ('icd' in x.lower() and 'date' not in x.lower()) else x
    )
    df['Analyzed_Feature_Name'] = df.apply(
        lambda row: f"{row['Feature']} (Time)" if 'date' in row['Feature'].lower() else row['Analyzed_Feature_Name'],
        axis=1
    )

    unique_features = df['Analyzed_Feature_Name'].unique()
    print(f"[INFO] Analyzing {len(unique_features)} features with Binning Strategy...")

    for feat in unique_features:
        subset = df[df['Analyzed_Feature_Name'] == feat].copy()
        if len(subset) < 10: continue

        # 1. Calculate global importance (used for sorting later)
        global_importance = subset['IG_Abs_Sum'].median()

        # 2. Determine variable type and process accordingly
        is_continuous = False
        try:
            if 'date' in feat.lower():
                subset['numeric_val'] = pd.to_datetime(subset['Value_Raw'], errors='coerce').apply(
                    lambda x: x.toordinal() if pd.notnull(x) else np.nan)
            else:
                subset['numeric_val'] = pd.to_numeric(subset['Value_Raw'], errors='coerce')

            # Condition for continuous variable: numeric, mostly non-null, and has many unique values
            if subset['numeric_val'].notna().sum() > len(subset) * 0.7 and subset['numeric_val'].nunique() > 5 and feat != "Merged_ICD_Codes":
                is_continuous = True
        except:
            is_continuous = False

        # === Core Strategy: Convert both continuous and categorical variables into "groups" for output ===
        grouped_data = []  # To store tuples of: (Group_Label, Median_IG, Count)

        if is_continuous:
            # --- Continuous Variables: Binning ---
            valid_subset = subset.dropna(subset=['numeric_val', 'IG_Sum'])

            if len(valid_subset) >= 15:  # Only perform binning if there are enough samples
                try:
                    # Attempt to split into 3 equal quantiles (Low, Mid, High)
                    # qcut splits by quantiles, ensuring roughly equal sample sizes per group
                    valid_subset['bin'] = pd.qcut(valid_subset['numeric_val'], q=3, labels=["Low", "Mid", "High"], duplicates='drop')

                    # Calculate median gradient for each group
                    for bin_label in ["Low", "Mid", "High"]:
                        bin_data = valid_subset[valid_subset['bin'] == bin_label]
                        if not bin_data.empty:
                            # Also calculate the numeric range of this group for readability
                            min_v = bin_data['numeric_val'].min()
                            max_v = bin_data['numeric_val'].max()

                            # Try to restore readability of Date variables
                            if 'date' in feat.lower():
                                range_str = f"{bin_label}"  # Converting dates back to strings is tedious, use Label for now
                            else:
                                range_str = f"{bin_label} ({min_v:.1f}-{max_v:.1f})"

                            grouped_data.append({
                                "Label": range_str,
                                "IG": bin_data['IG_Sum'].median(),
                                "Count": len(bin_data)
                            })
                except:
                    # If binning fails (e.g., all values are identical), fallback to overall group
                    grouped_data.append({
                        "Label": "All Values",
                        "IG": valid_subset['IG_Sum'].median(),
                        "Count": len(valid_subset)
                    })
        else:
            # --- Categorical Variables: Group by Value ---
            subset['str_val'] = subset['Value_Raw'].astype(str)
            subset = subset[~subset['str_val'].isin(['nan', 'None', 'null', ''])]
            groups = subset.groupby('str_val')

            for label, grp in groups:
                if len(grp) < 3: continue
                grouped_data.append({
                    "Label": label,
                    "IG": grp['IG_Sum'].median(),
                    "Count": len(grp)
                })

        # === 3. Generate Report Rows ===
        # Generate a row for each group under this feature
        for item in grouped_data:
            median_ig = item['IG']

            # Determine direction logic
            if median_ig > 0.001:
                dir_logic = "Risk (↑)"
            elif median_ig < -0.001:
                dir_logic = "Protective (↓)"
            else:
                dir_logic = "Neutral"

            report_rows.append({
                "Feature": feat,
                "Sub_Group": item['Label'],
                "Global_Importance": global_importance,  # Convenient for sorting
                "Group_Direction_Score": median_ig,      # Direction metric for the group
                "Direction_Logic": dir_logic,
                "Sample_Count": item['Count']
            })

    return pd.DataFrame(report_rows)


# === 5. Main Program ===
def main():
    # Ensure output directory exists before writing to it
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

    df_raw = get_extracted_data()
    if df_raw.empty:
        print("[INFO] No data extracted. Exiting.")
        return

    print("[INFO] Running Binned Analysis...")
    df_report = analyze_binned(df_raw)

    if not df_report.empty:
        # Sort: by global importance (desc), then feature name, then sub-group
        df_report = df_report.sort_values(
            by=['Global_Importance', 'Feature', 'Sub_Group'],
            ascending=[False, True, True]
        )

        # Organize columns
        cols = ['Feature', 'Sub_Group', 'Direction_Logic', 'Global_Importance', 'Group_Direction_Score', 'Sample_Count']
        df_report = df_report[cols]

        df_report.to_csv(OUTPUT_REPORT, index=False)
        print(f"[SUCCESS] Done. Report saved to: {OUTPUT_REPORT}")
        
        # Print the first few rows to check the output structure
        print("\nPreview of the generated report:")
        print(df_report.head(15))  
    else:
        print("[INFO] No results generated from the analysis.")


if __name__ == "__main__":
    main()