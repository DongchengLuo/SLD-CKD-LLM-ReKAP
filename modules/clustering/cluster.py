import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from cuml.cluster import KMeans  # Import cuML KMeans for GPU acceleration
import cupy as cp  # Import CuPy for GPU-accelerated NumPy equivalent operations
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# --- Configuration Area: File Paths ---
# ==============================================================================
# TODO: Update these paths to point to your actual dataset files before running
PATH_LK_KG = "./data/LK_KG_deepresearch.json"
PATH_ABSTRACT = "./data/abstract435.json"
PATH_RETRIEVAL_1001 = "./data/re_retrieval_1001.json"
PATH_RETRIEVAL_1002 = "./data/re_retrieval_1002.json"
PATH_RETRIEVAL_1003 = "./data/re_retrieval_1003.json"

# ==============================================================================
# --- Load Data ---
# ==============================================================================
print("[INFO] Loading JSON data files...")
with open(PATH_LK_KG, "r", encoding="utf-8") as f:
    LK_KG = json.load(f)
with open(PATH_ABSTRACT, "r", encoding="utf-8") as f:
    abstract435 = json.load(f)
with open(PATH_RETRIEVAL_1001, "r", encoding="utf-8") as f:
    re_retrieval_1001 = json.load(f)
with open(PATH_RETRIEVAL_1002, "r", encoding="utf-8") as f:
    re_retrieval_1002 = json.load(f)
with open(PATH_RETRIEVAL_1003, "r", encoding="utf-8") as f:
    re_retrieval_1003 = json.load(f)


# ==============================================================================
# --- Core Logic: Data Processing & Integration ---
# ==============================================================================

# 1. Create a mapping dictionary from DOI to paper_title
# This is the most efficient method, avoiding repeated lookups in the loop
doi_to_title_map = {
    doi: info.get('paper_title', doi)  # Fallback to original DOI if 'paper_title' is missing
    for doi, info in abstract435.items()
    if isinstance(info, dict)  # Ensure value is a dict to avoid data format issues
}

# 2. Prepare a new dictionary to store results with replaced key names
results_with_titles = {}

# 3. Iterate through each element of re_retrieval_1002
for doc_id, inner_dict in re_retrieval_1002.items():
    # Efficiently create a new inner dictionary using dictionary comprehension
    new_inner_dict = {
        # .get(doi, doi) is a safe lookup method:
        # - Uses mapped title if DOI is found
        # - Uses original DOI if not found, preventing data loss
        doi_to_title_map.get(doi, doi): score
        for doi, score in inner_dict.items()
    }
    results_with_titles[doc_id] = new_inner_dict

# Initialize dictionaries to store statistics
statistics_dict_entities = {}
statistics_dict = {}

# Iterate through the re_retrieval_1001 object for entity statistics
for id_key, data in re_retrieval_1001.items():
    selected_entities = data.get('selected_entities', [])
    statistics_dict_entities[id_key] = {}

    # Iterate through each literature in the LK_KG data
    for literature in LK_KG:
        paper_title = literature['document_metadata']['paper_title']
        extracted_entities = literature.get('extracted_entities', [])

        # Extract entity_name_as_in_text from each entity
        entity_ids = [ent['entity_name_as_in_text'] for ent in extracted_entities]

        # Count the number of matches
        count = sum(1 for ent_id in selected_entities if ent_id in entity_ids)

        # Store the count in the statistics dictionary
        if paper_title not in statistics_dict_entities[id_key]:
            statistics_dict_entities[id_key][paper_title] = 0
        statistics_dict_entities[id_key][paper_title] += count

# Iterate through the re_retrieval_1003 object for relationship statistics
for id_key, data in re_retrieval_1003.items():
    selected_relationships = data.get('selected_relationships', [])
    statistics_dict[id_key] = {}

    # Iterate through each literature in the LK_KG data
    for literature in LK_KG:
        paper_title = literature['document_metadata']['paper_title']
        extracted_relationships = literature.get('extracted_relationships', [])

        # Extract relationship_internal_id from each relationship
        relationship_ids = [rel['relationship_internal_id'] for rel in extracted_relationships]

        # Count the number of matches
        count = sum(1 for rel_id in selected_relationships if rel_id in relationship_ids)

        # Store the count in the statistics dictionary
        if paper_title not in statistics_dict[id_key]:
            statistics_dict[id_key][paper_title] = 0
        statistics_dict[id_key][paper_title] += count


# ==============================================================================
# --- Core Logic: Integration ---
# ==============================================================================

# 1. Find common first-level keys (intersection of IDs) among the three dictionaries
keys1 = set(results_with_titles.keys())
keys2 = set(statistics_dict.keys())
keys3 = set(statistics_dict_entities.keys())

common_doc_ids = keys1 & keys2 & keys3
print(f"[INFO] Common Document IDs found: {list(common_doc_ids)}")

# 2. Prepare a new dictionary to store the final combined results
combined_data = {}

# 3. Iterate through these common IDs
for doc_id in common_doc_ids:

    # Create a new dictionary to store internal paper data for the current ID
    merged_papers_for_id = {}

    # Get the inner dictionary corresponding to the current ID
    titles_abstract = results_with_titles.get(doc_id, {})
    titles_relationship = statistics_dict.get(doc_id, {})
    titles_entity = statistics_dict_entities.get(doc_id, {})

    # 4. Iterate using the paper_title keys of statistics_dict as the baseline
    for paper_title, relationship_value in titles_relationship.items():
        # a. Get abstract score from results_with_titles; default to 0 if absent
        abstract_score = titles_abstract.get(paper_title, 0)

        # b. Get entity value from statistics_dict_entities
        #    Assuming if a title exists in statistics_dict, it also exists in statistics_dict_entities
        entity_value = titles_entity.get(paper_title)

        # c. Assemble into the final structure
        merged_papers_for_id[paper_title] = {
            'abstract': abstract_score,
            'relationship': relationship_value,
            'entity': entity_value
        }

    # Store the assembled inner dictionary into the final result
    combined_data[doc_id] = merged_papers_for_id


# ==============================================================================
# --- Step 1 & 2: Data Transformation and Normalization ---
# ==============================================================================

# 1. Convert dictionary to long-format DataFrame
records = []
for sample_id, papers in combined_data.items():
    for paper_title, values in papers.items():
        records.append({
            'sample_id': sample_id,
            'title': paper_title,
            'abstract': values['abstract'],
            'entity': values['entity'],
            'relationship': values['relationship']
        })

df = pd.DataFrame(records)

# 2. Feature Normalization
features = ['abstract', 'entity', 'relationship']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

print("\n--- Step 1: Data Transformation and Normalization Completed ---")
print(df.head())

# ==============================================================================
# --- Step 3 & 4: Composite Scoring and Top-K Core Literature Extraction ---
# ==============================================================================

# 3. Calculate composite score (using simple weighting, all weights=1 here)
# You can adjust weights based on your requirements, e.g.:
# weights = {'abstract': 0.5, 'entity': 0.3, 'relationship': 0.2}
# df['score'] = df['abstract'] * weights['abstract'] + ...
df['score'] = df['abstract'] + df['entity'] + df['relationship']

# 4. Extract Top-K core literature sets for each sample
K = 50  # Define the number of core literature to extract per sample, adjust as needed

top_k_sets = df.groupby('sample_id').apply(
    lambda x: set(x.nlargest(K, 'score')['title'])
).tolist()

print(f"\n--- Step 2: Extraction of Top-{K} Core Literature Sets Completed ---")
# Result is a list, each element is a core literature set for a sample
print(top_k_sets)


# ==============================================================================
# --- Step 5: Construct K-Means Feature Matrix ---
# ==============================================================================
print("\n--- Step 5: Constructing K-Means Feature Matrix ---")

# Use pivot_table to convert the long-format df to a wide-format matrix
# Each row represents a sample (sample_id)
# Each column represents a paper (title)
# Cell values are the composite score of that paper for that sample
# fill_value=0 indicates a score of 0 if a sample lacks a particular paper
feature_df = df.pivot_table(index='sample_id', columns='title', values='score', fill_value=0)

# Preserve the order of sample IDs to map cluster results back later
sample_ids_order = feature_df.index.tolist()

# Convert DataFrame to NumPy matrix, which serves as our input X
X = feature_df.values

print(f"Feature matrix construction completed. Shape: {X.shape}")
print(f" (Represents {X.shape[0]} samples and {X.shape[1]} paper features)")

# ==============================================================================
# --- Step 6: Finding Optimal K using the "Elbow Method" (GPU Accelerated) ---
# ==============================================================================
print("\n--- Step 6: Using 'Elbow Method' to find Optimal K (GPU Accelerated) ---")

# Move data from CPU memory (NumPy) to GPU memory (CuPy)
X_gpu = cp.asarray(X, dtype=cp.float32)

# User inputs optimal K value based on their graph analysis
optimal_k = int(input("\nBased on the 'elbow' graph above, please enter your perceived optimal K value: "))

# ==============================================================================
# --- Step 7: Perform Final Clustering with Optimal K (GPU Accelerated) ---
# ==============================================================================
print(f"\n--- Step 7: Performing Final Clustering with K={optimal_k} (GPU Accelerated) ---")

kmeans_final_gpu = KMeans(n_clusters=optimal_k, random_state=42)
# fit_predict runs directly on GPU and returns a GPU array of labels
cluster_labels_gpu = kmeans_final_gpu.fit_predict(X_gpu)

# *** Crucial: Download results from GPU memory back to CPU memory for Pandas use ***
cluster_labels_cpu = cluster_labels_gpu.get()  # .get() is standard method from CuPy to NumPy array

print("Clustering completed! Organizing and analyzing results...")

# Create a results DataFrame to associate sample IDs with their respective cluster labels
results_df = pd.DataFrame({
    'sample_id': sample_ids_order,
    'cluster': cluster_labels_cpu
})

# Analyze the size of each cluster
print("\nSample Distribution Across Clusters:")
print(results_df['cluster'].value_counts().sort_index())

# Display some example sample IDs for each cluster
print("\nExample Sample IDs per Cluster:")
for i in range(optimal_k):
    # Filter sample IDs belonging to the current cluster
    sample_ids_in_cluster = results_df[results_df['cluster'] == i]['sample_id'].tolist()
    # Print cluster information and the first 5 sample IDs
    print(f"  - Cluster {i} (Total {len(sample_ids_in_cluster)} samples): {sample_ids_in_cluster[:5]}...")

# Free GPU memory
del X_gpu
del cluster_labels_gpu
cp.get_default_memory_pool().free_all_blocks()
print("\nGPU memory has been freed.")

# ==============================================================================
# --- Step 8: Validation - Analyze Top-K Paper Overlap within Clusters ---
# ==============================================================================
print("\n--- Step 8: Analyzing Top-K Papers per Cluster to Validate Clustering ---")

# --- Set the number of Top-K papers you want to analyze here ---
K_for_analysis = 20  # You can modify this value, e.g., 10, 20, 100, etc.

print(f"\nUsing the core literature set of Top-{K_for_analysis} for clustering validation...")

# --- Preparation: Recalculate Top-K paper sets for each sample based on the K set above ---
print("Re-extracting Top-K literature sets for each sample...")
sample_id_to_top_k_map = df.groupby('sample_id').apply(
    lambda x: set(x.nlargest(K_for_analysis, 'score')['title'])
).to_dict()

results_df['top_k_papers'] = results_df['sample_id'].map(sample_id_to_top_k_map)

print("Top-K literature sets successfully linked to each sample.")

# --- Core Analysis: Calculate unique Top-K literature per cluster ---
print("\n--- Cluster Core Literature Analysis Results ---")

cluster_analysis_results = []

for cluster_id in sorted(results_df['cluster'].unique()):
    cluster_df = results_df[results_df['cluster'] == cluster_id]
    num_samples_in_cluster = len(cluster_df)
    list_of_sets = cluster_df['top_k_papers'].tolist()

    if list_of_sets:
        unique_papers_in_cluster = set.union(*list_of_sets)
        num_unique_papers = len(unique_papers_in_cluster)
        focus_ratio = num_unique_papers / num_samples_in_cluster
    else:
        unique_papers_in_cluster = set()
        num_unique_papers = 0
        focus_ratio = 0

    # ==================================================================
    cluster_analysis_results.append({
        'Cluster_ID': cluster_id,
        'Number_of_Samples': num_samples_in_cluster,
        'Unique_Core_Papers_in_Cluster': num_unique_papers,
        'Focus_Ratio_Lower_is_Better': f"{focus_ratio:.2f}",
        'Core_Paper_List': str(list(unique_papers_in_cluster))
    })
    # ==================================================================

# Convert the analysis results to a clean DataFrame and print
analysis_summary_df = pd.DataFrame(cluster_analysis_results)

# Set Pandas display options for better console output visibility
pd.set_option('display.max_colwidth', 150)  # Set max column width
print(analysis_summary_df.to_string())

# Easy option to save the results to a CSV file
# output_csv_path = './output/cluster_analysis_summary.csv'
# analysis_summary_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
# print(f"\nAnalysis results saved to: {output_csv_path}")