import os
import json
import pandas as pd
import time
import torch
import gc
import torch.nn as nn
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === 0. Aggressive VRAM and Environment Configuration ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# --- 1. Configuration ---

# -- File Paths Configuration --
# TODO: Update these paths to point to your actual dataset and output directories before running
REPORT_1_DIR = "./data/report"
REPORT_2_PATH = "./data/A_Comprehensive_Risk_Assessment_Framework.txt"
CLUSTER_SAMPLES_CSV = "./data/cluster_samples.csv"
PARTICIPANT_DATA_JSON = "./data/participant_data_IG.json"
OUTPUT_DIR = "./output/IG_results"

# -- Task Range Configuration --
START_REPORT_INDEX = 0
END_REPORT_INDEX = None  # Set to None to process all reports

# -- Model Configuration --
# TODO: Set to your local model path or HuggingFace repo ID
MODEL_PATH = "path/to/your/local_model" # e.g., "Qwen/Qwen2.5-32B-Instruct"
max_memory_mapping = {
    0: "40GiB", 1: "40GiB", 2: "40GiB", 3: "40GiB",
    4: "40GiB", 5: "40GiB", 6: "40GiB", 7: "40GiB",
    "cpu": "800GiB"
}

# -- Integral Gradients (IG) Configuration --
IG_STEPS = 15
TARGET_KEY = "5_year_risk_percent"

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are a top-tier clinical AI reasoning expert. Your mission is to synthesize evidence for CKD risk assessment.

--- Evidence Source One ---
{report_1_content}

--- Evidence Source Two ---
{report_2_content}

Required JSON Output Format:
{{
    "evidence_summary": "<summary>",
    "reasoning_process": "<reasoning>",
    "risk_scores": {{
        "1_year_risk_percent": <float>,                
        "2_year_risk_percent": <float>,
        "3_year_risk_percent": <float>,
        "5_year_risk_percent": <float>
    }},
    "confidence": {{
        "confidence_score": <float>,
        "confidence_reasoning": "<reasoning>"
    }}
}}

Participant Information:
{patient_data_json}

Begin your analysis.
"""


# === Memory Management Functions ===
def clear_memory():
    """Force clear VRAM and synchronize CUDA streams"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_gpu_memory():
    """Print current VRAM status"""
    print("\n=== GPU Memory Usage ===")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: Alloc {allocated:.2f}GB / Res {reserved:.2f}GB")
    print()


# === Core Integral Gradients (IG) Class ===
class ManualIG:
    def __init__(self, model):
        self.model = model
        self.current_interpolated_embeds = None
        self.captured_tensor = None

    def _replace_embed_hook(self, module, input, output):
        if self.current_interpolated_embeds is not None:
            interpolated = self.current_interpolated_embeds.to(output.device).type(output.dtype)
            if not interpolated.requires_grad:
                interpolated.requires_grad_(True)
            interpolated.retain_grad()
            self.captured_tensor = interpolated
            return interpolated
        return output

    def attribute(self, full_input_ids, target_index, prompt_len, n_steps=20):
        """Execute Integral Gradients algorithm - Fixed Context mode"""
        print(f"\nStarting IG logic for token index: {target_index}")
        print(f"Applying Fixed-Context Baseline (Prompt Len: {prompt_len})...")

        embedding_layer = self.model.get_input_embeddings()
        target_token_id = full_input_ids[0, target_index].item()
        predict_pos = target_index - 1

        original_training = self.model.training
        self.model.train()
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad = False

        # === Phase 1: Prepare Baseline (CPU) ===
        print("Calculating Baseline...")
        clear_memory()

        with torch.no_grad():
            self.model.eval()
            embed_device = next(embedding_layer.parameters()).device

            original_embeds_gpu = embedding_layer(full_input_ids.to(embed_device))
            original_embeds_cpu = original_embeds_gpu.detach().cpu()
            del original_embeds_gpu
            clear_memory()

            baseline_embeds_cpu = original_embeds_cpu.clone()
            baseline_embeds_cpu[:, 1:prompt_len, :] *= 0.0

            self.current_interpolated_embeds = baseline_embeds_cpu
            handle = embedding_layer.register_forward_hook(self._replace_embed_hook)

            self.model.train()
            out = self.model(full_input_ids.to(self.model.device))
            baseline_score = out.logits[0, predict_pos, target_token_id].item()

            handle.remove()
            self.current_interpolated_embeds = None
            del out
            clear_memory()

        print(f"Baseline Score (with CoT fixed, Prompt zeroed): {baseline_score:.4f}")

        # === Phase 2: Integration Loop ===
        delta_cpu = original_embeds_cpu - baseline_embeds_cpu

        cot_delta_norm = delta_cpu[:, prompt_len:, :].norm().item()
        print(f"Verification: CoT part Delta Norm (Should be ~0): {cot_delta_norm}")

        total_gradients_cpu = torch.zeros_like(original_embeds_cpu, device='cpu')
        alphas = torch.linspace(0, 1, n_steps + 1)[1:]

        print(f"Running {n_steps} IG steps...")
        for step_idx, alpha in enumerate(tqdm(alphas, desc="IG Steps")):
            clear_memory()

            interpolated_cpu = baseline_embeds_cpu + alpha.item() * delta_cpu

            self.current_interpolated_embeds = interpolated_cpu
            handle = embedding_layer.register_forward_hook(self._replace_embed_hook)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(full_input_ids.to(self.model.device))
            target_score = outputs.logits[0, predict_pos, target_token_id]

            target_score.backward()

            if self.captured_tensor is not None and self.captured_tensor.grad is not None:
                grad_step = self.captured_tensor.grad.detach().cpu()
                total_gradients_cpu += grad_step
                self.captured_tensor.grad = None
                self.captured_tensor = None

            handle.remove()
            self.current_interpolated_embeds = None
            del outputs, target_score
            clear_memory()

        # === Phase 3: Result Calculation ===
        avg_gradients = total_gradients_cpu / n_steps
        ig_attributions = (delta_cpu * avg_gradients).sum(dim=-1).squeeze()

        if not original_training:
            self.model.eval()

        return ig_attributions.float().numpy(), baseline_score


# === Helper Function: Locate Target Token ===
def find_target_token_index(full_ids, tokenizer, target_key="5_year_risk_percent"):
    full_text = tokenizer.decode(full_ids[0])
    match = re.search(rf'"{target_key}":\s*(\d+(?:\.\d+)?)', full_text)
    if not match:
        return None, None

    number_str = match.group(1)
    print(f"Found target value in text: '{number_str}'")

    start_char_idx = match.start(1)
    prefix_text = full_text[:start_char_idx]
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    if full_ids[0, 0].item() == tokenizer.bos_token_id:
        target_idx = len(prefix_tokens) + 1
    else:
        target_idx = len(prefix_tokens)

    for offset in range(-2, 5):
        idx = target_idx + offset
        if 0 <= idx < full_ids.shape[1]:
            tk_str = tokenizer.decode([full_ids[0, idx]]).strip()
            if tk_str and (tk_str in number_str or number_str.startswith(tk_str)):
                print(f"Target Token found at idx {idx}: '{tk_str}' (ID: {full_ids[0, idx]})")
                return idx, full_ids[0, idx]

    return None, None


# === Data Loading Function ===
def load_data():
    """Load all required data files into memory."""
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
        print(f"[ERROR] Data file not found - {e}. Please check the paths in the script.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during data loading: {e}")
        exit(1)


# === Helper Function: Format Attribution Results ===
def format_attribution_data(generated_ids, ig_scores, prompt_len, tokenizer):
    """
    Align generated Token IDs with IG scores to generate a detailed data list.
    """
    # 1. Prepare data
    # Ensure conversion to numpy and flatten
    if torch.is_tensor(generated_ids):
        all_tokens = generated_ids[0].cpu().numpy()
    else:
        all_tokens = np.array(generated_ids[0])

    if torch.is_tensor(ig_scores):
        all_scores = ig_scores.cpu().numpy()
    elif isinstance(ig_scores, list):
        all_scores = np.array(ig_scores)
    else:
        all_scores = ig_scores

    # 2. Create detailed data list
    detailed_records = []

    for i, (token_id, score) in enumerate(zip(all_tokens, all_scores)):
        # Decode Token, handle special characters
        token_str = tokenizer.decode([token_id])
        # Use repr to handle invisible characters like newlines, remove quotes at both ends
        token_str_safe = repr(token_str)[1:-1]

        # Determine if it is Prompt or Generation
        section = "Prompt" if i < prompt_len else "Generation"

        record = {
            "index": i,
            "token_id": int(token_id),
            "token": token_str_safe,  # Readable text
            "score": float(score),  # IG score
            "abs_score": float(abs(score)),  # Absolute value for sorting
            "section": section  # Section marker
        }
        detailed_records.append(record)

    return detailed_records

# === Single Sample IG Processing Function ===
def process_single_sample_ig(sample_id, participant_info, report_1_content, report_2_content,
                              model, tokenizer, explainer):
    """
    Execute IG calculation for a single sample.
    Returns: Result dictionary
    """
    result = {}
    generated_ids = None
    inputs = None

    try:
        # Assemble prompt
        patient_data_json_str = json.dumps(participant_info)

        final_prompt = PROMPT_TEMPLATE.format(
            report_1_content=report_1_content,
            report_2_content=report_2_content,
            patient_data_json=patient_data_json_str
        )

        messages = [{"role": "user", "content": final_prompt}]

        # === Generate Response ===
        print(f"\nGenerating response for sample {sample_id}...")
        model.eval()

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs['input_ids'].shape[1]
        print(f"Prompt length: {prompt_len} tokens")

        with torch.no_grad():
            model.config.use_cache = True
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32768,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            model.config.use_cache = False

        generated_text = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
        print(f"Generated Preview:\n{generated_text[:500]}...")

        # Release inputs
        del inputs
        inputs = None
        clear_memory()

        # === Locate Target and Run IG ===
        target_idx, target_token_id = find_target_token_index(generated_ids, tokenizer, TARGET_KEY)

        if target_idx is not None:
            print(f"\nRunning IG for position {target_idx}...")

            ig_scores, baseline_score = explainer.attribute(
                generated_ids,
                target_idx,
                prompt_len=prompt_len,
                n_steps=IG_STEPS
            )

            # Extract IG scores for the Prompt section
            # === Use new formatting logic ===

            # Call helper function to get complete Token-Score mapping
            all_records = format_attribution_data(
                generated_ids,
                ig_scores,
                prompt_len,
                tokenizer
            )

            # 1. Sort by absolute score, extract Top K (e.g., Top 100)
            # This sorting covers Prompt and Generation (though Baseline strategy may cause Generation score to be 0)
            top_influential = sorted(all_records, key=lambda x: x['abs_score'], reverse=True)

            # 2. Extract only Prompt section records (in original order for easy heatmap visualization)
            prompt_records_ordered = [r for r in all_records if r['section'] == "Prompt"]

            result = {
                "status": "success",
                "generated_text": generated_text,
                "prompt_len": prompt_len,
                "target_idx": int(target_idx),
                "target_token_id": int(target_token_id),
                "baseline_score": float(baseline_score),

                # --- New/Modified Fields ---
                # 1. Top 100 with details, each element contains {index, token, score, section}
                "top_influential_tokens": top_influential,

                # 2. Complete Prompt data (including Token text), convenient for direct plotting or exporting to Excel
                # More intuitive than the previous pure list of scores
                "prompt_analysis": prompt_records_ordered,

                # Retain original simplified data to prevent compatibility issues (optional)
                "raw_ig_scores": [r['score'] for r in prompt_records_ordered]
            }
            # === End formatting logic ===
        else:
            result = {
                "status": "target_not_found",
                "generated_text": generated_text,
                "error": f"Could not find target key '{TARGET_KEY}' in output"
            }

    except Exception as e:
        import traceback
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        # === CRITICAL: Force clear VRAM regardless of success or failure ===
        if generated_ids is not None:
            del generated_ids
        if inputs is not None:
            del inputs
        clear_memory()

    return result


# === Cluster Processing Function ===
def process_cluster(cluster_id, sample_ids, report_2_content, all_participant_data,
                    model, tokenizer, explainer):
    """Process all samples for a single cluster"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"[Cluster {cluster_id}] Starting to process {len(sample_ids)} samples.")
    print(f"{'='*60}")

    cluster_results = {}
    report_1_path = os.path.join(REPORT_1_DIR, f"{cluster_id}.txt")

    try:
        with open(report_1_path, 'r', encoding='utf-8') as f:
            report_1_content = f.read()
    except FileNotFoundError:
        print(f"[Cluster {cluster_id}] ERROR: Report 1 file not found at {report_1_path}. Skipping this cluster.")
        return cluster_id, 0, len(sample_ids)

    successful_count = 0
    for i, sample_id in enumerate(sample_ids):
        sample_id_str = str(sample_id)
        print(f"\n[Cluster {cluster_id}] Processing sample {i + 1}/{len(sample_ids)} (ID: {sample_id_str})...")
        print_gpu_memory()

        participant_info = all_participant_data.get(sample_id_str)
        if not participant_info:
            print(f"[Cluster {cluster_id}] WARNING: Participant data not found for sample ID {sample_id_str}. Skipping.")
            cluster_results[sample_id_str] = {"status": "skipped", "error": "participant data not found"}
            # Clean up even if skipped
            clear_memory()
            continue

        # Execute IG calculation
        result = process_single_sample_ig(
            sample_id_str,
            participant_info,
            report_1_content,
            report_2_content,
            model,
            tokenizer,
            explainer
        )

        cluster_results[sample_id_str] = result

        if result.get("status") == "success":
            successful_count += 1
            print(f"[Cluster {cluster_id}] Sample {sample_id_str} IG calculation successful.")
        else:
            print(f"[Cluster {cluster_id}] Sample {sample_id_str} processing status: {result.get('status')} - {result.get('error', '')}")

        # Force clear and print VRAM status after each sample
        clear_memory()
        print_gpu_memory()

    # Save results for the entire cluster
    output_path = os.path.join(OUTPUT_DIR, f"{cluster_id}_ig.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_results, f, ensure_ascii=False, indent=2)
        duration = time.time() - start_time
        print(f"\n[Cluster {cluster_id}] Processing completed. Results saved to {output_path}. Time elapsed: {duration:.2f} seconds.")
    except Exception as e:
        print(f"[Cluster {cluster_id}] ERROR: Cannot save results to {output_path}. Error: {e}")

    return cluster_id, successful_count, len(sample_ids) - successful_count


# === Main Function ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    report_2_content, cluster_map, participant_data = load_data() 

    # Get cluster list
    try:
        report_files = [f for f in os.listdir(REPORT_1_DIR) if f.endswith('.txt')]
        all_cluster_ids = sorted([int(os.path.splitext(f)[0]) for f in report_files])
    except Exception as e:
        print(f"[ERROR] Error reading cluster reports from {REPORT_1_DIR}: {e}")
        return

    if END_REPORT_INDEX is None:
        selected_cluster_ids = all_cluster_ids[START_REPORT_INDEX:]
    else:
        selected_cluster_ids = all_cluster_ids[START_REPORT_INDEX:END_REPORT_INDEX]

    print(f"[INFO] Found {len(all_cluster_ids)} reports in total.")
    print(f"[INFO] Processing range from index {START_REPORT_INDEX} to {END_REPORT_INDEX if END_REPORT_INDEX else len(all_cluster_ids)}, total {len(selected_cluster_ids)} reports.")

    # === Load Model ===
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model with CPU Offload headroom...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        max_memory=max_memory_mapping,
        offload_folder="./offload_weights_temp",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )

    # === Model Configuration ===
    print("Configuring model for IG...")
    model.gradient_checkpointing_enable()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    try:
        model.enable_input_require_grads()
    except Exception as e:
        print(f"Warning setting input grads: {e}")

    print_gpu_memory()

    # Create IG explainer
    explainer = ManualIG(model)

    # === Process all clusters sequentially (no multithreading to avoid VRAM issues) ===
    total_success = 0
    total_fail = 0

    for cluster_id in selected_cluster_ids:
        sample_ids = cluster_map.get(cluster_id, [])
        if not sample_ids:
            print(f"[Cluster {cluster_id}] No samples, skipping.")
            continue

        _, success, fail = process_cluster(
            cluster_id,
            sample_ids,
            report_2_content,
            participant_data,
            model,
            tokenizer,
            explainer
        )
        total_success += success
        total_fail += fail

        # Clear VRAM after each cluster
        clear_memory()

    print("\n" + "="*60)
    print("All clusters have been processed")
    print("="*60)
    print(f"Total successful IG calculations: {total_success}")
    print(f"Total failures: {total_fail}")
    print(f"Results saved in: {OUTPUT_DIR}")

    # Clean up offload folder
    try:
        import shutil
        shutil.rmtree("./offload_weights_temp")
        print("Cleaned up offload folder.")
    except:
        pass


if __name__ == "__main__":
    main()