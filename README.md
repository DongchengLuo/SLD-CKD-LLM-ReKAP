🧬 LLM-ReKAP: A Knowledge Graph-Augmented Causal-Retriever LLM Framework
LLM-ReKAP (Large Language Model-Retriever Knowledge Graph-Augmented Predictive framework) is an end-to-end, training-free neuro-symbolic research framework. It is designed to synthesize unstructured biomedical literature into an evidence-weighted Knowledge Graph (KG) and perform personalized clinical risk assessment—specifically predicting incident Chronic Kidney Disease (CKD) in patients with Steatotic Liver Disease (SLD).
By replacing traditional flat embedding-based Retrieval-Augmented Generation (RAG) with an LLM-as-causal-retriever, this framework grounds predictive LLM ensembles in multi-dimensional domain knowledge, bridging the gap between general epidemiological scores (like CKD-PC) and complex, patient-specific cross-organ crosstalk.
🌟 Key Innovations & Features
LLM-as-Causal-Retriever: Departs from standard "naive" vector similarity RAG. It performs deep semantic matching across entities, relationships, and multidimensional literature metadata to extract and pre-synthesize noise-reduced, clinical conflict-reconciled Knowledge Summaries.
Dual-LLM Engine: Leverages Google Gemini 2.5 Pro for high-fidelity KG extraction/synthesis, and utilizes high-performance open-source foundation models (e.g., Qwen, Kimi, DeepSeek, GPT) guided by Chain-of-Thought (CoT) protocols for risk prediction.
Training-Free In-Context Learning: Integrates general clinical benchmarks (e.g., CKD-PC) with domain-specific KG insights without gradient-based retraining or parameter optimization, inherently preventing overfitting.
Interactive End-to-End Clinical Interface: A centralized Streamlit-based WebUI that automates the workflow from raw PDF parsing to visually interpretable risk stratification.
4D Explainability (XAI): Demystifies LLM reasoning through a four-dimensional interpretability strategy: Chain-of-Thought (CoT) logical tracing, SHapley Additive exPlanations (SHAP), token-level Integrated Gradients (IG), and Mechanistic Surrogate Cox Modeling.
GPU-Accelerated High-Dimensional Processing: Utilizes NVIDIA cuML for rapid K-Means phenotypic clustering to process large-scale cohorts efficiently.
🚀 Quick Start
1. Environment Setup
Python 3.12+ is recommended. For GPU-accelerated features (like patient clustering), please ensure a RAPIDS environment is installed alongside vLLM.
# Clone the repository
git clone [https://github.com/your-username/LLM-ReKAP.git](https://github.com/your-username/LLM-ReKAP.git)
cd LLM-ReKAP

# Install dependencies
pip install -r requirements.txt


2. Deploy the WebUI (Recommended)
The easiest way to operate the framework is via the integrated clinical reasoning dashboard.
streamlit run webui.py


Once started, access the UI at http://localhost:8501. The WebUI aligns with the four core operational modules:
Knowledge Graph Construction: Upload raw medical PDFs. The system utilizes customizable prompts to extract entities and map canonical names (standardization).
Knowledge Synthesis & Evaluation: The system summarizes standardized entities into coherent clinical narratives and scores biomedical relationships using OCEBM/GRADE evidence quality principles, outputting a unified JSON KG.
Knowledge Graph Retrieval: Input patient clinical profiles (JSON). The system ranks and retrieves tailored evidence directly from the literature and KG base.
Comprehensive Reasoning: Integrates patient phenotypes, consensus clinical rules, and the retrieved KG evidence. Outputs structured CoT reasoning, projected 3–15 year risk probabilities, and seamlessly integrates with traditional scores (e.g., CKD-PC) for definitive High/Low risk stratification.
📂 Modular Architecture
For researchers conducting ablation studies, cross-cohort validations, or custom developments, the framework is heavily decoupled:
Core Engines (/modules)
Module
Script
Description
KG Construction
kg_construction/
Multimodal PDF extraction (KG_extraction.py), entity normalization, and generative narrative aggregation.
Smart Retrieval
retrieval/
Causal semantic matching, composite score ranking, and participant-level evidence filtering.
Patient Clustering
clustering/
(cluster.py) GPU-accelerated cohort stratification via cuML to identify shared phenotypic representations.
Reasoning & Ensembles
prediction_work/
CoT inference engine combining KG Summaries (Report 1) and Consensus Clinical Risk (Report 2) with the foundational clinical baseline.

Interpretability Pipeline (/interpretable_analysis)
CoT_trace.py: Extracts and structures unstructured LLM rationales into functional clusters (e.g., Metabolic-Renal axis).
SHAP_analysis.py & IG_tokens.py: Computes global feature importance and granular, token-level feature attributions to trace the exact drivers of the LLM's prediction.
Surrogate_Cox.py: Maps non-linear LLM reasoning nodes onto conventional statistical Hazard Ratios for mechanistic corroboration.
🛠️ Configuration
Update the configuration parameters in the WebUI sidebar or script headers before running:
API Keys & Model Endpoints: Enter your Gemini API keys for KG construction, and configure your vLLM local endpoints (or API keys) for your chosen prediction foundation models (e.g., Qwen, DeepSeek) in the WebUI sidebar.
Paths: If using independent scripts, modify the PATH_ variables at the top of each script to point to your local longitudinal datasets (e.g., UKBB, Nanfang formats) and evidence corpora.
📝 Citation
If you use the LLM-ReKAP framework or our pre-synthesized SLD-CKD knowledge graphs in your research, please cite our paper:
@article{LLMReKAP2026,
  title={LLM-ReKAP: A Knowledge Graph-Augmented Causal-Retriever LLM Framework for Incident CKD Prediction in Steatotic Liver Disease},
  author={Your Name and Collaborators},
  journal={GitHub Repository},
  year={2026},
  url={[https://github.com/your-username/LLM-ReKAP](https://github.com/your-username/LLM-ReKAP)}
}


📄 License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
