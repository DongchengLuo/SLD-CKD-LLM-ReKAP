# **🧬 LLM-ReKAP: Knowledge Graph-Augmented Causal-Retriever LLM Framework for Incident CKD Prediction in Steatotic Liver Disease**



**LLM-ReKAP (LLM-Retriever Knowledge Graph-Augmented Predictive Framework)** is an end-to-end research framework designed to extract knowledge from 20 years of biomedical literature, construct evidence-weighted Knowledge Graphs (KGs), and provide clinical decision support for patients with Steatotic Liver Disease (SLD)—including NAFLD, MAFLD, or MASLD—specifically for Chronic Kidney Disease (CKD) risk prediction.prediction.



## **🌟 Key Features**



* **Neuro-Symbolic KG-RAG Architecture**: Replaces embedding-based retrieval with LLM-as-causal-retriever, grounding predictions in a domain-specific, evidence-weighted KG distilled from 412 independent studies.  

* **Multi-LLM Ensemble Engine**: Supports four foundation models (Qwen3.5-397b-A17b, Kimi-K2, DeepSeek-V3.1, GPT-OSS-120b) for Chain-of-Thought (CoT) reasoning and risk probability computation.  

* **Training-Free In-Context Learning**: Avoids gradient-based overfitting by anchoring risk stratification to prior domain knowledge, achieving zero-shot generalization across independent cohorts (N=133,086).  

* **Four-Dimensional Interpretability**:  Built-in CoT reasoning trace analysis, SHAP, Integrated Gradients (IG), and Surrogate Cox Modeling to ensure transparency and biological plausibility.

* **Superior Predictive Performance**: Achieves C-index of 0.833 (ΔC=0.037) in UK Biobank and 0.878 (ΔC=0.015) in Nanfang cohort, outperforming naive RAG by >2-fold incremental gain.



## **🚀 Quick Start**



### **1\. Environment Setup**



Python 3.9+ is recommended. For GPU-accelerated features (cluster.py), please ensure a [RAPIDS](https://rapids.ai/) environment is installed.



\# Clone the repository  

git clone \[https://github.com/your-username/BioKG-Risk.git\](https://github.com/your-username/BioKG-Risk.git)  

cd BioKG-Risk



\# Install dependencies  

pip install \-r requirements.txt



### **2\. Deploy WebUI (Recommended)**



This is the easiest way to use the framework. Launch the integrated management dashboard:



streamlit run webui.py



Once started, access the UI at http://localhost:8501. The WebUI includes four core tabs:



1. **PDF Entity Extraction**: Upload medical PDFs to generate structured JSON knowledge.  

2. **Entity Normalization**: Cleanse and merge synonymous entities (e.g., matching "CKD" to "Chronic Kidney Disease").  

3. **Entity Summarization**: Generate cross-literature summaries for specific biomarkers or risk factors.  

4. **Relationship Scoring**: Quantify evidence strength based on the **GRADE** principles.



## **📂 Modular Architecture**

For researchers performing ablation studies or deep customization, the following core modules are available:

### **Core Engines (/modules)**

| Module | Script | Description |
| :--- | :--- | :--- |
| **KG Construction** | `kg_construction/` | Includes PDF extraction (`KG_extraction.py`) and entity summarization logic. |
| **Smart Retrieval** | `retrieval/` | Semantic filtering of literature, entities, and relationships based on patient profiles. |
| **Patient Clustering** | `clustering/` | (`cluster.py`) GPU-accelerated stratification using cuML to identify subgroup phenotypes. |
| **Reasoning & Score** | `prediction_work/` | Multi-source evidence synthesis (e.g., report1+2 mode) for clinical conclusions. |


### **Interpretable Analysis (/interpretable analysis)**



* **SHAP.py & IG.py**: Analyze the contribution of patient metrics to risk scores and generate attribution heatmaps suitable for academic publication.



## **🛠️ Configuration**



Update the configuration parameters in the WebUI sidebar or script headers before running:



* **API Keys**: Enter your Gemini or OpenAI/NVIDIA API keys in the WebUI sidebar.  

* **Paths**: If using independent scripts, modify the PATH\_ variables at the top of each script to point to your local datasets.



## **📝 Citation**



If you use this framework in your research, please cite:



@article{LLM-ReKAP,  

  title={A Modular Framework for Biomedical Knowledge Graph Construction and Clinical Risk Assessment in Metabolic Diseases},  

  author={Dongcheng Luo},  

  journal={GitHub Repository},  

  year={2026},  

  url={\[https://github.com/your-username/BioKG-Risk\](https://github.com/your-username/BioKG-Risk)}  

}



## **📄 License**



This project is licensed under the Apache License 2.0. See the LICENSE file for details.
