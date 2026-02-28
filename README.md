# **🧬 BioKG-Risk: Framework for Biomedical Knowledge Graph Construction and Clinical Risk Assessment**

**BioKG-Risk** is an end-to-end research framework designed to extract knowledge from massive unstructured biomedical literature, construct standardized Knowledge Graphs (KGs), and provide clinical decision support for specific patient cohorts—such as those with NAFLD, MAFLD, or MASLD—specifically for Chronic Kidney Disease (CKD) risk prediction.

## **🌟 Key Features**

* **Integrated Interactive Experience**: A centralized Streamlit-based interface (webui.py) covering the entire workflow from PDF parsing to final risk scoring.  
* **Dual-LLM Engine**: Leverages **Google Gemini** for high-precision knowledge extraction and supports **OpenAI/Local Models** (e.g., Kimi, DeepSeek, GLM) for complex clinical reasoning.  
* **Modular Architecture**: Decoupled backend logic allowing researchers to independently invoke modules for knowledge extraction, entity normalization, or semantic retrieval.  
* **GPU-Accelerated Clustering**: Utilizes **NVIDIA cuML** for high-speed K-Means clustering, supporting phenotype stratification for large-scale patient cohorts.  
* **XAI (Explainable AI) Support**: Built-in **SHAP** and **Integrated Gradients (IG)** tools to ensure transparency and traceability of clinical prediction results.

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
| :---- | :---- | :---- |
| **KG Construction** | kg\_construction/ | Includes PDF extraction (KG\_extraction.py) and entity summarization logic. |
| **Smart Retrieval** | retrieval/ | Semantic filtering of literature, entities, and relationships based on patient profiles. |
| **Patient Clustering** | clustering/ | (cluster.py) GPU-accelerated stratification using cuML to identify subgroup phenotypes. |
| **Reasoning & Score** | prediction\_work/ | Multi-source evidence synthesis (e.g., report1+2 mode) for clinical conclusions. |

### **Interpretable Analysis (/interpretable analysis)**

* **SHAP.py & IG.py**: Analyze the contribution of patient metrics to risk scores and generate attribution heatmaps suitable for academic publication.

## **🛠️ Configuration**

Update the configuration parameters in the WebUI sidebar or script headers before running:

* **API Keys**: Enter your Gemini or OpenAI/NVIDIA API keys in the WebUI sidebar.  
* **Paths**: If using independent scripts, modify the PATH\_ variables at the top of each script to point to your local datasets.

## **📝 Citation**

If you use this framework in your research, please cite:

@article{BioKGRisk2026,  
  title={A Modular Framework for Biomedical Knowledge Graph Construction and Clinical Risk Assessment in Metabolic Diseases},  
  author={Your Name and Collaborators},  
  journal={GitHub Repository},  
  year={2026},  
  url={\[https://github.com/your-username/BioKG-Risk\](https://github.com/your-username/BioKG-Risk)}  
}

## **📄 License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).