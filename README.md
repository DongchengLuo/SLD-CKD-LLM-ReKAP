BioKG-Clinical-Risk-Analysis: A Modular Framework for Biomedical Knowledge Graph Construction and Clinical Risk Assessment
This repository provides an end-to-end framework for extracting, standardizing, and analyzing biomedical knowledge from research literature to support clinical decision-making, specifically focusing on Chronic Kidney Disease (CKD) risk prediction in NAFLD/MAFLD/MASLD patients.

This framework transitions from unstructured PDF literature to structured knowledge graphs and final clinical risk scores, integrating LLM-based reasoning and GPU-accelerated clustering.

🚀 Quick Start: WebUI Deployment
The project is designed to be fully operational via a Streamlit-based WebUI. This is the recommended entry point for most users.

1. Prerequisites
Ensure you have Python 3.9+ installed and the necessary hardware dependencies (especially if using GPU acceleration features).

Bash
# Install dependencies
pip install -r requirements.txt
2. Launch the Application
Run the main entry point to start the interactive interface:

Bash
streamlit run webui.py
After launching, open your browser to the URL provided by the Streamlit console (usually http://localhost:8501).

📂 Project Structure
For researchers interested in customizing the pipeline or integrating specific components into their own work, the repository is organized into a modular structure:

Plaintext
.
├── webui.py                 # Core deployment interface (Main entry point)
├── interpretable analysis/  # Explainability tools (IG, SHAP for model transparency)
│   ├── IG.py
│   ├── IG_aggregation.py
│   └── SHAP.py
└── modules/                 # Modular backend components
    ├── clustering/          # GPU-accelerated clustering (cuML) for patient stratification
    ├── kg_aggregation/      # Knowledge graph summary generation
    ├── kg_construction/     # Logic for PDF extraction and entity summarization
    ├── prediction_work/     # Logic for multi-source evidence synthesis
    └── retrieval/           # LLM-based filters for entities, relationships, and articles
🔬 Core Components Overview
Knowledge Extraction (modules/kg_construction/): Automates the conversion of medical PDFs into structured JSON-based knowledge graphs. It handles entity summarization and group-level aggregation.

Intelligent Retrieval (modules/retrieval/): Implements semantic filtering using local LLMs to query relevant articles, entities, and mechanistic relationships based on patient profiles.

Clustering & Stratification (modules/clustering/): Utilizes GPU-accelerated K-Means clustering to identify patient phenotypes based on literature-derived evidence.

Predictive Reasoning (modules/prediction_work/): Integrates multi-source evidence (e.g., population-specific models vs. general models) to perform de novo risk assessment.

Explainable AI (interpretable analysis/): Provides feature importance analysis (Integrated Gradients, SHAP) to ensure the clinical reasoning process is transparent and trustworthy.

🛠️ Configuration
Before deployment, please ensure you update the configuration sections in the respective scripts (or  sidebar) with:webui.py

API Keys: Configure your Google GenAI or OpenAI/Local LLM API keys.

Paths: Update the  variables in the specific module or the WebUI settings to point to your local  and  directories.PATHdata/output/

🤝 Citation
If you use this framework in your research, please cite the following:

代码段
@software{BioKG_Clinical_Risk_2026,
  author = {Your Name/Organization},
  title = {BioKG-Clinical-Risk-Analysis: A Modular Framework for Biomedical Knowledge Graph Construction},
  year = {2026},
  url = {https://github.com/your-username/repo-name}
}
📄 License
This project is licensed under the MIT License - see the  file for details.LICENSE
