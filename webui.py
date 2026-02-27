import streamlit as st
import json
import pandas as pd
from google import genai
from google.genai import types
import time
import io
import csv
# --- 新增 Imports ---
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

# --- 页面配置 ---
st.set_page_config(
    page_title="Biomedical Knowledge Graph Tool",
    page_icon="🧬",
    layout="wide"
)


# --- 辅助函数: 清理 JSON/CSV ---
def clean_llm_response(text):
    """清理 LLM 返回的 Markdown 格式"""
    text = text.strip()
    # 移除 json 代码块标记
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```csv"):
        text = text[6:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# --- 侧边栏: 全局配置 ---
st.sidebar.title("🚀 Mode Selection")
app_mode = st.sidebar.radio("Select Function Module", ["🛠️ Knowledge Graph Construction Tool", "🔍 Clinical Risk Retrieval System", "📊 Final Clinical Scoring System"])

st.sidebar.markdown("---")
if app_mode == "🛠️ Knowledge Graph Construction Tool":
    st.sidebar.title("🔧 Build configuration (Gemini)")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="input Google AI Studio API Key")
    model_name = st.sidebar.text_input("Model Name", value="gemini-3-preview",
                                       help="For example: gemini-3-flash-preview 或 gemini-1.5-pro")
    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            st.sidebar.success("Google The client is ready.")
        except Exception as e:
            st.sidebar.error(f"Client initialization failed: {e}")
    # --- 提示词模版 ---

    # 1. 提取提示词
    DEFAULT_EXTRACTION_PROMPT = '''
    "You are an expert biomedical AI assistant specialized in extracting structured information from medical research literature. Your primary task is to carefully read the provided PDF document (full text) and identify and extract knowledge pertinent or contributing to **predicting the risk of Chronic Kidney Disease (CKD) in individuals with Non-alcoholic Fatty Liver Disease (NAFLD), Metabolic dysfunction-Associated Fatty Liver Disease (MAFLD), or Metabolic dysfunction-Associated Steatotic Liver Disease (MASLD)**. Focus on extracting: * Risk factors for CKD in the context of NAFLD/MAFLD/MASLD. * Methods, models, or specific calculations used for CKD risk prediction in this population. * Biological mechanisms or pathways that are implicated in the progression from NAFLD/MAFLD/MASLD to CKD and could inform risk. * Interventions that might modify this risk. * Relevant study context, population characteristics, and specific outcomes. All extracted information must be directly supported by the provided text. The output MUST be a single, valid JSON object."


    "Please provide the output in a single, valid JSON object with the following top-level keys: 'document_metadata', 'study_details', 'population_characteristics', 'extracted_entities', and 'extracted_relationships'. If specific information for an attribute is not found in the text, use null, an empty string \"\", or an empty list [] as appropriate for that attribute's value. For all descriptive fields, provide concise summaries based on the paper's content."




    --- JSON OUTPUT SCHEMA ---
    {
      "document_metadata": {
        "paper_title": "Extract the full title from the document, e.g., 'A Novel Predictive Model for CKD in NAFLD Patients.', 'Genetic Factors in MASLD-Associated CKD', etc.",
        "pubmed_id": "Extract PubMed ID if available, e.g., 'PMID:12345678', or null",
        "publication_year": "Extract the year of publication as an integer, e.g., 2023, 2021, etc.",
        "journal_name": "Extract the name of the journal, e.g., 'Journal of Hepatology', 'Kidney International', etc."
      },
      "main_study_context": {
        "main_study_internal_id": "Generate a unique ID for this main context, e.g., 'MAIN_STUDY_CTX_PAPER_XYZ_01'",
        "study_type_primary": "Categorize the main study type, e.g., 'Prospective Cohort Study', 'Randomized Controlled Trial (RCT)', 'Case-Control Study', 'Meta-analysis', 'Systematic Review', 'Basic Experimental Research (in vitro/animal model)', 'Bioinformatic Analysis of Public Data', 'Clinical Guideline', etc.",
        "study_design_specifics": "Provide specific details of the main study design, e.g., 'Multicenter, prospective, observational cohort with 5-year follow-up', 'Double-blind, placebo-controlled RCT, phase 3', 'Mendelian Randomization study using UK Biobank GWAS data', etc.",
        "data_sources_description": "Describe the origin of the data for the main study, e.g., 'Data from 3 tertiary care hospital electronic health records (2010-2020)', 'National Health and Nutrition Examination Survey (NHANES) cycles 2005-2016', 'GEO dataset GSEXXXX for liver transcriptomics', etc.",
        "primary_objective_summary": "Summarize the study's main objective, especially concerning NAFLD/MAFLD/MASLD and CKD risk, e.g., 'To identify risk factors for incident CKD stage 3b in a large cohort of MASLD patients', 'To develop and validate a machine learning model for predicting rapid eGFR decline in NAFLD', etc."
      },
      "main_population_characteristics": {
        "main_population_internal_id": "Generate a unique ID for this main population, e.g., 'MAIN_POP_CTX_PAPER_XYZ_01'",
        "population_description_general": "Overall description, e.g., 'Adult patients aged 18-75 with imaging-diagnosed NAFLD and no prior CKD', 'Participants from the general population screened for MAFLD', etc.",
        "sample_size_details": "Total sample size and group sizes if applicable, e.g., 'Total N=2500; 1200 with progressive NAFLD, 1300 with stable NAFLD', etc.",
        "age_profile_summary": "e.g., 'Mean age 55.2 ± 8.1 years', 'Age range 30-70 years', etc.",
        "sex_distribution_summary": "e.g., 'Males 62%', '480 females (40%) and 720 males (60%)', etc.",
        "ethnicity_summary": "e.g., '70% Caucasian, 15% African American, 10% Hispanic, 5% Asian', etc.",
        "nafld_mafld_masld_diagnostic_criteria_used": "Criteria used, e.g., 'MAFLD diagnosed based on EASL-EASD-EASO 2020 criteria', 'NAFLD by abdominal ultrasound and exclusion of other liver diseases', 'MASLD defined by presence of steatosis plus at least one cardiometabolic risk factor', etc.",
        "baseline_liver_disease_severity": "Details on liver disease severity, e.g., '75% had simple steatosis, 25% had NASH based on non-invasive scores', 'Mean CAP score 310 dB/m', 'Predominantly F0-F2 fibrosis (80%) based on elastography', etc.",
        "baseline_ckd_definition_and_status": "e.g., 'CKD defined as eGFR < 60 mL/min/1.73m2 or UACR >= 30 mg/g; 12% had baseline CKD stage 1 or 2', 'Exclusion of individuals with baseline eGFR < 30', etc.",
        "key_baseline_comorbidities": [
          {"comorbidity": "e.g., 'Type 2 Diabetes Mellitus'", "prevalence_or_mean_value": "e.g., 'Present in 40% of participants', 'Mean HbA1c 7.8%', etc."},
          {"comorbidity": "e.g., 'Hypertension'", "prevalence_or_mean_value": "e.g., 'Diagnosed in 58%', 'Mean SBP 138 mmHg', etc."}
          // ... add more common comorbidities like Obesity, Dyslipidemia, etc.
        ],
        "follow_up_duration_for_main_study_outcomes": "e.g., 'Median follow-up of 6.5 years (IQR 4.2-8.1 years)', etc."
      },
      "specific_contexts_defined_in_paper": [ // List for distinct sub-studies, sub-group analyses, or specific experimental conditions within the paper
        {
          "context_internal_id": "Generate a unique ID for this specific context, e.g., 'CTX_SUBGROUP_DIABETIC_01'",
          "context_type": "Defines the nature of this context, e.g., 'SubgroupAnalysis', 'ExperimentalArm', 'AnimalModel', 'InVitroCondition', 'SensitivityAnalysis', 'BioinformaticValidationDataset', 'DifferentFollowUpPeriod', etc.",
          "context_description": "Detailed description of this specific context, e.g., 'Subgroup of NAFLD patients with co-existing Type 2 Diabetes at baseline, analyzed separately for CKD incidence.', 'C57BL/6 mice fed a high-fat, high-fructose diet for 16 weeks to model NASH-CKD progression.', 'Human podocytes exposed to high glucose and palmitate for 48 hours.', etc.",
          "sample_size_or_experimental_units_in_context": "e.g., 'N=450 in this subgroup', '10 animals per experimental group', 'Experiments performed in triplicate wells', etc.",
          "key_defining_characteristics_or_interventions_in_this_context": ["List of defining features or treatments specific to this context, e.g., 'All participants in this context had baseline HbA1c > 7.0%', 'Received DrugX at 10mg/kg daily while controls received vehicle', 'Analysis performed only on subjects with >3 years of follow-up', etc."],
          "methodology_specific_to_this_context": "If different from main study methodology, describe analytical methods, variable definitions, or assumptions unique to this context, e.g., 'A modified CKD definition was used for this subgroup (eGFR decline >25%)', 'Statistical analysis in this subgroup used propensity score matching', etc."
        }
        // ... more specific contexts if applicable
      ],
      "extracted_entities": [ // List of unique entities identified in the paper
        {
          "entity_name_as_in_text": "The exact name or phrase from the text, e.g., 'Body Mass Index', 'PNPLA3 rs738409 G-allele', 'Advanced Fibrosis (F3-F4)', etc.",
          "entity_type": "RiskFactor | OutcomeEvent_CKD | RiskPanel_CKD_PredictionModel | Intervention | GeneticMarker | BiologicalMechanism_Pathways | MolecularSignature_Biomarker | ProtectiveFactor | ImagingBiomarker | CirculatingBiomarker",
          "general_description_from_text": "A general definition or description of the entity as discussed in the paper, if available, independent of specific contexts, e.g., 'Hypertension was defined as systolic blood pressure >= 140 mmHg or diastolic blood pressure >= 90 mmHg, or use of antihypertensive medication.', etc.",
          "contextual_attributes_and_values": [ // List to hold context-specific attributes/values for this entity
            {
              "applies_to_context_id": "ID from 'specific_contexts_defined_in_paper' OR 'main_study_internal_id' OR 'main_population_internal_id'. This indicates the context in which the following attributes/values for this entity were reported.",
              "description_in_this_context": "How this entity is specifically described or relevant within THIS context, e.g., 'In the diabetic subgroup, mean BMI was significantly higher compared to non-diabetics.', etc.",
              "attributes_in_this_context": { // Attributes specific to this entity type, potentially with values observed in THIS context
                // --- RiskFactor Attributes ---
                // if entity_type is "RiskFactor":
                "risk_factor_for_ckd_in_nafld_context_here": "true/false (is this factor specifically discussed for CKD risk in THIS context?)",
                "category": "e.g., Demographic (Age, Sex), Lifestyle (Diet, Smoking), Anthropometric (BMI), ClinicalComorbidity (Diabetes), BloodBiochemistry_Kidney (eGFR, UACR), etc.",
                "specific_measurement_or_criteria_in_this_context": "e.g., 'Elevated ALT defined as > 1.5x ULN in this sub-analysis', 'UACR measured from spot urine sample', etc.",
                "observed_value_or_prevalence_in_this_context": "e.g., 'Mean baseline eGFR was 85 ± 15 mL/min/1.73m2 in the intervention arm', 'Prevalence of metabolic syndrome in the F3-F4 fibrosis group was 85%', etc."
                // ... other entity-type specific attributes as detailed in previous prompt, now framed as potentially context-specific
              },
              "supporting_evidence_quotes_for_this_contextual_attribute": ["e.g., 'The mean age of patients in the validation cohort was 62.3 years.'", "etc."]
            }
            // ... more contextual_attributes_and_values if the entity is discussed with different specifics in different contexts within the paper
          ],
          "overall_supporting_evidence_quotes_for_entity": ["General quotes supporting the entity's definition or importance, e.g., 'PNPLA3 I148M is a well-established genetic risk factor for NAFLD progression.'", "etc."]
        }
        // ... more entities
      ],
      "extracted_relationships": [ // List of relationships identified
        {
          "relationship_internal_id": "Generate a unique textual ID for this relationship instance, e.g., 'Diabetes_Predicts_CKD_in_NAFLD_SubgroupA_01'",
          "source_entity_text": "entity_name_as_in_text of the source entity (from 'extracted_entities' list)",
          "target_entity_text": "entity_name_as_in_text of the target entity (from 'extracted_entities' list)",
          "relationship_type_semantic_description": "Describe the nature of the relationship semantically, e.g., 'RiskFactor_Predicts_OutcomeCKD', 'GeneticMarker_AssociatedWith_IncreasedCKDRisk_In_MASLD_Population', 'Intervention_Reduces_ProgressionTo_OutcomeCKD', 'BiologicalMechanism_Mediates_EffectOf_RiskFactor_On_OutcomeCKD', 'InputFeature_ContributesTo_RiskPanelCKDPredictionModel_Performance', etc.",
          "detailed_description_from_text_of_relationship": "Sentence(s) from the text detailing this specific relationship and its context for NAFLD/MAFLD/MASLD and CKD risk, e.g., 'The study found that in NAFLD patients with existing type 2 diabetes, each 1% increase in HbA1c was associated with a higher likelihood of CKD progression over 5 years.', etc.",
          "context_for_this_relationship": { // **Embedded context specific to THIS reported relationship**
            "applies_to_context_id": "ID from 'specific_contexts_defined_in_paper' OR 'main_study_internal_id' OR 'main_population_internal_id'. This links to the pre-defined context in which this relationship and its attributes were specifically observed and reported.",
            "context_summary_for_relationship_relevance": "A brief summary explaining why the linked context (from applies_to_context_id) is relevant for interpreting THIS specific relationship finding, e.g., 'This hazard ratio was specifically calculated for the subgroup of patients older than 60 years.', 'The p-value reported here is from the analysis of the animal model treated with Drug Y.', 'This odds ratio pertains to the main cohort after adjustment for baseline renal function.', etc.",
            "key_prerequisites_or_conditions_for_this_finding": [ // List of critical conditions under which this specific finding is valid
                "e.g., 'Analysis adjusted for age, sex, and BMI.'",
                "e.g., 'Finding specific to patients with baseline eGFR > 60 and UACR < 300 mg/g.'",
                "e.g., 'Relationship observed only in male participants.'",
                "e.g., 'Statistical significance maintained after Bonferroni correction.'",
                "e.g., 'Result from per-protocol analysis of the RCT.'",
                "e.g., 'Based on a median follow-up of X years.'",
                "e.g., 'Specific definition of CKD progression used for this analysis was Y.'",
                "etc."
            ],
            "sample_size_for_this_specific_analysis_if_different": "If this relationship was derived from an analysis with a sample size different from the linked context's general N, specify here, e.g., 'N=150 for this particular correlation analysis due to missing biomarker data', etc., or null"
          },
          "attributes_of_the_relationship": { // Attributes of the relationship itself, observed within the specified context
            "association_direction_or_nature_of_effect": "e.g., 'IncreasesRiskOf', 'DecreasesRiskOf', 'PredictsHigherValueFor', 'AssociatedWithPresenceOf (Positive)', 'AssociatedWithAbsenceOf (Negative)', 'NoSignificantAssociationFound', 'ModulatesActivity (e.g., Upregulates/Downregulates)', 'IsAComponentOf', 'ContributesCausallyTo (if strong evidence like experimental manipulation)', 'CorrelatesPositivelyWith', 'CorrelatesNegativelyWith', etc.",
            "methodology_used_to_establish_this_relationship": "The specific method reported in the paper for this finding, e.g., 'Multivariable Cox Proportional Hazards Model', 'Logistic Regression Analysis', 'Spearman Rank Correlation', 'Student t-test for mean comparison', 'Machine Learning model (e.g., Random Forest) feature importance score (e.g., SHAP value)', 'Experimental validation (e.g., gene knockdown effect on phenotype in cell culture)', 'Pathway Analysis (e.g., GSEA enrichment score)', 'Finding from Systematic Review/Meta-analysis (specify which one if paper is a review)', 'Expert Consensus Statement from Guideline', etc.",
            "effect_size_reported_for_relationship": "Reported effect size with units/type, e.g., 'Hazard Ratio (HR): 2.1', 'Odds Ratio (OR): 3.5 (adjusted)', 'Relative Risk (RR): 1.7 (crude)', 'Beta-coefficient: 0.25 (per 1-SD increase in predictor)', 'Mean Difference: -5.2 mL/min/1.73m2 (eGFR change)', 'Standardized Mean Difference (SMD): 0.6', 'Correlation coefficient (r): 0.45 (Pearson)', 'Percentage change from baseline: -15% (UACR)', 'Fold Change in gene expression: 2.3', 'Area Under Curve (AUC) for prediction: 0.78', etc."(PS: Please distinguish between training set/validation set performance, calibration metrics),
            "confidence_interval_95_for_effect_size": "e.g., '1.5-2.8 for HR', '0.15 to 0.35 for Beta-coefficient', 'Not Reported', etc.",
            "p_value_or_fdr_for_relationship": "e.g., '<0.001', '0.045', 'FDR_q_value = 0.008', 'p_for_trend = 0.02', 'NS (Not Significant)', etc.",
            "statistical_model_specific_details_if_any": "Key details of the model used for this relationship, e.g., 'Cox model adjusted for age, sex, BMI, baseline eGFR, presence of diabetes, and use of ACE inhibitors', 'Logistic regression with Firth correction for rare events', 'Interaction term between [FactorA] and [FactorB] was included (p_interaction=X)', etc.",
            "evidence_quality_or_study_design_basis": "Nature of evidence supporting this specific relationship, e.g., 'Derived from primary analysis of prospective cohort data', 'Finding from a pre-specified secondary endpoint of an RCT', 'Result from an in-vitro experiment on human kidney cells', 'Prediction from a validated bioinformatic tool', 'Consistent finding across multiple sensitivity analyses', etc.",
            "temporal_or_dose_response_information": "If relevant for this relationship, e.g., 'Effect most pronounced after 3 years of exposure', 'Dose-dependent increase in risk observed for [DrugX]', 'Relationship stronger with longer duration of NAFLD', etc."
          },
          "natural_language_summary_of_finding_by_llm": "A concise, one-sentence summary of what this specific relationship finding means, considering its context, e.g., 'The study reports that in NAFLD patients with established diabetes (specific context), higher baseline HbA1c (source) is significantly associated with an increased hazard of developing advanced CKD (target) over a median of 7 years, even after adjusting for multiple confounders.'",
          "supporting_evidence_quotes_for_relationship": ["Direct quote(s) from the paper supporting this relationship, its quantification, and its context, e.g., 'In the multivariate Cox regression model adjusted for [confounders], each 1% increase in HbA1c was associated with an adjusted HR of 1.5 (95% CI, 1.2-1.9; p=0.002) for incident CKD stage 3b or higher within the diabetic NAFLD subgroup.'", "etc."]
        }
        // ... more relationships
      ],
      "Systemic Interactions" : "You are a biomedical AI assistant specializing in synthesizing complex interactions from research papers. You have already received a basic, structured extraction of entities and their direct relationships from the provided document. Your current task is to re-read the document and supplement the initial extraction by specifically identifying and structuring the interactive, dynamic, and systemic mechanisms.

    Do NOT re-extract the basic entities or simple one-to-one relationships already captured. This task is focused on a higher level of synthesis. Focus on following areas:

        How do certain conditions or factors modulate biological processes or interactions? How do pathological states (like obesity or metabolic stress) or experimental interventions change the system's behavior? Does a single key factor exert its influence through multiple, distinct mechanisms? Describe the system-level crosstalk or communication between different organs, tissues, or cellular systems. What are the key factors and what are the consequences of this communication for overall outcomes?
    "
    }
    --- END JSON OUTPUT SCHEMA ---

    Final Reminders for the LLM:
    1.  The **central goal** is extracting information relevant to **CKD risk prediction in NAFLD/MAFLD/MASLD**.
    2.  For each `extracted_relationship`, the `context_for_this_relationship` block is critical. It must detail the specific conditions (study design, population subgroup, analysis prerequisites) under which the reported relationship attributes (like effect size, p-value) are valid, linking to a context defined in `specific_contexts_defined_in_paper` or the main study/population context.
    3.  For `extracted_entities`, use the `contextual_attributes_and_values` list to capture how an entity's specific attributes or measured values are reported under different contexts within the paper.
    4.  Prioritize extracting quantitative data (effect sizes, performance metrics, p-values, specific measurements, cutoffs) and precise definitions.
    5.  Ensure all information is directly traceable to the source text via `supporting_evidence_quotes`.
    6.  Adhere strictly to the provided JSON structure, using `null`, `""`, or `[]` for missing information.
    '''

    # 2. 标准化提示词 (你提供的内容)
    DEFAULT_NORMALIZATION_PROMPT = '''
    System Role: You are an expert Medical Data Governance Specialist. You possess deep knowledge of medical terminology, clinical concepts, abbreviations, and multi-language translations (specifically English, Chinese, Spanish, and Russian).
    Task Description: I have a raw list of medical entity names (Original Entity Name) that contains duplicates, synonyms, abbreviations, and terms in various languages. Your task is to normalize this list by clustering semantically identical terms and assigning them a single, standardized "Canonical Name" and a unique "Entity ID".

    Step-by-Step Instructions:
    1. Concept Grouping (Deduplication): Analyze the semantics of each Original Entity Name. Group together all terms that refer to the exact same medical concept.
       - Synonyms: (e.g., "Renal failure" = "Kidney failure")
       - Abbreviations: (e.g., "CKD" = "Chronic Kidney Disease")
       - Multilingual Matches: (e.g., "慢性肾脏病" = "Chronic Kidney Disease")
       - Spelling Variations: Correct minor typos.
    2. Canonical Name Selection (Mapping Logic): For each identified concept group, select ONE representative name to be the Mapped Value.
       - Priority: High (Mixed English/Chinese or comprehensive definitions) > Medium (Formal English academic terms) > Low (Abbreviations).
    3. ID Assignment: Assign a unique integer Entity ID to each unique concept group.
    4. Output Format: Output the result as a STRICT CSV table (comma-separated values) with headers: "Original Entity Name", "Entity ID", "Mapped Value". Do not include any other text.

    Constraint: Do not leave any entity unmapped.

    Input List of Entities:
    {entity_list}
    '''

    # 3. 摘要提示词
    DEFAULT_SUMMARY_PROMPT = '''
    You are a biomedical knowledge synthesis expert. Your primary task is to create a comprehensive, consolidated summary for a single biomedical entity based on the information provided below. The summary must naturally reflect the primary context of the source evidence, which is **predicting the risk of CKD in individuals with NAFLD, MAFLD, or MASLD**.

    **Your instructions:**
    1. Analyze all the provided `evidence_snippets`.
    2. Synthesize this information into a single, comprehensive, and coherent summary.
    3. If descriptions are contradictory, resolve them or note the discrepancy.
    4. Use a neutral, third-person scientific tone.
    5. Generate a JSON output that strictly follows the `REQUIRED OUTPUT JSON SCHEMA`.

    --- REQUIRED OUTPUT JSON SCHEMA ---
    {
      "synonyms": ["List synonyms"],
      "consolidated_description": "Narrative paragraph synthesizing definition and role.",
      "key_attributes_and_roles_summary": ["List of key roles", "Common definitions", "Interactions"],
      "prevalence_in_nafld_ckd_context": "Summary of prevalence or null.",
      "evidence_source_count": "Integer count of unique papers.",
      "paper_titles": ["List of paper titles"]
    }
    '''

    # --- 主界面 ---
    st.title("🧬 Biomedical Knowledge Graph Construction Tool")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 1. PDF Entity extraction", "🔄 2. Entity Standardization (Cleaning)", "📝 3. Comprehensive Summary of Entity", "⚖️ 4. Relationship Score Export"])

    # ==========================================
    # TAB 1: 实体提取 (Entity Extraction)
    # ==========================================
    with tab1:
        st.header("Knowledge extraction from PDF documents")

        with st.expander("📝 CUSTOM EXTRACTION PROMPT", expanded=False):
            extraction_prompt = st.text_area("Extraction Prompt", value=DEFAULT_EXTRACTION_PROMPT, height=200)

        uploaded_files = st.file_uploader("Upload PDF document", type=["pdf"], accept_multiple_files=True)

        if st.button("Beginning to parse the document", type="primary", disabled=(not client or not uploaded_files)):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_part = types.Part.from_bytes(mime_type='application/pdf', data=pdf_bytes)

                    response = client.models.generate_content(
                        model=model_name,
                        contents=[pdf_part, extraction_prompt]
                    )

                    json_text = clean_llm_response(response.text)
                    parsed_json = json.loads(json_text)

                    if 'document_metadata' not in parsed_json:
                        parsed_json['document_metadata'] = {}
                    parsed_json['document_metadata']['source_filename'] = uploaded_file.name

                    results.append(parsed_json)
                    st.success(f"✅ {uploaded_file.name} Parsing successful")

                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} Processing failed: {e}")

                progress_bar.progress((idx + 1) / len(uploaded_files))
                time.sleep(1)

            if results:
                st.success("All parsing completed.！")
                json_output = json.dumps(results, indent=4, ensure_ascii=False)
                st.download_button("📥 Download original extraction results (JSON)", data=json_output,
                                   file_name="raw_extraction_results.json",
                                   mime="application/json")

    # ==========================================
    # TAB 2: 实体标准化 (Entity Normalization)
    # ==========================================
    with tab2:
        st.header("Entity Name Standardization and Normalization")
        st.info("This step merges synonyms (e.g., CKD = Chronic Kidney Disease) to produce higher-quality data.")

        raw_json_file = st.file_uploader("Upload the original extraction results (JSON)", type=["json"], key="upload_raw_json")

        if raw_json_file:
            raw_data = json.load(raw_json_file)

            # 1. 收集所有实体名称
            all_raw_entities = set()
            for doc in raw_data:
                # 收集 extracted_entities 中的名称
                for ent in doc.get("extracted_entities", []):
                    name = ent.get("entity_name_as_in_text")
                    if name: all_raw_entities.add(name)
                # 收集 extracted_relationships 中的名称
                for rel in doc.get("extracted_relationships", []):
                    s_name = rel.get("source_entity_text")
                    t_name = rel.get("target_entity_text")
                    if s_name: all_raw_entities.add(s_name)
                    if t_name: all_raw_entities.add(t_name)

            # 排序以保证批处理的一致性
            sorted_raw_entities = sorted(list(all_raw_entities))
            st.write(f"📊 A total of {len(sorted_raw_entities)} unique raw entity names were found.")

            with st.expander("📝 View all entities pending standardization"):
                st.write(sorted_raw_entities)

            if st.button("Initiating standardized mapping", type="primary", disabled=not client):
                # 2. 分批处理 (每批 200 个)
                BATCH_SIZE = 200
                batches = [sorted_raw_entities[i:i + BATCH_SIZE] for i in
                           range(0, len(sorted_raw_entities), BATCH_SIZE)]

                mapping_dict = {}  # {original_name: mapped_name}
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    for i, batch in enumerate(batches):
                        status_text.text(f"Normalizing batch {i + 1}/{len(batches)} of entities ({BATCH_SIZE} per batch)...")

                        # 构造 Prompt
                        batch_str = "\n".join(batch)
                        prompt = DEFAULT_NORMALIZATION_PROMPT.replace("{entity_list}", batch_str)

                        # 调用 LLM
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[prompt]
                        )

                        # 解析 CSV 响应
                        csv_text = clean_llm_response(response.text)

                        # 使用 csv 模块解析字符串
                        f = io.StringIO(csv_text)
                        reader = csv.DictReader(f)

                        for row in reader:
                            # 健壮性检查：确保列名大致匹配（防止LLM输出不规范）
                            # 尝试找到对应的 key
                            orig_key = next((k for k in row.keys() if "Original" in k), None)
                            mapped_key = next((k for k in row.keys() if "Mapped" in k), None)

                            if orig_key and mapped_key:
                                orig_val = row[orig_key].strip()
                                mapped_val = row[mapped_key].strip()
                                mapping_dict[orig_val] = mapped_val

                        progress_bar.progress((i + 1) / len(batches))
                        time.sleep(1)  # 避免限流

                    st.success("Standardized mapping table generation completed! Replacing original JSON now....")

                    # 3. 替换原始 JSON 中的实体
                    normalized_data = raw_data  # 浅拷贝，直接修改
                    replacement_count = 0

                    for doc in normalized_data:
                        # 替换实体列表
                        for ent in doc.get("extracted_entities", []):
                            orig = ent.get("entity_name_as_in_text")
                            if orig and orig in mapping_dict:
                                ent["entity_name_as_in_text"] = mapping_dict[orig]
                                # 可选：保留原始名称作为同义词字段
                                if "original_raw_name" not in ent:
                                    ent["original_raw_name"] = orig
                                replacement_count += 1

                        # 替换关系列表
                        for rel in doc.get("extracted_relationships", []):
                            s_orig = rel.get("source_entity_text")
                            t_orig = rel.get("target_entity_text")

                            if s_orig and s_orig in mapping_dict:
                                rel["source_entity_text"] = mapping_dict[s_orig]
                                replacement_count += 1
                            if t_orig and t_orig in mapping_dict:
                                rel["target_entity_text"] = mapping_dict[t_orig]
                                replacement_count += 1

                    st.write(f"✅Entity names have been replaced in {replacement_count} instances.")

                    # 4. 展示映射结果预览
                    df_mapping = pd.DataFrame(list(mapping_dict.items()), columns=["original_name", "standardized_name"])
                    st.dataframe(df_mapping.head(10))

                    # 5. 下载结果
                    norm_json_output = json.dumps(normalized_data, indent=4, ensure_ascii=False)
                    st.download_button(
                        "📥 Download the normalized JSON file (normalized_extraction.json)",
                        data=norm_json_output,
                        file_name="normalized_extraction.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"An error occurred during standardization{e}")
                    st.write("Raw text segment returned by the LLM (for debugging):")
                    st.text(csv_text[:500] if 'csv_text' in locals() else "N/A")

    # ==========================================
    # TAB 3: 实体摘要 (Entity Summarization)
    # ==========================================
    with tab3:
        st.header("Multi-document entity-centric summarization")
        st.info("Please use the **normalized_extraction.json** file generated in step 2 for best results.")

        norm_json_file = st.file_uploader("Upload JSON file", type=["json"], key="upload_norm_json")

        with st.expander("📝 Custom Summary Prompt", expanded=False):
            summary_prompt_template = st.text_area("Summarization Prompt", value=DEFAULT_SUMMARY_PROMPT, height=200)

        if norm_json_file and client:
            input_data = json.load(norm_json_file)

            # 自动提取唯一实体
            all_entities = set()
            entity_type_map = {}
            for doc in input_data:
                for ent in doc.get("extracted_entities", []):
                    name = ent.get("entity_name_as_in_text")
                    if name:
                        all_entities.add(name)
                        if name not in entity_type_map and ent.get("entity_type"):
                            entity_type_map[name] = ent.get("entity_type")

            sorted_entities = sorted(list(all_entities))
            st.write(f"📊 {len(sorted_entities)} unique entities (after normalization) were found.")

            # 全选/多选逻辑
            col1, col2 = st.columns([0.2, 0.8])
            with col1:
                select_all = st.checkbox("Select all entities")

            if select_all:
                target_entities = sorted_entities
                with col2:
                    st.info(f"""
All {len(target_entities)} entities have been selected
""")
            else:
                with col2:
                    target_entities = st.multiselect("Select entity", options=sorted_entities)

            if st.button("Begin generating summary", type="primary", disabled=len(target_entities) == 0):
                summary_results = {}
                prog = st.progress(0)
                status = st.empty()

                for idx, entity_name in enumerate(target_entities):
                    status.text(f"({idx + 1}/{len(target_entities)}) processing: {entity_name}...")

                    # 搜集证据
                    evidence = []
                    for doc in input_data:
                        title = doc.get('document_metadata', {}).get('paper_title', 'Unknown')

                        # 匹配实体
                        for ent in doc.get('extracted_entities', []):
                            if entity_name == ent.get('entity_name_as_in_text'):
                                evidence.append({
                                    "source": title,
                                    "desc": ent.get('general_description_from_text'),
                                    "quote": (ent.get('overall_supporting_evidence_quotes_for_entity') or [""])[0]
                                })
                        # 匹配关系
                        for rel in doc.get('extracted_relationships', []):
                            if entity_name in [rel.get('source_entity_text'), rel.get('target_entity_text')]:
                                evidence.append({
                                    "source": title,
                                    "desc": rel.get('detailed_description_from_text_of_relationship'),
                                    "quote": (rel.get('supporting_evidence_quotes_for_relationship') or [""])[0]
                                })

                    if not evidence: continue

                    # 调用 API
                    try:
                        prompt_data = {
                            "entity": entity_name,
                            "type": entity_type_map.get(entity_name, "Unknown"),
                            "evidence": evidence
                        }
                        prompt = f"{summary_prompt_template}\n\nDATA:\n{json.dumps(prompt_data, indent=2, ensure_ascii=False)}"

                        resp = client.models.generate_content(model=model_name, contents=[prompt])
                        summary_results[entity_name] = json.loads(clean_llm_response(resp.text))
                    except Exception as e:
                        st.error(f"Error ({entity_name}): {e}")

                    prog.progress((idx + 1) / len(target_entities))
                    time.sleep(0.5)

                status.text("Completed！")

                # 结果展示与下载
                st.json(dict(list(summary_results.items())[:2]), expanded=False)
                st.download_button(
                    "📥 Download comprehensive summary (JSON)",
                    data=json.dumps(summary_results, indent=4, ensure_ascii=False),
                    file_name="final_summaries.json",
                    mime="application/json"
                )

    # ==========================================
    # TAB 4: 关系评分与导出 (Relationship Scoring)
    # ==========================================
    with tab4:
        st.header("⚖️ Relationship Evidence Level Scoring and Export")
        st.markdown(
            """
This module will score each extracted relation by integrating the literature background information with the custom scoring criteria and produce a final CSV report.
""")

        # 1. 上传数据 (通常是标准化后的 JSON)
        score_input_file = st.file_uploader("Upload JSON file (it is recommended to use normalized_extraction.json from Step 2)",
                                            type=["json"],
                                            key="score_json")

        # 2. 定义评分原则 (Prompt Injection)
        default_scoring_criteria = """
    Here is the comprehensive Evidence Quality Scoring Principles document.

    In accordance with your requirements, this version avoids LaTeX rendering and standard Markdown tables. It is formatted to be highly readable in a pure text environment (e.g., Notepad, code comments, or simple text editors) while preserving every detail from your provided text and image.
    EVIDENCE QUALITY SCORING PRINCIPLES
    1. SYSTEM OVERVIEW

    An evidence quality scoring system was established to quantify the reliability of associations. This system integrates the Oxford Centre for Evidence-Based Medicine (2009) levels of evidence with the GRADE (Grading of Recommendations Assessment, Development and Evaluation) approach for adjusting confidence ratings.
    2. CALCULATION METHODOLOGY

    The final confidence weight for each relationship edge is dynamically calculated by adjusting the base study design score with specific quality factors. To ensure validity, the score is constrained to a non-negative range.

    Formula: Final_Weight = MAX(0, Base_Score + Sum_of_Adjustments)

    Definitions:

        Base_Score: The initial evidence score assigned based on the study design hierarchy (see Section 3).

        Sum_of_Adjustments: The total value of all upgrading and downgrading factors identified (see Section 4).

        MAX(0, ...): A rectified linear function ensures the final weight is never negative for low-quality evidence, while allowing high-certainty evidence to exceed the base scale.

    3. BASE EVIDENCE SCORES (INITIAL WEIGHTS)

    Scores are assigned based on the study design and source type hierarchy.

    [Level Ia] - Score: 10 Systematic reviews or meta-analyses of randomized controlled trials (RCTs).

    [Level Ib] - Score: 9 Individual, well-designed randomized controlled trial (RCT).

    [Level IIa] - Score: 8 Systematic reviews or meta-analyses of cohort studies.

    [Level IIb] - Score: 7 Individual, well-designed prospective or retrospective cohort study.

    [Level IIc] - Score: 6 "Outcomes Research."

    [Level IIIa] - Score: 5 Systematic reviews or meta-analyses of case-control studies.

    [Level IIIb] - Score: 5 Individual case-control study; large-scale cross-sectional study.

    [Level IV] - Score: 3 Case series; poor-quality cohort or case-control studies; small-scale cross-sectional studies.

    [Level V] - Score: 2 Expert opinion in reviews, editorials, or consensus; mechanistic or predictive inferences based on basic experiments (animal/cell) or bioinformatics/AI models.

    [N/A] - Score: 0 Background information; unclassifiable citations.
    4. QUALITY ADJUSTMENT FACTORS

    The base score is adjusted based on specific quality criteria extracted from the GRADE framework.
    A. DOWNGRADING FACTORS (Reducing Confidence)

    Applicable primarily to RCTs but relevant to all study types. Each factor reduces the score by 1 point (Serious) or 2 points (Very Serious).

    1. Risk of Bias

        Definition: Limitations in study conduct.

        Criteria:

            Failure to perform correct randomization or allocation concealment.

            Lack of blinding.

            High loss to follow-up (attrition bias) or failure to use intention-to-treat analysis.

            Selective outcome reporting.

            Trial stopped early due to apparent benefit.

        Adjustment: -1 (Serious) or -2 (Very Serious).

    2. Inconsistency

        Definition: Widely differing estimates of the treatment effect across studies.

        Criteria: Unexplained heterogeneity in results. Differences may stem from variations in populations, interventions, or outcome measures.

        Adjustment: -1 (Serious) or -2 (Very Serious).

        Note: If the inconsistency is caused by Risk of Bias, do not double-penalize (score is reduced only once).

    3. Indirectness

        Definition: Evidence does not directly answer the research question.

        Criteria:

            Absence of "Head-to-Head" direct comparisons (e.g., comparing Intervention A vs. Placebo and Intervention B vs. Placebo, rather than A vs. B).

            Major discrepancies in PICO characteristics (Population, Intervention, Comparator, Outcome) between the study and the actual application context.

        Adjustment: -1 (Serious) or -2 (Very Serious).

    4. Imprecision

        Definition: Data is insufficient to provide a confident estimate.

        Criteria: The study includes relatively few patients or observed events, resulting in wide confidence intervals.

        Adjustment: -1 (Serious) or -2 (Very Serious).

    5. Publication Bias

        Definition: High likelihood that negative studies are missing.

        Criteria:

            Many studies (usually small, showing negative results) remain unpublished.

            Evidence is limited to a small number of trials, all of which are industry-sponsored.

        Adjustment: -1 (Likely) or -2 (Very Likely).

    B. UPGRADING FACTORS (Increasing Confidence)

    Applicable primarily to enhance the quality of observational studies. These factors can upgrade evidence quality by 1 or 2 levels.

    1. Large Effect Size

        Definition: Methodologically rigorous observational studies show a significant treatment effect with consistent results.

        Criteria:

            Relative Risk (RR) > 2: Upgrade by 1 level.

            Relative Risk (RR) > 5: Upgrade by 2 levels.

        Adjustment: +1 (Large Effect) or +2 (Very Large Effect).

    2. Dose-Response Gradient

        Definition: A clear correlation exists between the intervention intensity and the outcome.

        Criteria: Evidence shows a clear dose-response relationship (e.g., higher dosage leads to a stronger effect).

        Adjustment: +1 (Present).

    3. Residual Confounding (Negative Bias)

        Definition: Biases present would likely underestimate the true effect.

        Criteria: When plausible biases or confounding factors would either:

            Reduce the demonstrated effect (meaning the true effect is likely stronger).

            Suggest a spurious effect when results actually show no effect.

        Adjustment: +1 (Present).
        """

        with st.expander("⚙️  System Prompt", expanded=True):
            scoring_criteria = st.text_area("Grading criteria and grouping rules", value=default_scoring_criteria, height=200)

        # 3. 开始处理
        if st.button("Start scoring and generate the report.", type="primary", disabled=(not score_input_file or not client)):
            try:
                input_data = json.load(score_input_file)

                # 准备结果容器
                scored_rows = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                total_docs = len(input_data)

                for doc_idx, doc in enumerate(input_data):
                    doc_title = doc.get('document_metadata', {}).get('paper_title', 'Unknown Paper')
                    status_text.text(f"({doc_idx + 1}/{total_docs}) Evaluating document: {doc_title[:50]}...")

                    # --- A. 构建背景上下文 (排除 extracted_entities) ---
                    # 使用深拷贝以免修改原始数据，或者构造一个新的字典
                    context_dict = {k: v for k, v in doc.items() if
                                    k not in ['extracted_entities', 'extracted_relationships']}
                    context_str = json.dumps(context_dict, indent=2, ensure_ascii=False)

                    # --- B. 获取该文档下的所有关系 ---
                    relationships = doc.get('extracted_relationships', [])
                    if not relationships:
                        progress_bar.progress((doc_idx + 1) / total_docs)
                        continue

                    # 为了节省 token，我们只提取评分所需的关键关系字段发送给 LLM
                    rels_to_score = []
                    for rel in relationships:
                        rels_to_score.append({
                            "relationship_internal_id": rel.get("relationship_internal_id"),
                            "source": rel.get("source_entity_text"),
                            "target": rel.get("target_entity_text"),
                            "type": rel.get("relationship_type_semantic_description"),
                            "description": rel.get("detailed_description_from_text_of_relationship"),
                            "attributes": rel.get("attributes_of_the_relationship", {})
                        })

                    rels_str = json.dumps(rels_to_score, indent=2, ensure_ascii=False)

                    # --- C. 构造最终提示词 ---
                    final_scoring_prompt = f"""
                    You are a senior biomedical data curator. Your task is to evaluate the strength of evidence for specific relationships extracted from a scientific paper.

                    --- 1. STUDY CONTEXT (Background Info) ---
                    {context_str}

                    --- 2. RELATIONSHIPS TO SCORE ---
                    {rels_str}

                    --- 3. YOUR INSTRUCTIONS ---
                    Based on the provided **STUDY CONTEXT** and the specific details of each **RELATIONSHIP**, apply the following **SCORING CRITERIA**:

                    {scoring_criteria}

                    --- 4. REQUIRED OUTPUT FORMAT ---
                    You must return a single valid JSON object containing a list called `scored_relationships`.
                    Each item in the list must correspond to one of the input relationships and contain:
                    - `relationship_internal_id`: (Must match the input ID exactly)
                    - `weight`: (Integer, 0-10 based on your scoring)
                    - `GroupName`: (String, a short category tag as requested)

                    Example Output:
                    {{
                        "scored_relationships": [
                            {{ "relationship_internal_id": "REL_01", "weight": 8, "GroupName": "Genetic_Risk" }},
                            {{ "relationship_internal_id": "REL_02", "weight": 3, "GroupName": "General_Comorbidity" }}
                        ]
                    }}

                    Do not output markdown code blocks (```json). Just the raw JSON string.
                    """

                    # --- D. 调用 API ---
                    try:
                        # 考虑到上下文可能较长，使用较新的模型
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[final_scoring_prompt]
                        )

                        llm_output = clean_llm_response(response.text)
                        parsed_output = json.loads(llm_output)

                        # --- E. 合并结果 ---
                        # 创建一个查找字典方便匹配: {id: {weight, group}}
                        score_map = {item['relationship_internal_id']: item for item in
                                     parsed_output.get('scored_relationships', [])}

                        # 遍历原始关系，结合 LLM 的评分生成最终行
                        for rel in relationships:
                            r_id = rel.get("relationship_internal_id")
                            score_info = score_map.get(r_id, {})

                            # 构建 CSV 行数据
                            row = {
                                "source_entity_text": rel.get("source_entity_text"),
                                "target_entity_text": rel.get("target_entity_text"),
                                "relationship_type_semantic_description": rel.get(
                                    "relationship_type_semantic_description"),
                                "GroupName": score_info.get("GroupName", "Uncategorized"),  # LLM 生成
                                "relationship_internal_id": r_id,
                                "weight": score_info.get("weight", 0),  # LLM 生成
                                # 额外保留一些有用的元数据，虽然 CSV 主要展示上面几列
                                "paper_title": doc_title,
                                "p_value": rel.get("attributes_of_the_relationship", {}).get(
                                    "p_value_or_fdr_for_relationship")
                            }
                            scored_rows.append(row)

                    except Exception as e:
                        st.error(f"An error occurred while processing the document {doc_title}: {e}")
                        # 如果出错，也可以选择跳过，或者记录错误

                    progress_bar.progress((doc_idx + 1) / total_docs)
                    time.sleep(1)  # 避免限流

                status_text.text("Grading process completed！")

                # --- 4. 展示与导出 CSV ---
                if scored_rows:
                    df_scores = pd.DataFrame(scored_rows)

                    # 调整列顺序以符合你的要求
                    cols_order = [
                        "source_entity_text",
                        "target_entity_text",
                        "relationship_type_semantic_description",
                        "GroupName",
                        "relationship_internal_id",
                        "weight"
                    ]
                    # 确保其他列也在后面
                    remaining_cols = [c for c in df_scores.columns if c not in cols_order]
                    final_df = df_scores[cols_order + remaining_cols]

                    st.subheader("📊 Preview of Rating Results")
                    st.dataframe(final_df.head(10))

                    # 转换为 CSV
                    csv_buffer = final_df.to_csv(index=False).encode('utf-8-sig')  # sig for Chinese support

                    st.download_button(
                        label="📥 Export the scoring sheet (CSV)",
                        data=csv_buffer,
                        file_name="relationship_scores_weighted.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No rating results could be generated; please check the JSON structure or the LLM response.")

            except Exception as e:
                st.error(f"A global error has occurred.: {e}")



# ==============================================================================
# 模式 B: 临床风险检索系统
# ==============================================================================
elif app_mode == "🔍 Clinical Risk Retrieval System":
    st.title("🔍 Clinical Risk Retrieval and Literature Scoring System")

    # --- 1. 原始 Prompt 模版 (预置，为了在文本框显示正常的JSON，已去掉f-string的双大括号转义) ---
    # 实体筛选 Prompt
    DEFAULT_ENTITY_PROMPT = """You are an expert in biomedical knowledge graphs and clinical data analysis. Your task is to select a curated list of exactly {num_to_select} entities from the "Candidate Entity List" based on the provided "Cluster Characteristics".

    **IMPORTANT CONTEXT ON INPUT DATA:**
    The section labeled "Patient Profile" below actually contains **aggregated statistical data** for a specific cluster of patients (e.g., means, standard deviations, phenotypic distributions, and prevalence rates). You must treat this as a **cohort description**, not a single individual's record.

    Your selection process must intelligently balance the following three core principles:

    1.  **High Relevance:** Prioritize entities that are most relevant to the **aggregate clinical features and dominant phenotypes** described in the "Patient Profile" section. Focus on the statistical patterns (e.g., abnormal means, high prevalence of specific conditions) distinct to this cluster. Although you will not output the scores, this relevance assessment is the primary basis for your selection.
    2.  **Comprehensive Coverage:** A key objective is to maximize the coverage of different `Semantic Group` and `Functional Group` categories in your final list. For instance, if entities A and B are both highly relevant and in the same group, but entity C is slightly less relevant but belongs to a different group, you should prefer selecting A and C over A and B. This ensures the final selection has a broader medical and biological scope.
    3.  **Evidence Value:** Consider the `'evidence_source_count'`. A higher count indicates the entity is mentioned more frequently in literature, which adds to its value and should be an important factor in your selection.

    **How to Identify the Entity Name:**
    The `Candidate Entity List` is a JSON object where each **top-level key** IS the official entity name you must return. The value associated with each key contains descriptive details but is not the name itself.

    For example, given this structure:
    ```json
    {
        "Hyperglycaemia": {
            "consolidated_description": "Hyperglycemia, defined as fasting plasma glucose...",
            "evidence_source_count": 3,
            "Semantic Group": "Metabolic Dysfunction States",
            "Functional Group": "Metabolic Risk Factors"
        },
        "Genetic polymorphisms (PNPLA3, HSD17B13, TM6SF2, MBOAT7, GCKR)": { ... }
    }
    ```

    The exact, original entity name to use for the first entry is the key itself: "Hyperglycaemia". For the second, it is "Genetic polymorphisms (PNPLA3, HSD17B13, TM6SF2, MBOAT7, GCKR)".

    Crucial Instruction: The entity names in your output list MUST EXACTLY MATCH these top-level keys from the 'Candidate Entity List'. Do not alter, abbreviate, rephrase, or derive names from the descriptions. Every character, including parentheses, commas, and casing, must be identical.

    Your output MUST be a single, valid JSON object containing only the key "selected_entities". Do not include any text, notes, or explanations outside of the JSON object.

    {
    "selected_entities": [
    "Entity Name A",
    "Entity Name C",
    ...
    ]
    }

    The selected_entities list must contain EXACTLY {num_to_select} entity names.

    Patient Profile (Cluster Statistics):

    {patient_data_str}

    Candidate Entity List (in JSON format):

    {entities_prompt_str}
    """

    # 关系筛选 Prompt
    DEFAULT_RELATIONSHIP_PROMPT = """You are an expert in biomedical knowledge graphs and clinical data analysis. Your task is to select a curated list of exactly {num_to_select} relationships from the "Candidate Relationship List" based on the provided "Cluster Characteristics".

    **IMPORTANT CONTEXT ON INPUT DATA:**
    The section labeled "Patient Profile" below actually contains **aggregated statistical data** for a specific cluster of patients (e.g., means, standard deviations, phenotypic distributions, and prevalence rates). You must treat this as a **cohort description**, not a single individual's record.

    Your selection process must intelligently balance the following three core principles:

    1.  **High Relevance:** Prioritize relationships that are most relevant to the **aggregate clinical features and dominant phenotypes** described in the "Patient Profile" section. Focus on the statistical patterns (e.g., abnormal means, high prevalence of specific conditions) distinct to this cluster. This relevance assessment is the primary basis for your selection.
    2.  **Comprehensive Coverage:** A key objective is to maximize the coverage of different `GroupName` categories in your final list. For instance, if relationship A and B are both highly relevant and in the same group, but relationship C is slightly less relevant but belongs to a different group, you should prefer selecting A and C over A and B. This ensures the final selection has a broader medical and biological scope.
    3.  **Relationship Weight:** Consider the `'weight'`. A higher weight indicates the relationship is more significant or supported by stronger evidence, which adds to its value and should be an important factor in your selection.

    **How to Identify the Relationship:**
    The `Candidate Relationship List` is a JSON object where each **top-level key** IS the official `relationship_internal_id` you must return. The value associated with each key contains the descriptive details for your analysis.

    For example, given this structure:
    ```json
    {
        "R#OBESITY_KIDNEY_EFFECTS_01": {
            "full_description": "Obesity --[RiskFactor_AssociatedWith_OutcomeCKD]--> Chronic Kidney Damage (CKD)",
            "GroupName": "General_RiskFactorFor_CKD",
            "weight": 2
        },
        "LOGFLI_PREDICTS_MAU_ADJ_CCA_2018": { ... }
    }
    ```

    The exact, original relationship identifier to use for the first entry is the key itself: "R#OBESITY_KIDNEY_EFFECTS_01".

    Crucial Instruction: The relationship identifiers in your output list MUST EXACTLY MATCH these top-level keys (relationship_internal_id) from the 'Candidate Relationship List'. Do not alter, abbreviate, or rephrase them. Every character must be identical.

    Your output MUST be a single, valid JSON object containing only the key "selected_relationships". Do not include any text, notes, or explanations outside of the JSON object.

    { "selected_relationships": [ "R#OBESITY_KIDNEY_EFFECTS_01", "LOGFLI_PREDICTS_MAU_ADJ_CCA_2018", ... ] }

    The selected_relationships list must contain EXACTLY {num_to_select} relationship identifiers.

    Patient Profile (Cluster Statistics):

    {patient_data_str}

    Candidate Relationship List (in JSON format):

    {relationships_prompt_str}
    """

    # 摘要评分 Prompt
    DEFAULT_ABSTRACT_PROMPT = """You are a highly precise medical literature analyst AI. Your task is to rank a list of scientific articles based on their relevance to a specific cluster of health data (representing a group of individuals with shared characteristics) and the article's journal impact factor (IF). After a comprehensive evaluation, you will select and output **only the top 25 strongest articles** from the list you are given.

    You will be given two pieces of information:

    1.  **Patient Information**: A JSON object. **IMPORTANT NOTE:** Although labeled 'Patient Information', this object actually contains the **aggregated clinical features and descriptive statistics of a specific health cluster** (e.g., means, standard deviations, distributions, or dominant phenotypes).
    2.  **Article Data**: A JSON object containing the articles for evaluation.

    ---
    **How to Understand the 'Article Data' Structure:**

    The 'Article Data' is a JSON object that functions as a dictionary. **The DOI number of each article is the TOP-LEVEL KEY for each entry.** The value associated with that key is another object containing the 'paper_title', 'abstract', 'journal', and 'IF'. You must use this top-level key (the DOI) for your output.

    Here is a small example of the data structure:
    ```json
    {
        "10.1016/j.jhep.2017.08.024.": {
            "paper_title": "A fatty liver leads to decreased kidney function?",
            "abstract": "BACKGROUND & AIMS: Non-alcoholic fatty liver disease (NAFLD) has been associated...",
            "journal": "J Hepatol",
            "IF": 26.8
        },
        "10.24546/0100489395.": {
            "paper_title": "A High Fibrosis-4 Index is Associated with a Reduction in...",
            "abstract": "Liver fibrosis is associated with non-alcoholic fatty liver disease (NAFLD), and...",
            "journal": "Kobe J Med Sci",
            "IF": 1.1
        }
    }
    ```

    In this example, "10.1016/j.jhep.2017.08.024." is a DOI. You will extract this exact string for your output.

    Your Goal and Ranking Rules:

    Goal: Internally evaluate ALL articles in the provided Article Data. Based on this evaluation, create a ranked list of only the top 25 strongest articles.

    Scoring (0-100): Generate a single numerical score for each article.

    Contribution (50/50): The score is determined equally by:

        Relevance: How related the 'abstract' is to the **cluster's aggregate characteristics and statistical features** found in the 'Patient Information'.

        Impact Factor: The 'IF' value.

    Sorting: The final output must be sorted by score in descending order (highest to lowest).

    Required Output Format and CRITICAL REQUIREMENTS:

    Format: The output MUST be a single JSON object containing a dictionary.

    Content: This dictionary must contain ONLY the top 25 articles with the highest scores.

    Structure: The structure must be "DOI number": score.

    CRITICAL REQUIREMENTS:

    The keys of your output dictionary MUST be an EXACT, character-for-character match to the top-level keys (the DOIs) from the input Article Data.

    You are strictly forbidden from including anything else in the output. DO NOT output the paper title, abstract, journal name, or the IF value. The final JSON object must contain ONLY the DOI numbers and their corresponding numerical scores.

    Now, analyze the following new data:

    Patient Information: {patient_data_json}

    Article Data: {abstracts_data}

    Begin your analysis. Use the top-level keys from the Article Data as the DOI numbers. Evaluate all articles, select the top 25, and return ONLY the final JSON object for those 25 articles, formatted and sorted exactly as specified.
    """

    # --- 2. 侧边栏配置 ---
    st.sidebar.title("🔧 Retrieval Configuration (OpenAI)")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    openai_base_url = st.sidebar.text_input("Base URL", value="https://integrate.api.nvidia.com/v1")
    openai_model = st.sidebar.text_input("Model Name", value="moonshotai/kimi-k2-instruct-0905")

    st.sidebar.markdown("### 📂 Knowledge base file upload")
    file_entities = st.sidebar.file_uploader("1. Candidate entity library (JSON)", type=["json"], help="entities_summary_simp.json")
    file_relationships = st.sidebar.file_uploader("2. Candidate Relationship Repository (CSV)", type=["csv"], help="relationship_df_simp.csv")
    file_abstracts = st.sidebar.file_uploader("3. Abstract Database (JSON)", type=["json"], help="abstract435.json")
    file_kg = st.sidebar.file_uploader("4. Complete knowledge graph (JSON)", type=["json"],
                                       help="LK_KG / normalized_extraction.json")

    # --- 3. 提示词模版编辑区 ---
    st.subheader("📝Prompt Configuration (Preset)")
    with st.expander("View/Edit Prompts", expanded=False):
        prompt_entity_template = st.text_area("1. Entity retrieval Prompt", value=DEFAULT_ENTITY_PROMPT, height=300)
        prompt_rel_template = st.text_area("2. Relationship retrieval Prompt", value=DEFAULT_RELATIONSHIP_PROMPT, height=300)
        prompt_abstract_template = st.text_area("3. Abstract Rating Prompt", value=DEFAULT_ABSTRACT_PROMPT, height=300)

    # --- 4. 患者信息输入 ---
    st.subheader("🏥 Patient clinical information")
    patient_input_text = st.text_area("Please enter the patient description or clinical data (in JSON format or as a text description).", height=150,
                                      value='{\n  "age": 65,\n  "gender": "Male",\n  "diagnosis": "T2DM with NAFLD",\n  "lab_values": {"HbA1c": 8.5, "eGFR": 55}\n}')


    # --- 辅助函数 ---
    def call_openai(prompt, api_key, base_url, model):
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=False
        )
        return response.choices[0].message.content


    def extract_json_from_text(text):
        try:
            matches = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text)
            if matches: return json.loads(matches[-1])
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1: return json.loads(text[start:end + 1])
        except:
            pass
        return {}


    def normalize_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\s+', '', text)
        return text.strip(".,;:'\"")


    def clean_doi(doi_str):
        if not isinstance(doi_str, str): return ""
        return doi_str.strip().rstrip('.')


    # --- 主执行逻辑 ---
    if st.button("🚀 Begin retrieval and scoring", type="primary"):
        if not (openai_api_key and file_entities and file_relationships and file_abstracts and file_kg):
            st.error("Please configure the API key and upload all necessary database files first.(1-4)！")
        else:
            status = st.status("Smart search in progress (process: split into two batches to retrieve entities/relations/summaries → aggregate full-scale scoring)....", expanded=True)
            progress_bar = st.progress(0)

            try:
                # 0. 准备基础数据
                status.write("📂 [0/4] Loading and preprocessing knowledge base data...")
                db_entities = json.load(file_entities)
                db_relationships_df = pd.read_csv(file_relationships)
                db_abstracts = json.load(file_abstracts)
                db_kg = json.load(file_kg)

                # 预处理关系数据为字典
                db_relationships = {}
                for _, row in db_relationships_df.iterrows():
                    rid = row['relationship_internal_id']
                    db_relationships[rid] = {
                        "full_description": f"{row['source_entity_text']} --[{row['relationship_type_semantic_description']}]--> {row['target_entity_text']}",
                        "GroupName": row['GroupName'],
                        "weight": row['weight']
                    }

                # 解析患者输入
                try:
                    patient_data = json.loads(patient_input_text)
                except:
                    patient_data = {"description": patient_input_text}
                patient_str = json.dumps(patient_data, indent=2, ensure_ascii=False)

                progress_bar.progress(10)

                # ---------------------------------------------------------
                # Step 1: 实体检索 (Split -> 2 calls -> Merge)
                # ---------------------------------------------------------
                status.write("🔍 [1/4] Entity retrieval: selecting key entities in two batches...")

                # A. 拆分数据
                all_ent_keys = sorted(list(db_entities.keys()))
                mid_ent = len(all_ent_keys) // 2
                ent_keys_1 = all_ent_keys[:mid_ent]
                ent_keys_2 = all_ent_keys[mid_ent:]

                ent_subset_1 = {k: db_entities[k] for k in ent_keys_1}
                ent_subset_2 = {k: db_entities[k] for k in ent_keys_2}

                # B. 第一批调用 (Top 20)
                status.write("  - Analyzing entities Part 1 ...")
                prompt_ent_1 = prompt_entity_template \
                    .replace("{num_to_select}", "20") \
                    .replace("{patient_data_str}", patient_str) \
                    .replace("{entities_prompt_str}", json.dumps(ent_subset_1, indent=2, ensure_ascii=False))

                res_ent_1 = extract_json_from_text(
                    call_openai(prompt_ent_1, openai_api_key, openai_base_url, openai_model)).get("selected_entities",
                                                                                                  [])

                # C. 第二批调用 (Top 20)
                status.write("  - Analyzing entities Part 2 ...")
                prompt_ent_2 = prompt_entity_template \
                    .replace("{num_to_select}", "20") \
                    .replace("{patient_data_str}", patient_str) \
                    .replace("{entities_prompt_str}", json.dumps(ent_subset_2, indent=2, ensure_ascii=False))

                res_ent_2 = extract_json_from_text(
                    call_openai(prompt_ent_2, openai_api_key, openai_base_url, openai_model)).get("selected_entities",
                                                                                                  [])

                # D. 合并
                selected_entities = res_ent_1 + res_ent_2
                status.write(
                    f"✅ Entity filtering complete: a total of {len(selected_entities)} selected (Part1: {len(res_ent_1)} + Part2: {len(res_ent_2)})")
                progress_bar.progress(30)

                # ---------------------------------------------------------
                # Step 2: 关系检索 (Split -> 2 calls -> Merge)
                # ---------------------------------------------------------
                status.write("🔗 [2/4] Relation retrieval: Filtering mechanistic pathways in two batches...")

                # A. 拆分数据
                all_rel_keys = sorted(list(db_relationships.keys()))
                mid_rel = len(all_rel_keys) // 2
                rel_keys_1 = all_rel_keys[:mid_rel]
                rel_keys_2 = all_rel_keys[mid_rel:]

                rel_subset_1 = {k: db_relationships[k] for k in rel_keys_1}
                rel_subset_2 = {k: db_relationships[k] for k in rel_keys_2}

                # B. 第一批调用 (Top 20)
                status.write("  - Analyzing relationships Part 1 ...")
                prompt_rel_1 = prompt_rel_template \
                    .replace("{num_to_select}", "20") \
                    .replace("{patient_data_str}", patient_str) \
                    .replace("{relationships_prompt_str}", json.dumps(rel_subset_1, indent=2, ensure_ascii=False))

                res_rel_1 = extract_json_from_text(
                    call_openai(prompt_rel_1, openai_api_key, openai_base_url, openai_model)).get(
                    "selected_relationships", [])

                # C. 第二批调用 (Top 20)
                status.write("  - Analyzing relationships Part 2 ...")
                prompt_rel_2 = prompt_rel_template \
                    .replace("{num_to_select}", "20") \
                    .replace("{patient_data_str}", patient_str) \
                    .replace("{relationships_prompt_str}", json.dumps(rel_subset_2, indent=2, ensure_ascii=False))

                res_rel_2 = extract_json_from_text(
                    call_openai(prompt_rel_2, openai_api_key, openai_base_url, openai_model)).get(
                    "selected_relationships", [])

                # D. 合并
                selected_relationships = res_rel_1 + res_rel_2
                status.write(
                    f"✅ Filtering relationships completed: a total of {len(selected_relationships)} items selected (Part 1: {len(res_rel_1)} + Part 2: {len(res_rel_2)})")
                progress_bar.progress(60)

                # ---------------------------------------------------------
                # Step 3: 摘要评分 (Split -> 2 calls -> Merge -> Top 50)
                # ---------------------------------------------------------
                status.write("📄 [3/4] The abstracts are being evaluated in two batches....")

                # A. 拆分数据
                abs_items = list(db_abstracts.items())
                mid_abs = len(abs_items) // 2
                abs_subset_1 = dict(abs_items[:mid_abs])
                abs_subset_2 = dict(abs_items[mid_abs:])

                # B. 第一批调用 (Top 25)
                status.write("  - Evaluating the summary Part 1 (Top 25) ...")
                prompt_abs_1 = prompt_abstract_template \
                    .replace("{patient_data_json}", patient_str) \
                    .replace("{abstracts_data}", json.dumps(abs_subset_1, ensure_ascii=False))

                # 注意：摘要返回的是字典 {"DOI": score}
                res_abs_1 = extract_json_from_text(
                    call_openai(prompt_abs_1, openai_api_key, openai_base_url, openai_model))

                # C. 第二批调用 (Top 25)
                status.write("  - Evaluating the summary Part 2 (Top 25) ...")
                prompt_abs_2 = prompt_abstract_template \
                    .replace("{patient_data_json}", patient_str) \
                    .replace("{abstracts_data}", json.dumps(abs_subset_2, ensure_ascii=False))

                res_abs_2 = extract_json_from_text(
                    call_openai(prompt_abs_2, openai_api_key, openai_base_url, openai_model))

                # D. 合并并保留 Top 50
                combined_abstract_scores = {**res_abs_1, **res_abs_2}  # 合并字典
                # 按分数排序
                sorted_abs = sorted(combined_abstract_scores.items(), key=lambda x: x[1], reverse=True)
                top_50_abstracts = dict(sorted_abs[:50])  # 截取 Top 50

                status.write(f"✅ Abstract evaluation completed: After merging, {len(top_50_abstracts)} high-scoring abstracts were selected.")
                progress_bar.progress(80)

                # ---------------------------------------------------------
                # Step 4: 综合全量计算 (Script 4 Logic - Modified for ALL papers)
                # ---------------------------------------------------------
                status.write("⚖️ [4/4] Final calculation: comprehensive scoring of all knowledge base articles in progress...")

                # A. 预处理 Master KG (加速查找)
                kg_lookup = {}
                for paper in db_kg:
                    title = paper.get('document_metadata', {}).get('paper_title')
                    if not title: continue

                    # 预先标准化实体和关系
                    norm_ents = set(
                        normalize_text(e.get('entity_name_as_in_text')) for e in paper.get('extracted_entities', []))
                    norm_rels = set(normalize_text(r.get('relationship_internal_id')) for r in
                                    paper.get('extracted_relationships', []))

                    kg_lookup[title] = {
                        'norm_entities': norm_ents,
                        'norm_relationships': norm_rels
                    }

                # B. 建立 DOI -> Title 映射 (用于将摘要评分对应到文章标题)
                doi_to_title = {clean_doi(k): v.get('paper_title') for k, v in db_abstracts.items()}

                # C. 准备 LLM 选中的实体/关系集合 (标准化)
                llm_ents_norm = set(normalize_text(e) for e in selected_entities)
                llm_rels_norm = set(normalize_text(r) for r in selected_relationships)

                # D. 遍历 KG 中 **所有** 文章进行打分
                records = []
                all_titles = list(kg_lookup.keys())  # 全量

                for title in all_titles:
                    # 1. 获取摘要分 (如果该文章在 Step 3 的 Top 50 中，则有分，否则为 0)
                    raw_abs_score = 0
                    # 反查：看看这篇 Title 对应的 DOI 是否在 top_50_abstracts 里
                    # 这里的效率略低，但为了逻辑清晰暂且如此。
                    # 更优解是反转 top_50_abstracts 为 title->score map
                    for d_doi, d_score in top_50_abstracts.items():
                        if doi_to_title.get(clean_doi(d_doi)) == title:
                            raw_abs_score = d_score
                            break

                    # 2. 实体匹配分 (全量匹配)
                    ent_hits = len(llm_ents_norm & kg_lookup[title]['norm_entities'])

                    # 3. 关系匹配分 (全量匹配)
                    rel_hits = len(llm_rels_norm & kg_lookup[title]['norm_relationships'])

                    # 只要有任意一项得分，就记录下来 (避免展示全 0 的无意义行)
                    if raw_abs_score > 0 or ent_hits > 0 or rel_hits > 0:
                        records.append({
                            "Paper Title": title,
                            "Abstract Score": raw_abs_score,
                            "Entity Hits": ent_hits,
                            "Relationship Hits": rel_hits
                        })

                progress_bar.progress(100)

                if records:
                    df = pd.DataFrame(records)

                    # 归一化 (MinMaxScaler)
                    scaler = MinMaxScaler()
                    cols = ["Abstract Score", "Entity Hits", "Relationship Hits"]
                    # 只有当列中有不同值时才归一化，否则为0
                    for col in cols:
                        if df[col].max() > df[col].min():
                            df[col] = scaler.fit_transform(df[[col]])
                        elif df[col].max() > 0:
                            df[col] = 1.0  # 只有一个值且>0，归一为1
                        else:
                            df[col] = 0.0

                    # 加权计算 (权重可调: 0.4, 0.3, 0.3)
                    df["Final Score"] = (df["Abstract Score"] * 0.4 +
                                         df["Entity Hits"] * 0.3 +
                                         df["Relationship Hits"] * 0.3)

                    # 排序
                    df = df.sort_values("Final Score", ascending=False)

                    # =========== 【新增】保存结果到 Session State ===========
                    st.session_state['scored_df'] = df  # 缓存评分结果
                    st.session_state['kg_data_cache'] = db_kg  # 缓存完整 KG 数据 (避免重新解析)
                    st.session_state['patient_str_cache'] = patient_str  # 缓存患者信息
                    # ======================================================

                    status.update(label="🎉 Calculation complete！", state="complete", expanded=False)
                    st.success(f"Calculation complete. Scores have been generated for {len(df)} relevant articles.")

                    # 展示所有结果
                    st.dataframe(
                        df[["Paper Title", "Final Score", "Abstract Score", "Entity Hits",
                            "Relationship Hits"]].style.background_gradient(subset=["Final Score"], cmap="Greens"),
                        use_container_width=True
                    )
                else:
                    st.warning("Result is empty：No articles matching the filter criteria were found in the knowledge base.")

            except Exception as e:
                status.update(label="❌ An error has occurred.", state="error")
                st.error(f"An exception occurred during processing: {e}")


    # ==========================================
    # 5. 智能综合报告生成 (新增模块)
    # ==========================================

    # 辅助函数：内存文本格式化 (改编自你的参考脚本)
    def generate_context_text(doc_data, rank, score):
        """将单个文档的 JSON 转换为易于 LLM 阅读的结构化文本 (不生成文件)"""
        buffer = io.StringIO()

        def format_val(v):
            if v is None: return "Not specified"
            if isinstance(v, bool): return "Yes" if v else "No"
            return str(v)

        def write_kv(k, v, indent=0):
            prefix = "  " * indent
            buffer.write(f"{prefix}- {k.replace('_', ' ').title()}: {format_val(v)}\n")

        def write_quotes(quotes, indent=0):
            if quotes:
                prefix = "  " * indent
                buffer.write(f"{prefix}- Supporting Evidence Quotes:\n")
                for q in quotes:
                    clean_q = str(q).replace('\n', ' ').strip()
                    buffer.write(f"{prefix}  > \"{clean_q}\"\n")

        # Header
        meta = doc_data.get("document_metadata", {})
        title = meta.get('paper_title', 'Untitled')
        buffer.write(f"\n{'=' * 40}\n")
        buffer.write(f"PAPER RANK: {rank} | SCORE: {score:.4f}\n")
        buffer.write(f"TITLE: {title}\n")
        buffer.write(f"{'=' * 40}\n")

        # 1. Metadata
        buffer.write("\n[1] METADATA:\n")
        for k, v in meta.items():
            if k != 'paper_title': write_kv(k, v)

        # 2. Main Context
        context = doc_data.get("main_study_context", {})
        if context:
            buffer.write("\n[2] STUDY DESIGN:\n")
            for k, v in context.items(): write_kv(k, v)

        # 3. Population
        pop = doc_data.get("main_population_characteristics", {})
        if pop:
            buffer.write("\n[3] POPULATION:\n")
            for k, v in pop.items():
                if k == 'key_baseline_comorbidities' and isinstance(v, list):
                    buffer.write("  - Key Comorbidities:\n")
                    for c in v:
                        buffer.write(
                            f"    * {c.get('comorbidity', 'N/A')}: {c.get('prevalence_or_mean_value', 'N/A')}\n")
                else:
                    write_kv(k, v)

        # 4. Entities (简化版，只列出关键属性)
        entities = doc_data.get("extracted_entities", [])
        if entities:
            buffer.write("\n[4] KEY ENTITIES & RISK FACTORS:\n")
            for ent in entities:
                name = ent.get('entity_name_as_in_text', 'N/A')
                e_type = ent.get('entity_type', 'N/A')
                desc = ent.get('general_description_from_text')
                buffer.write(f"  * Entity: {name} ({e_type})\n")
                if desc: buffer.write(f"    Desc: {desc}\n")

                # Contextual Data
                for ctx in ent.get('contextual_attributes_and_values', []):
                    attrs = ctx.get('attributes_in_this_context', {})
                    if attrs:
                        buffer.write("    - Findings in context:\n")
                        for ak, av in attrs.items():
                            buffer.write(f"      {ak}: {av}\n")

        # 5. Relationships (重点)
        rels = doc_data.get("extracted_relationships", [])
        if rels:
            buffer.write("\n[5] MECHANISTIC RELATIONSHIPS:\n")
            for rel in rels:
                src = rel.get('source_entity_text')
                tgt = rel.get('target_entity_text')
                rtype = rel.get('relationship_type_semantic_description')
                summary = rel.get('natural_language_summary_of_finding_by_llm')

                buffer.write(f"  * {src} --[{rtype}]--> {tgt}\n")
                if summary: buffer.write(f"    Summary: {summary}\n")

                attrs = rel.get('attributes_of_the_relationship', {})
                eff_size = attrs.get('effect_size_reported_for_relationship')
                p_val = attrs.get('p_value_or_fdr_for_relationship')
                if eff_size or p_val:
                    buffer.write(f"    Stats: Effect={eff_size}, P-val={p_val}\n")

        return buffer.getvalue()


    # --- 界面逻辑 ---

    st.markdown("---")
    st.header("🧠 5. Intelligent Comprehensive Report Generation")

    # 检查是否有缓存数据
    if 'scored_df' in st.session_state and st.session_state['scored_df'] is not None:

        col_cfg1, col_cfg2 = st.columns([1, 2])

        with col_cfg1:
            top_n = st.slider("Number of studies selected for analysis (Top N)", min_value=1, max_value=100, value=20,
                              help="Select the top-N highest-scoring papers and extract the detailed knowledge-graph content from each to serve as the context.")

        with col_cfg2:
            default_report_prompt = """
Part 1: Role, Mission, and Final Objective                                                                                              
    • [Role]                                                                                                                            
        ◦ You are a top-tier clinical epidemiologist and risk-modelling methodology expert.                                             
    • [Core Mission]                                                                                                                    
        ◦ Your task is to generate a comprehensive methodological report on **“how to assess future chronic kidney disease (CKD) risk in
    • [Key Output Requirements]                                                                                                         
        ◦ Nature of Output:                                                                                                             
            ▪ An evidence-based clinical decision-support report.                                                                       
            ▪ It must simulate how a clinical expert would systematically synthesise, weigh and interpret a large body of literature to 
            ▪ The focus is on **“how to perform a composite assessment”**, not on “how to build a prediction model”.                    
        ◦ Core Content:                                                                                                                 
            ▪ Systematically answer: under the current body of evidence, which of the supplied patient fields should be identified, quan
        ◦ Target Audience:                                                                                                              
            ▪ Clinical researchers and data scientists.                                                                                 
            ▪ The report must combine clinical insight with methodological rigour.                                                      

Part 2: Inputs & Knowledge-Base Structure                                                                                               
        ◦ PDF literature: one or more PDF scientific articles constituting your sole knowledge source.                                  
        ◦ Target Patient Fields: a list of patient-level variables {instance_field}.                                                    
    • [Framework for Understanding the Knowledge Base]                                                                                  
        ◦ While analysing the PDFs, internally build a structured knowledge map containing:                                             
            ▪ Document Metadata: title, year, journal, impact factor, etc. (impact factor is one metric of evidence quality).           
            ▪ Study Context: design (e.g. cohort, RCT), objective, sample size, baseline population characteristics; study design is the
            ▪ Extracted Entities: biomarkers, diseases, risk factors, drugs mentioned.                                                  
            ▪ Extracted Relationships: demonstrated associations (e.g. “A causes B” or “A positively correlates with B”)—your main evide
            ▪ Systemic Interactions: crucial—pay special attention to discussions/conclusions on multi-factor interactions, synergies or

Part 3: Required Report Structure                                                                                                       
Your final report must be clearly structured and contain at least the following sections:                                               
    1. Introduction                                                                                                                     
        ◦ Briefly state why CKD-risk assessment in NAFLD matters.                                                                       
        ◦ Clarify that this report aims to build a methodological framework based on the supplied literature.                           
    2. Relevant Risk-Factor Identification & Mapping                                                                                    
        ◦ List which fields in the “target list” map to CKD risk factors documented in the literature.                                  
        ◦ Apply a two-stage mapping:                                                                                                    
            ▪ A. Primary Risk Factors – Direct & Quantifiable Links:                                                                    
                • Identify fields proven in the knowledge base as independent, quantifiable CKD risk factors.                           
                • List them and note their risk nature (demographic, metabolic, liver-severity, etc.).                                  
            ▪ B. Contextual & Associated Factors – Indirect or Conceptual Links:                                                        
                • For all remaining fields, perform a second “contextual” search.                                                       
                • Guidance:                                                                                                             
                    ◦ Broaden the Concept: if a specific field (e.g. apple_intake) is absent, search its broader parent (e.g. “diet”, “l
                    ◦ Identify Indirect Links: check whether the field relates to management/background of a primary factor (e.g. socioe
                    ◦ Summarise the Context:                                                                                            
                        – First, state explicitly: “Based on the current knowledge base these fields are not independent, quantifiable C
                        – Then add any background mention (e.g. “However, the knowledge base highlights ‘healthy diet’ as part of overal
            ▪ Handle Truly Irrelevant Fields:                                                                                           
                • Only if a field plus all parent concepts are completely absent may you classify it as “no relevance found in current k
    3. Risk Quantification & Evidence Appraisal                                                                                         
        ◦ For each identified factor provide quantitative estimates (HR, OR) and statistical significance (p-value).                    
        ◦ Grade the quality of each supporting piece of evidence by integrating:                                                        
            ▪ Study design (RCT > prospective cohort > retrospective).                                                                  
            ▪ Study quality (journal impact factor, sample size).                                                                       
            ▪ Effect size magnitude and consistency (confidence intervals).                                                             
    4. Multi-Factor Interactions & Conflict Resolution                                                                                  
        ◦ Interaction analysis: explore synergistic or antagonistic effects among factors.                                              
        ◦ Cross-study synthesis:                                                                                                        
            ▪ Corroboration: if multiple high-quality studies agree, highlight as strong evidence.                                      
            ▪ Conflict Resolution: when studies disagree:                                                                               
                • Contrast designs, populations, methods to explain discordance.                                                        
                • Give a directional judgement based on evidence hierarchy (e.g. favour the more rigorous design).                      
    5. Recommendations for a Composite Risk Model                                                                                       
        ◦ Propose an integrative risk-assessment strategy based on the above.                                                           
        ◦ Discuss statistical issues when combining factors: independence, multicollinearity, interaction terms.                        
        ◦ Recommend prioritising factors with higher evidence grade, stronger effect size and cross-study consistency.                  
    6. Practical Considerations: Guiding Principles for Missing Data                                                                    
        ◦ This section is essential. Provide explicit guidance on how to adapt the assessment when key variables are missing in a real-c
            ▪ Tier-Based Downgrade: refer to the report’s “tiered assessment strategy”. If data for a higher tier (e.g. liver-fibrosis s
            ▪ Acknowledge Limitations: warn that absence of key metrics (especially high-weight factors like fibrosis) markedly reduces 
            ▪ Use of Proxies: suggest usable surrogates—e.g. if FIB-4 cannot be calculated but history shows “cirrhosis”, imaging proves
            ▪ Recommend Data Collection: emphasise that for high-weight missing data (especially basic labs needed for FIB-4/NFS) the fi
    7. Limitations                                                                                                                      
        ◦ State limitations of the current knowledge base (supplied PDFs) and any research areas not covered.                           

Part 4: Your Thinking Process & Workflow                                                                                                
Follow this sequence exactly:                                                                                                           
    • [Primary Workflow]                                                                                                                
        1. Step 1 – Deconstruct & Map: precisely match the “target field list” to key entities in the knowledge base.                   
        2. Step 2 – Find & Extract Evidence: retrieve the core relationship between each mapped entity and “CKD”.                       
        3. Step 3 – Assess Evidence Strength: grade quality, recording source, design, p-value and risk metrics.                        
        4. Step 4 – Investigate Interactions: revisit source papers, focusing on any discussion of systemic interactions.               
        5. Step 5 – Synthesize & Report: integrate all findings into the final report following Part 3 structure.                       
    • [Advanced Research Patterns]                                                                                                      
        1. Pattern 1 – Global Exploration                                                                                               
            ▪ When: initial clues scarce or important factors possibly missing.                                                         
            ▪ Action: rapidly scan abstracts/conclusions of all papers to build a global NAFLD-CKD risk map, then zoom in.              
        2. Pattern 2 – Pivoting & Deep Dive                                                                                             
            ▪ When: a pivotal paper (e.g. high-quality review) or recurrent key entity (e.g. “insulin resistance”) appears.             
            ▪ Action: use that paper/entity as a pivot to excavate all internal entities, relationships and context.                    
        3. Pattern 3 – Comparative Analysis                                                                                             
            ▪ When: need to verify consistency of a specific association (e.g. “hypertension → CKD risk”) across papers.                
            ▪ Action: horizontally contrast findings across all papers to address corroboration vs conflict.                            
        4. Pattern 4 – Contextual Inquiry                                                                                               
            ▪ When: a key relationship is found but its conditions are unclear.                                                         
            ▪ Action: analyse the paper’s “study design & context” and “population characteristics” to define the applicable population.
            """
            report_prompt = st.text_area("System Prompt", value=default_report_prompt.strip(), height=200)

        if st.button("🚀 Generate a comprehensive analysis report", type="primary"):
            with st.spinner("Extracting full-text knowledge and generating report..."):
                try:
                    # 1. 获取 Top N 列表
                    df_top = st.session_state['scored_df'].head(top_n)
                    target_titles = df_top["Paper Title"].tolist()

                    # 2. 构建上下文 (在内存中)
                    full_context_buffer = io.StringIO()

                    # 写入患者信息
                    patient_info = st.session_state.get('patient_str_cache', "No patient data provided.")
                    full_context_buffer.write(f"=== PATIENT CLINICAL PROFILE ===\n{patient_info}\n\n")

                    full_context_buffer.write(f"=== TOP {top_n} RETRIEVED LITERATURE EVIDENCE ===\n")

                    # 遍历完整 KG 寻找对应 Title 的内容
                    # (为了加速，可以预先构建 title->doc 索引，但这里 Top N 较小，直接遍历也可)
                    found_count = 0
                    kg_data = st.session_state['kg_data_cache']

                    for idx, row in df_top.iterrows():
                        title = row['Paper Title']
                        score = row['Final Score']
                        rank = found_count + 1

                        # 在 KG 中查找匹配文档
                        doc_match = next(
                            (d for d in kg_data if d.get('document_metadata', {}).get('paper_title') == title), None)

                        if doc_match:
                            # 调用上面的格式化函数
                            doc_text = generate_context_text(doc_match, rank, score)
                            full_context_buffer.write(doc_text)
                            found_count += 1
                        else:
                            full_context_buffer.write(
                                f"\n[Warning: Full content for '{title}' not found in KG cache]\n")

                    context_str = full_context_buffer.getvalue()

                    # 3. 组装最终 LLM Prompt
                    final_llm_prompt = f"{report_prompt}\n\n{context_str}"

                    # 4. 调用 LLM
                    # 使用之前定义的 helper call_openai
                    # 4. 调用 LLM
                    report_response = call_openai(final_llm_prompt, openai_api_key, openai_base_url, openai_model)

                    # =========== 【新增/修改】保存到 Session State ===========
                    # 将生成的报告文本保存到一个特定的 Key 中
                    st.session_state['step5_generated_report'] = report_response
                    st.toast("✅ The report has been saved to memory and can be used directly in the 'Final Score' module!")  # 弹出提示
                    # ======================================================

                    # 5. 展示结果
                    st.divider()
                    st.subheader("📋 Clinical Comprehensive Analysis Report")
                    st.markdown(report_response)
                    # 可选：展示发送给 LLM 的上下文 (用于调试)
                    with st.expander("View the full context sent to the LLM"):
                        st.text(context_str)

                except Exception as e:
                    st.error(f"Error generating the report.{e}")
    else:
        st.info("ℹ️ Please complete the rating process in [Step 4] above first to generate the data for analysis.")

# ... (之前的代码)

# ==============================================================================
# 模式 C: 最终临床评分与综合推理 (LLM Scoring)
# ==============================================================================
elif app_mode == "📊 Final Clinical Scoring System":
    st.title("📊 Final Clinical Score and Comprehensive Reasoning")
    st.markdown("This module performs final risk quantification and inference using a large language model, drawing on patient data and medical reports sourced either from memory or local files.")

    # --- 1. LLM 模型配置 (独立配置，因为最后一步可能需要更强的模型) ---
    with st.expander("⚙️ Reasoning Model Configuration (Overrides Default Settings)", expanded=True):
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            # 默认使用之前的配置，但允许修改
            final_api_key = st.text_input("API Key", value='', type="password", key="final_key")
            final_base_url = st.text_input("Base URL", value='https://integrate.api.nvidia.com/v1', key="final_url")
        with c_col2:
            final_model = st.text_input("Model Name", value='moonshotai/kimi-k2-instruct-0905', key="final_model")
            final_temp = st.slider("Temperature", 0.0, 1.0, 0.2, help="The lower the inference, the more stable it is.")

    # --- 2. 证据来源选择 ---
    st.subheader("📂 Evidence-source configuration")

    col_source1, col_source2 = st.columns(2)

    # 来源 A: 内存中的报告 (Step 5 生成的)
    report_memory_content = ""
    use_memory_report = False

    with col_source1:
        st.markdown("""Memorandum: Report Compiled Out of Memory""")

        # 检查 Session State 中是否有 Step 5 生成的报告
        if 'step5_generated_report' in st.session_state and st.session_state['step5_generated_report']:
            st.success("✅ The consolidated report generated in Step 5 has been detected.")
            use_memory_report = st.checkbox("Use in-memory knowledge reports (Source 1)", value=True)

            if use_memory_report:
                # 直接读取之前保存的文本
                report_memory_content = st.session_state['step5_generated_report']

                # 可选：显示预览
                with st.expander("Preview memory report contents"):
                    st.markdown(report_memory_content[:500] + "...")

        # 如果没有报告，但在 Step 4 算过分，提示用户去 Step 5 生成
        elif 'scored_df' in st.session_state:
            st.warning("⚠️ Rating data has been detected, but a comprehensive report has not yet been generated. Please proceed to [Step 5] and click Generate.")
        else:
            st.info("ℹ️ No memory report detected (please run Steps 4 & 5 first).")

    # 来源 B: 本地上传的补充报告
    report_upload_content = ""
    with col_source2:
        st.markdown("Source B: Local Supplementary Report (optional)")
        uploaded_report_file = st.file_uploader("Upload expert consensus or guidelines in TXT/MD format", type=['txt', 'md'])
        if uploaded_report_file:
            report_upload_content = uploaded_report_file.getvalue().decode("utf-8")
            st.success(f"✅ Loaded: {uploaded_report_file.name}")

    # --- 3. 患者数据与系统提示词 ---
    st.subheader("🏥 Patient data and instructions")
    col_pat1, col_pat2 = st.columns([3, 1])
    with col_pat1:
        # 尝试从缓存获取患者数据
        default_patient = st.session_state.get('patient_str_cache', '{\n  "age": 65,\n  "condition": "..." \n}')
        final_patient_data = st.text_area("Patient Clinical Data (JSON)", value=default_patient, height=150)

    with col_pat2:
        st.markdown("**Auxiliary scoring (optional)**")
        ckd_pc_score = st.number_input(
            "CKD-PC Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            help="Input the CKD-PC score based on traditional clinical indicators. If the value entered is greater than 0, the reference threshold of the combined model will be shown in the results."
        )

    # 预置的高级 Prompt (包含你提供的 Principles)
    DEFAULT_SCORING_PROMPT = """You are a top-tier clinical AI reasoning expert. Your mission is to generate an authoritative risk assessment based on the provided evidence.

--- Guiding Principles for Reasoning (Few-Shot Scoring Principles) ---
Disclaimer: The following are concise guiding principles designed to demonstrate core reasoning patterns and are not exhaustive. For any new case, you must conduct a more complex and comprehensive, individualized reasoning process based on the totality and uniqueness of the evidence presented.

Principle 1: Synergistic Risk - Assign High Weight
Scenario: The patient presents with multiple metabolic abnormalities (e.g., hyperlipidemia, hyperuricemia, fatty liver indicators).
Reasoning: The synergistic effect of this cluster of abnormalities poses a risk far greater than the sum of its individual parts. This cluster should be assigned a disproportionately high positive weight.

Principle 2: Strong Protective Factor - Assign Negative Weight
Scenario: The patient exhibits one or more powerful protective factors, such as a long-term, consistent level of physical activity far exceeding recommended guidelines.
Reasoning: An exceptionally high level of a positive behavior provides a potent risk mitigation effect and should be assigned a strong negative (protective) weight.

Principle 3: Risk Correlation & Avoiding Double-Counting
Scenario: The patient has numerous negative health indicators that are highly correlated (e.g., severe obesity, hypertension, dyslipidemia).
Reasoning: Avoid "double-counting" risk. The primary drivers (e.g., severe obesity) should receive the majority of the risk weight, while the correlated secondary markers contribute only a minor additional weight.

Principle 4: Confirming Absence of Major Accelerators
Scenario: The patient's clinical profile suggests a potential for a major, uncaptured risk accelerator.
Reasoning: If supplementary information clearly rules out the presence of this major risk accelerator, it increases confidence that the risk is not being driven by this specific high-impact pathway.

Principle 5: Strong Independent Factor - Assign Major Weight
Scenario: The patient has a strong risk (or protective) factor that has a low degree of biological pathway correlation with other observed factors.
Reasoning: When a factor's mechanism is relatively independent, the risk (or benefit) it confers is a purely additive increment. The weight assigned to it should be higher.

Principle 6: Data Quality Scrutiny - Modulate Confidence & Weight
Scenario: A key piece of information has quality issues (e.g., a crucial lab test is from several years ago).
Reasoning: The weight assigned to decisions based on old, vague, or low-quality evidence must be significantly dampened, and the confidence_score must reflect this uncertainty.

Principle 7: Competing Risks Consideration
Scenario: A patient has an extremely severe, non-renal comorbidity with a high short-to-medium term mortality risk.
Reasoning: The long-term risk of developing CKD is influenced by the "competing risk" of death from another cause. For very long-term predictions (10-15 years), a modest downward dampening factor should be applied.

--- End of Principles ---

Required JSON Output Format:
```json
{{
    "evidence_summary": "<A concise, neutral summary of the key findings from the participant's information, noting the primary source of evidence for each key point (which report, or a consensus of multiple reports).>",
    "reasoning_process": "<**You must follow these steps, citing the evidence source (Report 1, Report 2, or consensus) for your evidence-based quantitative reasoning:** 1. Foundational Risk Factor & Trajectory Inventory: Systematically identify risk factors. Crucially, compare the baseline data with the most recent follow-up to calculate the 'Rate of Change' for key metrics (e.g., eGFR slope, weight change direction). Identify if the patient is a 'Rapid Decliner', 'Stable', or 'Improving' phenotype. 2. **Integrative Risk Calculation and Pathway Analysis:** Construct the risk score through a weighted synthesis of all evidence. a. **Identify Intersections and Differences in Evidence:** Clearly state which risk assessments are based on multi-source consensus and which are based on the unique insights of a specific report. b. **Critical Weight Allocation:** Assign a quantitative risk contribution to each factor. When information conflicts, clearly articulate which viewpoint you are adopting and why, and explain how you will handle it to avoid double-counting (Principle 3). c. **Identify and Quantify Synergies:** Actively seek out and quantify synergistic (1+1>2) or antagonistic effects between factors.d. Weighing History vs. Current Status: Explicitly state how the participant's historical stability or instability influences the final score. If recent data contradicts the long-term trend (e.g., sudden acute spike), apply Principle 6 (Data Quality/Acute Context) to decide if it's a permanent shift or an outlier. 3. **Time Horizon and Competing Risk Analysis:** Project the synthesized risk over 1, 2, 3, and 5-year intervals, explaining the evolution of the risk trajectory and and considering competing risks (Principle 7). 4. **Final Score Synthesis and Sanity Check:** Aggregate all weighted contributions into the final risk scores for each time point and perform a final 'Clinical Plausibility Sanity Check,' explaining any differences between your synthesized assessment and the prediction a single model might produce, and justifying the rationale for these differences.>",
    "risk_scores": {{
        "3_year_risk_percent": <float>,                
        "5_year_risk_percent": <float>,
        "10_year_risk_percent": <float>,
        "15_year_risk_percent": <float>
    }},
    "confidence": {{
        "confidence_score": <float, from 0.0 to 1.0>,
        "confidence_reasoning": "<Justify the confidence score based on the quality, completeness, and consistency of the available data. Specifically mention the availability of core metrics emphasized in the key reports (e.g., the components for calculating liver fibrosis scores).>"
    }}
}}
"""
    system_instruction = st.text_area(" System Prompt", value=DEFAULT_SCORING_PROMPT, height=200)


    # --- 4. 核心逻辑: 动态提示词构建函数 ---
    def construct_final_prompt(sys_prompt, patient_json, report1_text, report2_text):
        """
        根据可用报告的数量动态组装 Prompt
        """
        # 1. 基础头部
        full_prompt = f"{sys_prompt}\n\n"
        full_prompt += "--- ANALYSIS TASK ---\n"

        # 2. 动态插入报告内容与特定任务描述

        # 情况 A: 两个报告都存在
        if report1_text and report2_text:
            full_prompt += f"""
You will receive three distinct sources of information:
1.  **A methodological report on CKD risk for a specific population ("Report 1").**
2.  **A report on a well-established CKD risk model for the general population ("Report 2").**
3.  **The participant's comprehensive longitudinal health records, including baseline data, historical follow-up visits, and temporal biomarker trends.**

Your analysis must be a deep synthesis of all sources.

--- Evidence Base and Background ---

**Evidence Source One: Specific Population Risk Perspective (e.g., a report on CKD risk in a fatty liver disease population)**
{report1_text[:15000]}
* **Role and Value**: This report provides in-depth insights into CKD risk within a specific disease context (such as fatty liver disease). It may contain detailed analysis of risk phenotypes unique to this population, key biomarkers, and distinct risk interactions.

**Evidence Source Two: General Population Risk Perspective (e.g., a report on a general risk model like CKD-PC)**
{report2_text[:15000]}
* **Role and Value**: This report offers a robust risk assessment framework validated in large-scale, multi-ethnic cohorts. It includes core risk factors with widely confirmed efficacy and their quantitative weights, serving as a solid benchmark and providing supplementary evidence for factors that may not be quantitatively detailed in 'Report 1' (such as the Urine Albumin-to-Creatinine Ratio, UACR).


--- Core Task: Evidence Synthesis and Critical Thinking Guide ---

Your core task is to **reconcile, integrate, and elevate** all information sources. You cannot simply choose one report and ignore the other; both reports are of equal importance. You must follow these critical thinking guidelines:

1.  **Identify Consensus, Establish Strong Evidence**:
    * When "Report 1," "Report 2," and the previous reasoning log align on a risk factor (e.g., age, diabetes), it should be treated as the highest level of evidence.

2.  **Analyze Differences, Conduct Critical Weighing**:
    * When sources offer unique or seemingly contradictory viewpoints (e.g., one report emphasizes BMI while another focuses on the "lean fatty liver" paradox), you must engage in deep critical thinking. Your reasoning process must clearly articulate:
        * **Contextual Specificity**: Does "Report 1" (specific population) offer a viewpoint that is more explanatory in a specific pathophysiological context than a general rule?
        * **Evidence Strength**: Does "Report 2" (general model) include a key variable validated in larger-scale studies that is not included in "Report 1"?
        * **Decision-Making**: How do you ultimately decide to adopt, merge, or adjust this information? For example, you might decide to use the variables from "Report 2" as the primary stratification tool while using the special phenotypes from "Report 1" to fine-tune the risk within that stratification (or vice versa).

3.  **Construct a Comprehensive, Multidimensional Argument**:
    * Your final assessment should not be a simple average or a choice between viewpoints but must be a new, coherent, and synthesized conclusion. You need to clearly argue why certain pieces of evidence were prioritized in the final risk calculation. Use evidence from one source to **validate, supplement, or challenge** evidence from another, thereby constructing a logically rigorous and sound risk profile.

4.  **Maintain Flexibility, No Pre-set Starting Point**:
    * Both reports are solid foundations for your analysis. You can start from the strengths of either report and use the insights from the other to supplement and refine it, or vice versa. The key is to build the most comprehensive and reasonable risk assessment.

5. Incorporate Temporal Dynamics (The "Velocity" of Disease):

    Trend over Value: Do not rely solely on the most recent static value. You must calculate and analyze the trajectory (e.g., the slope of eGFR decline, the fluctuation of liver enzymes). A rapid decline from normal to borderline is often riskier than a stable borderline value.

    Visit Pattern Analysis: Consider the frequency and regularity of follow-up visits ("visit counts"). High frequency might indicate active health management (protective context) or unstable disease requiring monitoring (risk context)—you must interpret this based on the clinical notes.

"""

        # 情况 B: 只有报告 1 (内存报告)
        elif report1_text and not report2_text:
            full_prompt += f"""


You will receive A methodological report on CKD risk for a specific population and The participant's individual health data.

Your analysis must be a deep synthesis of all informations.

--- Evidence Base and Background ---

{report1_text[:20000]}
* **Role and Value**: This report provides in-depth insights into CKD risk within a specific disease context (such as fatty liver disease). It may contain detailed analysis of risk phenotypes unique to this population, key biomarkers, and distinct risk interactions.
"""

        # 情况 C: 只有报告 2 (上传报告)
        elif not report1_text and report2_text:
            full_prompt += f"""
You will receive  A report on a well-established CKD risk model for the general population and the participant's individual health data.

Your analysis must be a deep synthesis of all sources.

--- Evidence Base and Background ---

**General Population Risk Perspective (e.g., a report on a general risk model like CKD-PC)**
{report2_text[:20000]}
* **Role and Value**: This report offers a robust risk assessment framework validated in large-scale, multi-ethnic cohorts. It includes core risk factors with widely confirmed efficacy and their quantitative weights, serving as a solid benchmark.
"""

        # 情况 D: 无报告 (纯盲测/通用知识)
        else:
            full_prompt += f"""
**No external reports provided.**
**Instruction:**
- Perform the assessment based **solely** on your internal top-tier clinical knowledge (General Medical Knowledge).
- Clearly state that no specific reference documents were used.
"""

        # 3. 插入患者数据 (固定部分)
        full_prompt += f"""
--- PATIENT DATA ---
{patient_json}

Now, begin your analysis and output the JSON response.
"""
        return full_prompt


    # --- 5. 执行推理 ---
    if st.button("🚀 Generate the final scoring report", type="primary"):
        if not final_api_key:
            st.error("Please provide the API key.")
        else:
            with st.spinner("Inferring based on multiple sources of evidence..."):
                try:
                    # 1. 动态构建 Prompt
                    final_prompt_str = construct_final_prompt(
                        system_instruction,
                        final_patient_data,
                        report_memory_content if use_memory_report else None,
                        report_upload_content if report_upload_content else None
                    )

                    # 2. 调用 LLM (复用之前的 call_openai 辅助函数)
                    # 注意：这里可能需要支持更大的 max_tokens
                    client = OpenAI(api_key=final_api_key, base_url=final_base_url)
                    response = client.chat.completions.create(
                        model=final_model,
                        messages=[{"role": "user", "content": final_prompt_str}],
                        temperature=final_temp,
                        stream=False
                    )
                    result_text = response.choices[0].message.content

                    # 3. 展示结果
                    st.divider()
                    col_res1, col_res2 = st.columns([0.6, 0.4])

                    with col_res1:
                        st.subheader("📝 Reasoning result")
                        st.markdown(result_text)
                        # ======================================================
                        # 新增模块：定量风险参考标准展示 (基于 PDF 数据)
                        # ======================================================
                        # ======================================================
                        # 新增模块：定量风险参考标准展示 (带自动计算功能)
                        # ======================================================
                        st.divider()
                        st.subheader("📏 Risk stratification reference standards (quantitative aids)")

                        # --- 1. 尝试从 LLM 结果中提取 5年风险评分 ---
                        llm_5y_score = None
                        try:
                            # 简单的 JSON 提取逻辑 (兼容代码块或纯文本)
                            json_str = result_text
                            if "```json" in result_text:
                                import re

                                match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL)
                                if match:
                                    json_str = match.group(1)
                            elif "{" in result_text:
                                p1 = result_text.find('{')
                                p2 = result_text.rfind('}')
                                if p1 != -1 and p2 != -1:
                                    json_str = result_text[p1:p2 + 1]

                            # 解析
                            data_json = json.loads(json_str)
                            # 尝试获取 5年 风险值 (根据 Prompt 定义的 keys)
                            if "risk_scores" in data_json:
                                llm_5y_score = float(data_json["risk_scores"].get("5_year_risk_percent", 0))
                        except Exception as e:
                            # 解析失败不阻断流程，只是不显示计算结果
                            print(f"JSON Parse Error: {e}")
                            pass

                        # --- 2. 定义 PDF 中的参考数据字典 ---
                        # 数据来源：Cumulative Incidence Curves PDF (At 16.2 yrs)
                        ref_data = {
                            "Kimi-K2": {
                                "base": {"cutoff": 7.00, "sens": "64.4%", "spec": "81.0%"},
                                "plus_ckdpc": {"cutoff": 9.04, "sens": "72.7%", "spec": "80.5%"}
                            },
                            "DeepSeek-V3.1": {
                                "base": {"cutoff": 12.00, "sens": "56.0%", "spec": "82.5%"},
                                "plus_ckdpc": {"cutoff": 11.43, "sens": "71.7%", "spec": "80.5%"}
                            },
                            "GPT-OSS-120b": {
                                "base": {"cutoff": 26.50, "sens": "67.6%", "spec": "80.5%"},
                                "plus_ckdpc": {"cutoff": 18.70, "sens": "70.2%", "spec": "80.5%"}
                            },
                            "GLM-4.5-air": {
                                "base": {"cutoff": 27.50, "sens": "55.6%", "spec": "78.9%"},
                                "plus_ckdpc": {"cutoff": 19.48, "sens": "66.6%", "spec": "79.5%"}
                            }
                        }

                        # --- 3. 构建展示逻辑 ---

                        # 场景 A: 用户输入了 CKD-PC 评分
                        if ckd_pc_score > 0:
                            st.success(f"✅ Joint model analysis mode has been activated. (CKD-PC Score: {ckd_pc_score})")

                            # 如果成功提取到了 LLM 的分数，显示计算过程
                            if llm_5y_score is not None:
                                # 核心计算：简单加权平均
                                combined_score = (llm_5y_score + ckd_pc_score) / 2

                                # 展示计算结果卡片
                                c1, c2, c3 = st.columns(3)
                                c1.metric("LLM predict (5 years)", f"{llm_5y_score:.2f}%")
                                c2.metric("ckd_pc_score", f"{ckd_pc_score:.2f}%")
                                c3.metric(
                                    "📊 Your Joint Score (Simple Average)",
                                    f"{combined_score:.2f}%",
                                    delta="High Risk" if combined_score > 9.04 else "Low/Med Risk",  # 仅以 Kimi 为例显示颜色
                                    delta_color="inverse",
                                    help=f"formula({llm_5y_score} + {ckd_pc_score}) / 2 = {combined_score}"
                                )
                                st.caption("Note: Delta color indicators are for reference only, based on the Kimi-K2 threshold—please see the model details below.")
                            else:
                                st.warning(
                                    "⚠️ Unable to auto-extract '5_year_risk_percent' from the LLM; please manually calculate the combined score: (LLM score + CKD‐PC) / 2")
                                combined_score = None

                            st.markdown("---")
                            st.markdown("#### 🔬 Combined Model Cutoffs")

                            cols = st.columns(4)
                            models = list(ref_data.keys())

                            for i, model_name in enumerate(models):
                                with cols[i]:
                                    data = ref_data[model_name]["plus_ckdpc"]
                                    cutoff_val = data['cutoff']

                                    # 判断风险状态
                                    risk_status = "N/A"
                                    status_color = "gray"
                                    if combined_score is not None:
                                        if combined_score > cutoff_val:
                                            risk_status = "🔴 High Risk"
                                            status_color = "red"
                                        else:
                                            risk_status = "🟢 Low/Med Risk"
                                            status_color = "green"

                                    st.markdown(f"**{model_name}**")
                                    st.markdown(f"Threshold: `> {cutoff_val}`")
                                    if combined_score is not None:
                                        st.markdown(f":{status_color}[{risk_status}]")

                                    st.caption(f"Sens: {data['sens']} | Spec: {data['spec']}")

                        # 场景 B: 仅显示基础 LLM 参考
                        else:
                            st.info("ℹ️ The current display shows the reference standard for the [Base Large Model]. Enter the CKD-PC score above to view the joint model calculation results.")

                            if llm_5y_score is not None:
                                st.metric("LLM prediction (5years)", f"{llm_5y_score:.2f}%")

                            st.markdown("---")
                            cols = st.columns(4)
                            models = list(ref_data.keys())

                            for i, model_name in enumerate(models):
                                with cols[i]:
                                    data = ref_data[model_name]["base"]
                                    cutoff_val = data['cutoff']

                                    # 判断风险状态
                                    risk_status = ""
                                    if llm_5y_score is not None:
                                        if llm_5y_score > cutoff_val:
                                            risk_status = "🔴 High Risk"
                                        else:
                                            risk_status = "🟢 Low/Med"

                                    st.markdown(f"**{model_name}**")
                                    st.markdown(f"Cutoff: `> {cutoff_val}` {risk_status}")
                                    st.caption(f"Sens: {data['sens']} | Spec: {data['spec']}")

                        # --- 4. 修正后的免责与说明备注 ---
                        st.warning(
                            "⚠️ **Reference Note**:"
                            "1. **Combined Model Definition**: In this system, the combined risk score is calculated using a **simple weighted average method**, i.e.:"
                            "   $$\\text{Combined Score} = \\frac{\\text{LLM 5-Year Risk} + \\text{CKD-PC Score}}{2}$$\n"
                            "2. **Validation Source**: The high‑risk thresholds (top 25%) and their sensitivity/specificity parameters above are based on large‑scale retrospective validation in the UK Biobank (UKBB) population (median follow‑up of 16.2 years)."
                            "3. **Usage Restrictions**: This data is provided only for clinical qualitative grading reference and should not be used as a sole basis for diagnosis."
                        )

                    with col_res2:
                        st.subheader("🔍 Prompt structure Preview")
                        st.info("The system dynamically constructed the following Prompt structure based on your input.:")
                        if report_memory_content and report_upload_content:
                            st.write("✅ Synthesis")
                        elif report_memory_content:
                            st.write("✅ Application")
                        elif report_upload_content:
                            st.write("✅ Guideline")
                        else:
                            st.write("⚠️ General Knowledge")

                        with st.expander("viewFull Prompt"):
                            st.text(final_prompt_str)

                except Exception as e:
                    st.error(f"anErrorOccurredDuringReasoningE")