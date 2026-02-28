from google import genai
from google.genai import types
import pathlib
import os

# --- Configuration Section ---
# Please make sure to replace these with your actual configurations

# 1. Configure your API Key
# Method 1: Set environment variable GOOGLE_API_KEY (Recommended)
# os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY_HERE'
# Method 2: Configure directly in code (Note security risks; do not include real Key when uploading to public repos like GitHub)
API_KEY = 'YOUR_API_KEY_HERE'  # <--- Replace your API Key here

# 2. Define PDF input and TXT output directories
# TODO: Update the following paths to your actual local dataset and output paths
PDF_INPUT_DIR = './data/pdf_input'  # <--- Replace with your PDF files directory
TXT_OUTPUT_DIR = './output/txt_output' # <--- Replace with your desired TXT output directory

# 3. Use the model name specified in the image
MODEL_NAME_FROM_IMAGE = 'gemini-2.5-pro-preview-05-06' # Model name from your image

# 4. Use the prompt specified in the image or your custom prompt

PROMPT_TO_USE = '''
  "You are an expert biomedical AI assistant specialized in extracting structured information from medical research literature. Your primary task is to carefully read the provided PDF document (full text) and identify and extract knowledge pertinent or contributing to **predicting the risk of Chronic Kidney Disease (CKD) in individuals with Non-alcoholic Fatty Liver Disease (NAFLD), Metabolic dysfunction-Associated Fatty Liver Disease (MAFLD), or Metabolic dysfunction-Associated Steatotic Liver Disease (MASLD)**. Focus on extracting: * Risk factors for CKD in the context of NAFLD/MAFLD/MASLD. * Methods, models, or specific calculations used for CKD risk prediction in this population. * Biological mechanisms or pathways that are implicated in the progression from NAFLD/MAFLD/MASLD to CKD and could inform risk. * Interventions that might modify this risk. * Relevant study context, population characteristics, and specific outcomes. All extracted information must be directly supported by the provided text. The output MUST be a single, valid JSON object."


    "Please provide the output in a single, valid JSON object with the following top-level keys: 'document_metadata', 'study_details', 'population_characteristics', 'extracted_entities', and 'extracted_relationships'. If specific information for an attribute is not found in the text, use null, an empty string "", or an empty list [] as appropriate for that attribute's value. For all descriptive fields, provide concise summaries based on the paper's content."




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

# --- Configuration End ---

def ensure_dir_exists(directory_path: str):
    """Check if directory exists, create it if it does not."""
    path = pathlib.Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured directory exists: {path}") # Optional log output

def process_pdf_strictly_per_image_method(
    pdf_path: pathlib.Path,
    output_txt_path: pathlib.Path,
    client: genai.Client, # Pass in the genai.Client() instance
    model_name: str,
    prompt: str
):
    """
    Process a single PDF file strictly according to the method in the image.
    """
    print(f"\n[INFO] Processing: {pdf_path.name}, using model: '{model_name}'")
    try:
        # 1. Read PDF file bytes
        pdf_file_bytes = pdf_path.read_bytes()

        # 2. Prepare PDF file Part, strictly using types.Part.from_bytes
        pdf_file_part = types.Part.from_bytes(
            mime_type='application/pdf', # MIME type for PDF
            data=pdf_file_bytes
        )

        # 3. Prepare content list to send to API
        contents_for_api = [pdf_file_part, prompt]

        # 4. Strictly call client.models.generate_content(...) as in the image
        print(f"  Attempting to process {pdf_path.name} via client.models.generate_content API call...")
        response = client.models.generate_content(
            model=model_name,
            contents=contents_for_api
        )

        # 5. Save response text
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"  [SUCCESS] Content generated successfully and saved to: {output_txt_path.name}")

    except AttributeError as ae:
        error_message = (
            f"  [ERROR] Processing file {pdf_path.name} caused AttributeError: {ae}\n"
            f"  This usually means the 'google-generativeai' SDK version in your current Python environment,\n"
            f"  does not have a 'models' attribute on the 'genai.Client()' object, or no 'generate_content' method on the 'client.models' object,\n"
            f"  or the arguments for this method do not match the call in the script.\n"
            f"  Please confirm your SDK version and environment exactly match the original context of the code snippet in your image.\n"
            f"  For example, whether the model name '{model_name}' requires a special prefix (like 'models/{model_name}') or if this calling method is for a specific API version.\n"
        )
        print(error_message)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(error_message)
    except Exception as e:
        error_message = f"  [ERROR] Unexpected error while processing file {pdf_path.name}: {type(e).__name__} - {e}"
        print(error_message)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(error_message)

def main_batch_processor():
    """
    Main function for batch processing PDFs.
    """
    # Configure API Key
    if API_KEY == 'YOUR_API_KEY_HERE' and not os.getenv('GOOGLE_API_KEY'):
        print("[ERROR] API_KEY not configured. Please set API_KEY at the top of the script or set the GOOGLE_API_KEY environment variable.")
        return

    if API_KEY != 'YOUR_API_KEY_HERE':
        print("[INFO] API Key configured via in-script variable.")
    elif os.getenv('GOOGLE_API_KEY'):
        # SDK will automatically use the environment variable, usually no need to call genai.configure() additionally
        print("[INFO] API Key will be automatically configured via GOOGLE_API_KEY environment variable.")
    # If neither is set, genai operations will fail

    # Check path configurations
    if PDF_INPUT_DIR == 'path/to/your/pdf_files' or TXT_OUTPUT_DIR == 'path/to/your/output_txt_files':
        print("[ERROR] PDF_INPUT_DIR or TXT_OUTPUT_DIR is not configured correctly. Please modify the paths at the top of the script.")
        return

    input_dir = pathlib.Path(PDF_INPUT_DIR)
    output_dir = pathlib.Path(TXT_OUTPUT_DIR)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory '{input_dir}' does not exist or is not a valid directory.")
        return

    ensure_dir_exists(str(output_dir))

    # Strictly initialize client = genai.Client() as in the image
    print("[INFO] Initializing genai.Client()...")
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize genai.Client(): {e}")
        print("        Please ensure 'google-generativeai' library is installed correctly and API Key is configured.")
        return

    print(f"\n[INFO] Starting batch processing of PDF files from directory '{input_dir}'...")
    print(f"         Output will be saved to directory: '{output_dir}'")
    print(f"         Using model (from image): '{MODEL_NAME_FROM_IMAGE}'")
    print(f"         Using prompt: '{PROMPT_TO_USE}'")
    print("-" * 50)

    pdf_files_lower = list(input_dir.glob('*.pdf'))
    pdf_files_upper = list(input_dir.glob('*.PDF'))
    pdf_files_found = pdf_files_lower + pdf_files_upper

    if not pdf_files_found:
        print(f"[INFO] No PDF files found in directory '{input_dir}'.")
        return

    print(f"[INFO] Found {len(pdf_files_found)} PDF files to process.")

    for pdf_file_path in pdf_files_found:
        # Name the output txt file
        output_txt_filename = pdf_file_path.stem + '.txt'
        output_txt_file_path = output_dir / output_txt_filename

        process_pdf_strictly_per_image_method(
            pdf_path=pdf_file_path,
            output_txt_path=output_txt_file_path,
            client=client, # Pass client instance
            model_name=MODEL_NAME_FROM_IMAGE,
            prompt=PROMPT_TO_USE
        )

    print("-" * 50)
    print("[INFO] All PDF file processing workflows finished.")

if __name__ == '__main__':
    main_batch_processor()