# import pandas as pd
# import numpy as np
# import random 
# import os
# from google import genai
# from google.genai.errors import APIError 
# from google.genai.types import GenerationConfig 
# # The following imports from your original code are kept for context but not strictly used 
# # in the final execution flow below:
# # from sklearn.model_selection import train_test_split
# # from transformers import AutoTokenizer
# # from datasets import Dataset 

# # --- 1. DATA LOADING ---
# # NOTE: arg_quality_rank_30k.csv, train.csv, and test.csv are loaded but not used 
# # in the core debate report generation, only debate_speeches_df is needed.
# debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv") 

# print("--- Data Loading Complete ---")
# print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")

# # --- 2. ARGUMENT QUALITY SCORER (MOCK CLASS) ---

# class ArgumentQualityScorer:
#     """Mock class simulating a trained model predicting argument quality (WA)."""
#     def __init__(self, model_path=None): pass
#     def score(self, text: str) -> float:
#         # Score remains mocked based on text length for simulation
#         score = min(1.0, len(text) / 200) 
#         return round(score, 3)
# quality_scorer = ArgumentQualityScorer()

# # --- DYNAMIC STANCE CLASSIFIER CLASS (Topic-Agnostic) ---
# class StanceClassifier:
#     """
#     A purely general, topic-agnostic classifier that predicts stance based on 
#     explicit debate phrases.
#     """
#     def __init__(self): pass
    
#     def predict_stance_general(self, speech_text: str) -> str:
#         text = speech_text.lower()

#         # High-confidence PRO phrases (i.e., supporting the motion)
#         if any(phrase in text for phrase in ['we advocate', 'we support', 'we stand for', 'our position is pro']):
#             return 'PRO'
        
#         # High-confidence CON phrases (i.e., opposing the motion)
#         if any(phrase in text for phrase in ['we oppose', 'we stand against', 'we do not support', 'our position is con']):
#             return 'CON'
        
#         # Fallback to random choice
#         return random.choice(['PRO', 'CON']) 

# stance_classifier = StanceClassifier()

# # --- DYNAMIC DEBATE STRUCTURE GENERATOR ---

# def generate_debate_report_dynamic(speeches_df: pd.DataFrame) -> tuple:
#     """
#     Constructs the report using a random debate, and dynamic rebuttal analysis 
#     based on turn order.
#     """
#     if speeches_df.empty:
#         return [{"Speech_ID": 1, "Side": "N/A", "Topic": "N/A", "Argument_Quality_Score": 0.0, 
#                  "Rebuttal_Analysis": "Data loading failed.", "Snippet": "..."}], "UNKNOWN"

#     # 1. DYNAMIC SELECTION: Pick a random topic ID
#     available_topics = speeches_df['topic_id'].unique()
#     if len(available_topics) == 0:
#         return [], "UNKNOWN"
                 
#     random_debate_id = random.choice(available_topics)
    
#     # FIX: Filter by topic_id and rely on the inherent row order (safe fix for KeyError)
#     filtered_speeches = speeches_df[speeches_df['topic_id'] == random_debate_id]
#     debate_speeches = filtered_speeches.reset_index(drop=True)
    
#     if debate_speeches.empty: # Safety check
#         return [], "UNKNOWN" 
        
#     target_topic = debate_speeches['topic'].iloc[0]
#     report = []
    
#     # Process the first 5 speeches to establish flow
#     for i in range(min(5, len(debate_speeches))):
#         speech = debate_speeches.iloc[i]
#         speech_text = speech['text']
#         speech_id = i + 1
        
#         # 2. DYNAMIC SIDE ASSIGNMENT: Relying on the general StanceClassifier
#         side = stance_classifier.predict_stance_general(speech_text)

#         # 3. Argument Scoring (Mocked)
#         quality_score = quality_scorer.score(speech_text)
        
#         # 4. DYNAMIC REBUTTAL ANALYSIS: Based on Turn Flow
#         rebuttal_note = ""
        
#         if i == 0:
#             rebuttal_note = "Opening Constructive Argument (Initial Thesis) from the first side."
#         elif i == 1:
#             # We reference the side of the first speaker to make the rebuttal note specific
#             rebuttal_note = f"Opening Constructive/Direct Rebuttal to Speech 1 (Side: {report[0]['Side']})."
#         elif i == 2:
#             rebuttal_note = f"Defense/Consolidation and Counter-Rebuttal to Speech 2."
#         elif i == 3:
#             rebuttal_note = f"Second Line of Attack/Rebuttal to Speech 3."
#         elif i == 4:
#             rebuttal_note = "Final Summary/Closing Statement."

#         report.append({
#             "Speech_ID": speech_id,
#             "Side": side,
#             "Topic": target_topic,
#             "Argument_Quality_Score": round(quality_score, 3), 
#             "Rebuttal_Analysis": rebuttal_note,
#             "Snippet": speech_text[:150].replace('\n', ' ') + "..."
#         })
        
#     return report, target_topic

# # --- FINAL JUDGE PROMPT GENERATOR (UNCHANGED) ---
# def generate_final_judge_prompt(debate_report: list, overall_topic: str) -> str:
#     report_text = "\n".join([
#         f"--- Speech {s['Speech_ID']} ({s['Side']} on '{s['Topic']}'): "
#         f"Quality Score: {s['Argument_Quality_Score']:.3f}. Rebuttal: {s['Rebuttal_Analysis']}\n"
#         f"Content Snippet: {s['Snippet']}"
#         for s in debate_report
#     ])
    
#     prompt = f"""
#     You are an expert, impartial debate judge. Your task is to analyze the debate flow based on the technical report provided below and issue a comprehensive verdict.

#     The motion being debated is: **{overall_topic}**

#     Your analysis must strictly adhere to the following structure and criteria:

#     1.  **VERDICT:** State the winning side (**PRO** or **CON**) clearly.
#     2.  **REASON FOR DECISION (RFD):** Provide a detailed analysis (3-4 paragraphs) explaining:
#         * The central clash or point of contention in the debate.
#         * Why the winning side successfully fulfilled its burden of proof and/or demonstrated greater impact.
#         * Why the losing side failed to adequately rebut or mitigate the winner's best arguments.
#         * Use the Argument Quality Score and Rebuttal Analysis data points where appropriate (e.g., "The CON side had consistently higher-rated initial arguments...").
#     3.  **CONSTRUCTIVE FEEDBACK:** Provide 2-3 specific, actionable points for both the PRO and CON teams.

#     --- TECHNICAL DEBATE REPORT ---
#     {report_text}
#     """
#     return prompt

# # --- GEMINI API INTEGRATION AND EXECUTION ---

# client = None

# def initialize_gemini_client():
#     global client
#     try:
#         # Initialize the client. This relies on the GEMINI_API_KEY environment variable.
#         client = genai.Client()
#         print("\nGemini Client initialized successfully.")
#         return True
#     except Exception as e:
#         print(f"\n[CRITICAL ERROR] Failed to initialize Gemini Client. Details: {e}")
#         return False
        
# def get_gemini_judgment(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
#     global client
#     if not client:
#         return "[API EXECUTION FAILED] Client was not initialized."
    
#     try:
#         response = client.models.generate_content(
#             model=model_name,
#             contents=prompt,
#         )
#         return response.text
#     except Exception as e:
#         # Handle cases where the API key is invalid or an internal error occurs
#         return f"\n[API ERROR] An error occurred while generating content: {e}"

# # --- FINAL EXECUTION FLOW ---

# # 1. Use the NEW, DYNAMIC function
# debate_report, overall_topic = generate_debate_report_dynamic(debate_speeches_df)
# final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

# print("\n\n" + "="*50)
# print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
# print("="*50)
# print(final_prompt)

# if initialize_gemini_client():
#     print(f"\n-> Requesting Judgment for topic '{overall_topic}' from gemini-2.5-pro...")
#     final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

#     print("\n" + "="*50)
#     print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
#     print("="*50)
#     print(final_judgment_text)
# else:
#     print("\n[CRITICAL FAILURE] Gemini API client could not initialize. Aborting final judgment.")







import pandas as pd
import numpy as np
import random 
import os
from google import genai
from google.genai.errors import APIError 
from google.genai.types import GenerationConfig 
# The following imports from your original code are kept for context:
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer
# from datasets import Dataset 


#JUST HAVE ONE PARAMEER FOR ARGUMENTT MINIG WHICH IS LENGTH

# --- 1. DATA LOADING ---
# NOTE: Only debate_speeches_df is strictly necessary for the core pipeline flow.
debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv") 

print("--- Data Loading Complete ---")
print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")

# --- 2. ARGUMENT QUALITY SCORER (MOCK CLASS) ---

class ArgumentQualityScorer:
    """Mock class simulating a trained model predicting argument quality (WA)."""
    def __init__(self, model_path=None): pass
    def score(self, text: str) -> float:
        # Score remains mocked based on text length for simulation
        score = min(1.0, len(text) / 200) 
        return round(score, 3)
quality_scorer = ArgumentQualityScorer()

# --- DYNAMIC STANCE CLASSIFIER CLASS (Topic-Agnostic) ---
class StanceClassifier:
    """
    A purely general, topic-agnostic classifier that predicts stance based on 
    explicit debate phrases.
    """
    def __init__(self): pass
    
    def predict_stance_general(self, speech_text: str) -> str:
        text = speech_text.lower()

        # High-confidence PRO phrases (i.e., supporting the motion)
        if any(phrase in text for phrase in ['we advocate', 'we support', 'we stand for', 'our position is pro']):
            return 'PRO'
        
        # High-confidence CON phrases (i.e., opposing the motion)
        if any(phrase in text for phrase in ['we oppose', 'we stand against', 'we do not support', 'our position is con']):
            return 'CON'
        
        # Fallback to random choice
        return random.choice(['PRO', 'CON']) 

stance_classifier = StanceClassifier()

# --- DYNAMIC DEBATE STRUCTURE GENERATOR (UPDATED TO INCLUDE ALL SPEECHES) ---

def generate_debate_report_dynamic(speeches_df: pd.DataFrame) -> tuple:
    """
    Constructs the report using a random debate, and dynamic rebuttal analysis 
    based on the full turn order of ALL speeches.
    """
    if speeches_df.empty:
        return [{"Speech_ID": 1, "Side": "N/A", "Topic": "N/A", "Argument_Quality_Score": 0.0, 
                 "Rebuttal_Analysis": "Data loading failed.", "Snippet": "..."}], "UNKNOWN"

    # 1. DYNAMIC SELECTION: Pick a random topic ID
    available_topics = speeches_df['topic_id'].unique()
    if len(available_topics) == 0:
        return [], "UNKNOWN"
                 
    random_debate_id = random.choice(available_topics)
    
    # FIX: Filter by topic_id and rely on the inherent row order (safe fix for KeyError)
    filtered_speeches = speeches_df[speeches_df['topic_id'] == random_debate_id]
    debate_speeches = filtered_speeches.reset_index(drop=True)
    
    if debate_speeches.empty: # Safety check
        return [], "UNKNOWN" 
        
    target_topic = debate_speeches['topic'].iloc[0]
    report = []
    total_speeches = len(debate_speeches) # Get total length here

    # Process ALL speeches for the selected debate
    for i in range(total_speeches):
        speech = debate_speeches.iloc[i]
        speech_text = speech['text']
        speech_id = i + 1
        
        # 2. DYNAMIC SIDE ASSIGNMENT: Relying on the general StanceClassifier
        side = stance_classifier.predict_stance_general(speech_text)

        # 3. Argument Scoring (Mocked)
        quality_score = quality_scorer.score(speech_text)
        
        # 4. DYNAMIC REBUTTAL ANALYSIS: Based on Full Turn Flow
        rebuttal_note = ""
        
        if i == 0:
            rebuttal_note = "Opening Constructive Argument (Initial Thesis) from the first side."
        elif speech_id == total_speeches: # The very last speech is the closing statement
            rebuttal_note = "Final Summary/Closing Statement."
        elif i == 1:
            rebuttal_note = f"Direct Rebuttal/Second Constructive to Speech 1 (Side: {report[0]['Side']})."
        else:
            # All other speeches are mid-debate rebuttals or extensions
            rebuttal_note = "Mid-debate Rebuttal, Consolidation, or Extension."

        report.append({
            "Speech_ID": speech_id,
            "Side": side,
            "Topic": target_topic,
            "Argument_Quality_Score": round(quality_score, 3), 
            "Rebuttal_Analysis": rebuttal_note,
            "Snippet": speech_text[:150].replace('\n', ' ') + "..."
        })
        
    return report, target_topic

# --- FINAL JUDGE PROMPT GENERATOR (UNCHANGED) ---
def generate_final_judge_prompt(debate_report: list, overall_topic: str) -> str:
    report_text = "\n".join([
        f"--- Speech {s['Speech_ID']} ({s['Side']} on '{s['Topic']}'): "
        f"Quality Score: {s['Argument_Quality_Score']:.3f}. Rebuttal: {s['Rebuttal_Analysis']}\n"
        f"Content Snippet: {s['Snippet']}"
        for s in debate_report
    ])
    
    prompt = f"""
    You are an expert, impartial debate judge. Your task is to analyze the debate flow based on the technical report provided below and issue a comprehensive verdict.

    The motion being debated is: **{overall_topic}**

    Your analysis must strictly adhere to the following structure and criteria:

    1.  **VERDICT:** State the winning side (**PRO** or **CON**) clearly.
    2.  **REASON FOR DECISION (RFD):** Provide a detailed analysis (3-4 paragraphs) explaining:
        * The central clash or point of contention in the debate.
        * Why the winning side successfully fulfilled its burden of proof and/or demonstrated greater impact.
        * Why the losing side failed to adequately rebut or mitigate the winner's best arguments.
        * Use the Argument Quality Score and Rebuttal Analysis data points where appropriate (e.g., "The CON side had consistently higher-rated initial arguments...").
    3.  **CONSTRUCTIVE FEEDBACK:** Provide 2-3 specific, actionable points for both the PRO and CON teams.

    --- TECHNICAL DEBATE REPORT ---
    {report_text}
    """
    return prompt

# --- GEMINI API INTEGRATION AND EXECUTION ---

client = None

def initialize_gemini_client():
    global client
    try:
        # Initialize the client. This relies on the GEMINI_API_KEY environment variable.
        client = genai.Client()
        print("\nGemini Client initialized successfully.")
        return True
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize Gemini Client. Details: {e}")
        return False
        
def get_gemini_judgment(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
    global client
    if not client:
        return "[API EXECUTION FAILED] Client was not initialized."
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"\n[API ERROR] An error occurred while generating content: {e}"

# --- FINAL EXECUTION FLOW ---

# 1. Use the NEW, DYNAMIC function
debate_report, overall_topic = generate_debate_report_dynamic(debate_speeches_df)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)

if initialize_gemini_client():
    print(f"\n-> Requesting Judgment for topic '{overall_topic}' from gemini-2.5-pro...")
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\n[CRITICAL FAILURE] Gemini API client could not initialize. Aborting final judgment.")