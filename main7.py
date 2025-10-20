import pandas as pd
import numpy as np
import random 
import os
# FIX: Adjusted import to be more reliable in execution environment
import google.genai as genai
from google.genai.errors import APIError 
from google.genai.types import GenerationConfig 

# --- 1. DATA LOADING ---
debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv") 

print("--- Data Loading Complete ---")
print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")

# --- 2. ADVANCED ARGUMENT QUALITY SCORER (Feature-Engineered Simulation) ---

class ArgumentQualityScorer:
    """
    Simulates a trained model's quality prediction using a weighted combination 
    of engineered features (features that a real ML model would be trained on), 
    as simple length is insufficient.
    """
    def __init__(self, model_path=None): pass
    
    def score(self, text: str) -> float:
        # Feature 1: Development/Fluency (Word Count, capped at 400 words)
        word_count = len(text.split())
        score_development = min(1.0, word_count / 400.0)
        
        # Feature 2: Structure/Clarity (Paragraph Density)
        paragraph_count = text.count('\n\n') + 1
        # Normalize density: penalty for having too many short paragraphs (lowers score)
        score_structure = max(0.0, 1.0 - (paragraph_count / (word_count / 100 + 1))) 
        
        # Feature 3: Simulated Complexity (Long Word Ratio, > 6 chars)
        long_words = [w for w in text.split() if len(w) > 6]
        long_word_ratio = len(long_words) / (word_count + 1e-6)
        # Assume complexity contributes positively, capped.
        score_complexity = min(0.5, long_word_ratio * 1.5)

        # Final Weighted Score (mimicking regression weights w1=0.4, w2=0.3, w3=0.3)
        final_score = (
            (score_development * 0.4) + 
            (score_structure * 0.3) + 
            (score_complexity * 0.3)
        )
        
        # Clamp the score between 0.0 and 1.0
        return round(np.clip(final_score, 0.0, 1.0), 3)

quality_scorer = ArgumentQualityScorer()
print(f"Argument Quality Score is now calculated using 3 weighted features.")

# --- DYNAMIC STANCE CLASSIFIER CLASS (Rule-Based Simulation) ---
class StanceClassifier:
    """
    A rule-based classifier that predicts stance based on explicit debate phrases.
    True training is omitted as it is environment-prohibitive.
    """
    def __init__(self): pass
    
    def predict_stance_general(self, speech_text: str) -> str:
        text = speech_text.lower()
        if any(phrase in text for phrase in ['we advocate', 'we support', 'we stand for', 'our position is pro']):
            return 'PRO'
        if any(phrase in text for phrase in ['we oppose', 'we stand against', 'we do not support', 'our position is con']):
            return 'CON'
        return random.choice(['PRO', 'CON']) 

stance_classifier = StanceClassifier()

# --- DYNAMIC DEBATE STRUCTURE GENERATOR (Processes ALL Speeches) ---

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
    
    # Filter by topic_id and rely on the inherent row order (safe fix for KeyError)
    filtered_speeches = speeches_df[speeches_df['topic_id'] == random_debate_id]
    debate_speeches = filtered_speeches.reset_index(drop=True)
    
    if debate_speeches.empty: # Safety check
        return [], "UNKNOWN" 
        
    target_topic = debate_speeches['topic'].iloc[0]
    report = []
    total_speeches = len(debate_speeches) 

    # Process ALL speeches for the selected debate
    for i in range(total_speeches):
        speech = debate_speeches.iloc[i]
        speech_text = speech['text']
        speech_id = i + 1
        
        # 2. DYNAMIC SIDE ASSIGNMENT
        side = stance_classifier.predict_stance_general(speech_text)

        # 3. ADVANCED Argument Scoring
        quality_score = quality_scorer.score(speech_text)
        
        # 4. DYNAMIC REBUTTAL ANALYSIS: Based on Full Turn Flow
        rebuttal_note = ""
        
        if i == 0:
            rebuttal_note = "Opening Constructive Argument (Initial Thesis) from the first side."
        elif speech_id == total_speeches: # The very last speech is the closing statement
            rebuttal_note = "Final Summary/Closing Statement."
        elif i == 1:
            # We reference the side of the first speaker to make the rebuttal note specific
            if report:
                 rebuttal_note = f"Direct Rebuttal/Second Constructive to Speech 1 (Side: {report[0]['Side']})."
            else:
                 rebuttal_note = "Direct Rebuttal/Second Constructive."
        else:
            # All other speeches are mid-debate rebuttals or extensions
            rebuttal_note = "Mid-debate Rebuttal, Consolidation, or Extension."

        report.append({
            "Speech_ID": speech_id,
            "Side": side,
            "Topic": target_topic,
            "Argument_Quality_Score": quality_score, 
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

# Use the correct way to initialize the client
client = None

def initialize_gemini_client():
    global client
    try:
        # Assuming genai is imported correctly
        client = genai.Client()
        print("\nGemini Client initialized successfully.")
        return True
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize Gemini Client. Details: {e}")
        # Note: If API key is not set, this will fail, which is expected behavior 
        # for systems requiring environment variables.
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
        # Log the full exception for debugging
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
