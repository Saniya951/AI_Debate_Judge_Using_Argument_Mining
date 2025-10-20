import pandas as pd
import numpy as np
import random 
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from google import genai
from google.genai.errors import APIError 
from google.genai.types import GenerationConfig 

# $env:GEMINI_API_KEY="AIzaSyCQFC54KWvoGRwXIIzUN-z-0_TXGM_onzI"
# USED StanceClassifier FOR PREDICTING PROS ND CONS BASED ON KEYWORDS ND STRUCTURE
# topic is hardcoded here - ban cosmetic surgery

# --- 1. DATA LOADING ---
debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv") 
arg_quality_df = pd.read_csv("arg_quality_rank_30k.csv") 
evidence_train_df = pd.read_csv("train.csv")
evidence_test_df = pd.read_csv("test.csv")
evidence_df = pd.concat([evidence_train_df, evidence_test_df], ignore_index=True)

print("--- Data Loading Complete ---")
print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")

# --- 2. ARGUMENT QUALITY SCORER (MOCK CLASS) ---
arg_quality_df = arg_quality_df.dropna(subset=['argument', 'WA'])
arg_quality_df['labels'] = arg_quality_df['WA'].astype(float) 
MODEL_NAME = "roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ArgumentQualityScorer:
    """Mock class simulating a trained model predicting argument quality (WA)."""
    def __init__(self, model_path=None): pass
    def score(self, text: str) -> float:
        score = min(1.0, len(text) / 200) 
        return round(score, 3)
quality_scorer = ArgumentQualityScorer()

# --- NEW STANCE CLASSIFIER CLASS ---
class StanceClassifier:
    """
    Mocks a trained ML model that determines if a text supports or opposes a topic.
    In a real project, this would be trained on arg_quality_rank_30k stance_WA labels.
    """
    def __init__(self): pass
    
    def predict_stance(self, speech_text: str, topic: str) -> str:
        # Example of Content-Based Logic:
        # 1. Identify key PRO and CON words for the topic "ban cosmetic surgery"
        if "ban cosmetic surgery" in topic.lower():
            
            # Words supporting the BAN (PRO)
            if any(word in speech_text.lower() for word in ['harm', 'unethical', 'pressure', 'ban', 'abolish']):
                return 'PRO'
            
            # Words opposing the BAN (CON)
            if any(word in speech_text.lower() for word in ['safe', 'choice', 'autonomy', 'regulation', 'freedom']):
                return 'CON'
        
        # 2. Fallback to general debate logic if keywords aren't found
        # (This is still better than alternating, as it looks at the text)
        if "we oppose" in speech_text.lower() or "we stand against" in speech_text.lower():
             return 'CON'
        if "we advocate" in speech_text.lower() or "we support" in speech_text.lower():
            return 'PRO'

        # Default fallback (simulating the model being unsure)
        return random.choice(['PRO', 'CON']) 

stance_classifier = StanceClassifier()

# --- 4. DEBATE STRUCTURE GENERATOR (FIXED LOGIC) ---

def generate_debate_report_real(speeches_df: pd.DataFrame) -> tuple:
    """
    Constructs the report using actual speech content and Stance Classification 
    to assign the PRO/CON labels, removing the manual alternation.
    """
    if speeches_df.empty:
        return [{"Speech_ID": 1, "Side": "N/A", "Topic": "N/A", "Argument_Quality_Score": 0.0, 
                 "Rebuttal_Analysis": "Data loading failed.", "Snippet": "..."}], "UNKNOWN"

    # 1. Select the single target debate: topic_id 1161 ("We should ban cosmetic surgery")
    debate_id = 1161 
    debate_speeches = speeches_df[speeches_df['topic_id'] == debate_id].reset_index(drop=True)
    target_topic = debate_speeches['topic'].iloc[0]
    
    report = []
    
    # Process the first 5 speeches
    for i in range(min(5, len(debate_speeches))):
        speech = debate_speeches.iloc[i]
        speech_text = speech['text']
        
        # 2. ***FIX: Use Stance Classification to get the side***
        side = stance_classifier.predict_stance(speech_text, target_topic)
        speech_id = i + 1
        
        # 3. Argument Scoring (Mocked: Uses the length of the real speech text)
        quality_score = quality_scorer.score(speech_text)
        
        # 4. Rebuttal Analysis (Simulated based on order—the only non-content part of flow)
        rebuttal_note = "Opening Constructive Argument (Initial Thesis)."
        if i == 1:
            rebuttal_note = f"Direct Rebuttal to Speech 1 (Stance: {report[0]['Side']})."
        elif i == 2:
            rebuttal_note = f"Defense/Consolidation and Counter-Rebuttal to Speech 2."
        elif i == 3:
            rebuttal_note = f"Second Line of Attack/Rebuttal to Speech 3."
        elif i == 4:
            rebuttal_note = "Final Summary/Closing Statement."

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
    # ... (function body remains the same) ...
    report_text = "\n".join([
        f"--- Speech {s['Speech_ID']} ({s['Side']} on '{s['Topic']}'): "
        f"Quality Score: {s['Argument_Quality_Score']:.3f}. Rebuttal: {s['Rebuttal_Analysis']}\n"
        f"Content Snippet: {s['Snippet']}"
        for s in debate_report
    ])
    
    prompt = f"""
    You are an expert, impartial debate judge. Your task is to analyze the debate flow based on the technical report provided below and issue a comprehensive verdict.
    ... (rest of the prompt)
    """
    return prompt

# --- GEMINI API INTEGRATION AND EXECUTION ---

client = None

def initialize_gemini_client():
    global client
    try:
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
        return f"\n[API ERROR] An error occurred: {e}"

# --- FINAL EXECUTION FLOW ---

# 1. Use the NEW, FIXED function
debate_report, overall_topic = generate_debate_report_real(debate_speeches_df)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)

if initialize_gemini_client():
    print("\n-> Requesting Judgment from gemini-2.5-pro...")
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\n[CRITICAL FAILURE] Gemini API client could not initialize. Aborting final judgment.")