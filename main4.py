import pandas as pd
import numpy as np
import random 
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from google import genai
from google.genai.errors import APIError 
from google.genai.types import GenerationConfig # Note: This is removed in the execution flow for stability


# ALTERNATE PRO /CONS LABELING

# --- 1. DATA LOADING (UPDATED) ---
# Load the new full debate speeches dataset
debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv") 

# Load local files (used for mock scoring preparation)
arg_quality_df = pd.read_csv("arg_quality_rank_30k.csv") 
evidence_train_df = pd.read_csv("train.csv")
evidence_test_df = pd.read_csv("test.csv")
evidence_df = pd.concat([evidence_train_df, evidence_test_df], ignore_index=True)

print("--- Data Loading Complete ---")
print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")
print(f"Arg Quality Rank size: {len(arg_quality_df)}")

# --- 2. ARGUMENT QUALITY SCORER (MOCK CLASS - UNCHANGED) ---
arg_quality_df = arg_quality_df.dropna(subset=['argument', 'WA'])
arg_quality_df['labels'] = arg_quality_df['WA'].astype(float) 
MODEL_NAME = "roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ArgumentQualityScorer:
    """Mock class simulating a trained model predicting argument quality (WA)."""
    def __init__(self, model_path=None): pass
    def score(self, text: str) -> float:
        # Simple score based on length (mock)
        score = min(1.0, len(text) / 200) 
        return round(score, 3)
quality_scorer = ArgumentQualityScorer()

# --- 3. EVIDENCE CONVINCINGNESS RANKER (MOCK CLASS - UNCHANGED) ---
class EvidenceRanker:
    def __init__(self, model_path=None): pass
    def rank(self, text1: str, text2: str) -> int:
        return 1 if len(text1) > len(text2) else 2
evidence_ranker = EvidenceRanker()

# --- 4. DEBATE STRUCTURE GENERATOR (REAL DATA LOGIC) ---

def generate_debate_report_real(speeches_df: pd.DataFrame) -> tuple:
    """
    Constructs the report using actual speeches, topics, and simulated rebuttal flow.
    """
    if speeches_df.empty:
        return [{"Speech_ID": 1, "Side": "N/A", "Topic": "N/A", "Argument_Quality_Score": 0.0, 
                 "Rebuttal_Analysis": "Data loading failed.", "Snippet": "..."}], "UNKNOWN"

    # 1. Select a debate: Use the debate with topic_id 1161 ("We should ban cosmetic surgery")
    # This ID is visible in the snippet and contains good text.
    debate_id = 1161 
    
    # Filter speeches for this debate (the index is the speech order)
    debate_speeches = speeches_df[speeches_df['topic_id'] == debate_id].reset_index(drop=True)
    target_topic = debate_speeches['topic'].iloc[0]
    
    # The uploaded data snippet doesn't contain a 'speech_side' column, 
    # but debates alternate sides (PRO, CON, PRO, CON...)
    SIDES = ['PRO', 'CON'] 
    
    report = []
    
    # We will process the first 5 speeches to create a structured report (2 rounds + closing)
    for i in range(min(5, len(debate_speeches))):
        speech = debate_speeches.iloc[i]
        speech_text = speech['text']
        
        # 2. Assign Side & Order
        side = SIDES[i % 2]
        speech_id = i + 1
        
        # 3. Argument Scoring (Mocked: Uses the length of the real speech text)
        quality_score = quality_scorer.score(speech_text)
        
        # 4. Rebuttal Analysis (Simulated based on order)
        if i == 0:
            rebuttal_note = "Opening Constructive Argument (Initial Thesis)."
        elif i == 1:
            rebuttal_note = f"Direct Rebuttal to {SIDES[0]} Opening (Speech 1)."
        elif i == 2:
            rebuttal_note = f"Defense/Consolidation and Counter-Rebuttal to {SIDES[1]} attack (Speech 2)."
        elif i == 3:
            rebuttal_note = f"Second Line of Attack on {SIDES[0]} core claims."
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
    # ... (function body remains the same)
    report_text = "\n".join([
        f"--- Speech {s['Speech_ID']} ({s['Side']} on '{s['Topic']}'): "
        f"Quality Score: {s['Argument_Quality_Score']:.3f}. Rebuttal: {s['Rebuttal_Analysis']}\n"
        f"Content Snippet: {s['Snippet']}"
        for s in debate_report
    ])
    
    prompt = f"""
    You are an expert, impartial debate judge. Your task is to analyze the debate flow based on the technical report provided below and issue a comprehensive verdict.

    OVERALL DEBATE TOPIC: {overall_topic}

    --- DEBATE JUDGING REPORT ---
    {report_text}
    --- END OF REPORT ---

    INSTRUCTIONS:
    1. SYNTHESIS: Summarize the strongest overall argument (highest cumulative quality score and least rebutted) for the PRO side and the CON side.
    2. REBUTTAL ASSESSMENT: Identify the single most effective rebuttal or defense in the entire debate based on the 'Rebuttal_Analysis'.
    3. FINAL VERDICT: Declare the winning side and assign a final numerical score (0-100) to both sides based on argument quality, evidence strength, and effective rebuttal. Provide a brief, concise rationale.

    --- JUDGMENT OUTPUT FORMAT ---
    [SYNTHESIS]
    PRO Strongest Argument: ...
    CON Strongest Argument: ...

    [REBUTTAL ASSESSMENT]
    Most Effective Rebuttal: ...

    [FINAL VERDICT]
    Winner: [PRO or CON]
    Score PRO: X/100
    Score CON: Y/100
    Rationale: ...
    """
    return prompt

# --- GEMINI API INTEGRATION AND EXECUTION ---

# (Gemini API Functions and Execution Flow remain the same for final call)

client = None

def initialize_gemini_client():
    global client
    try:
        # Note: This requires the GEMINI_API_KEY environment variable to be set
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
        # Simplified call (FIXED the config error)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"\n[API ERROR] An error occurred: {e}"

# --- FINAL EXECUTION FLOW ---

# 1. Use the NEW REAL-DATA function
debate_report, overall_topic = generate_debate_report_real(debate_speeches_df)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)

if initialize_gemini_client():
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\n[CRITICAL FAILURE] Gemini API client could not initialize. Aborting final judgment.") 



