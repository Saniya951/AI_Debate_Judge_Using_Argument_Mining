import pandas as pd
import numpy as np
import random # Needed for sampling
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from google import genai
from google.genai.types import GenerationConfig
from google.genai.errors import APIError

# --- 1. DATA LOADING (UNCHANGED) ---
arg_quality_df = pd.read_csv("arg_quality_rank_30k.csv") 
arg_sentences_df = pd.read_csv("argumentative_sentences_in_spoken_language_with split.csv") 
evidence_train_df = pd.read_csv("train.csv")
evidence_test_df = pd.read_csv("test.csv")
evidence_df = pd.concat([evidence_train_df, evidence_test_df], ignore_index=True)

print("--- Data Loading Complete ---")
print(f"Arg Quality Rank size: {len(arg_quality_df)}")
print(f"Arg Sentences size: {len(arg_sentences_df)}")
print(f"Evidence Convincingness size: {len(evidence_df)}")

# --- 2. ARGUMENT QUALITY SCORER (MOCK CLASS - UNCHANGED) ---
# Pre-process arg_quality_df for training
arg_quality_df = arg_quality_df.dropna(subset=['argument', 'WA'])
arg_quality_df['labels'] = arg_quality_df['WA'].astype(float) 

# Tokenizer initialization
MODEL_NAME = "roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ArgumentQualityScorer:
    """Mock class simulating a trained model predicting argument quality (WA)."""
    def __init__(self, model_path=None): pass
    def score(self, text: str) -> float:
        # NOTE: This function is now redundant as we use the real 'WA' score, but kept for consistency.
        score = min(1.0, len(text) / 200) 
        return round(score, 3)

quality_scorer = ArgumentQualityScorer()

# --- 3. EVIDENCE CONVINCINGNESS RANKER (MOCK CLASS - UNCHANGED) ---
class EvidenceRanker:
    """Mock class simulating a trained model predicting which evidence is stronger."""
    def __init__(self, model_path=None): pass
    def rank(self, text1: str, text2: str) -> int:
        return 1 if len(text1) > len(text2) else 2

evidence_ranker = EvidenceRanker()

# --- 4. FINAL JUDGE LLM PROMPT GENERATOR (FULLY AUTOMATIC) ---

def generate_debate_report_automatic(arg_quality_df: pd.DataFrame) -> tuple:
    """
    Automatically simulates a debate using arguments from arg_quality_df 
    that match the single most frequent topic and assigns sides based on 'stance_WA'.
    """
    # 1. Select the most frequent topic (non-manual topic selection)
    target_topic = arg_quality_df['topic'].mode()[0]
    
    # 2. Filter data for the target topic
    debate_args = arg_quality_df[arg_quality_df['topic'] == target_topic].copy()
    
    # 3. Assign PRO/CON side based on stance_WA column
    # stance_WA = 1 is PRO (supports topic); stance_WA = -1 is CON (opposes topic)
    debate_args['Side'] = debate_args['stance_WA'].apply(lambda x: 'PRO' if x == 1 else 'CON')
    
    # 4. Separate sides and ensure we have at least 3 PRO and 2 CON arguments
    pro_speeches = debate_args[debate_args['Side'] == 'PRO']
    con_speeches = debate_args[debate_args['Side'] == 'CON']
    
    # Use replace=True to handle cases where we might not have 3 or 2 unique arguments
    pro_samples = pro_speeches.sample(n=3, random_state=42, replace=True)
    con_samples = con_speeches.sample(n=2, random_state=42, replace=True)

    # 5. Create a structured flow: PRO 1, CON 1, PRO 2, CON 2, PRO 3 (Summary)
    debate_flow = [
        pro_samples.iloc[0], # Speech 1: PRO constructive
        con_samples.iloc[0], # Speech 2: CON rebuttal (responds to PRO 1)
        pro_samples.iloc[1], # Speech 3: PRO rebuttal/defense (responds to CON 1)
        con_samples.iloc[1], # Speech 4: CON rebuttal (responds to PRO 2)
        pro_samples.iloc[2], # Speech 5: PRO Summary/Closing
    ]

    report = []
    
    for i, speech in enumerate(debate_flow):
        # 6. Use the actual Weighted Average (WA) score as the Quality Score
        quality_score = speech['WA']
        
        # 7. Rebuttal Analysis (Simulate linkage based on flow position)
        if i == 0:
            rebuttal_note = "Opening Constructive Argument."
        elif i == 1:
            rebuttal_note = "Directly rebuts the central claim of the Opening PRO speech (Speech 1)."
        elif i == 2:
            rebuttal_note = "Defends PRO position and rebuts the CON's first main attack (Speech 2)."
        elif i == 3:
            rebuttal_note = "Presents a new point and rebuts the PRO's defense (Speech 3)."
        elif i == 4:
            rebuttal_note = "Closing statement, synthesizing points and defending against the last CON attack (Speech 4)."

        report.append({
            "Speech_ID": i + 1,
            "Side": speech['Side'],
            "Topic": target_topic,
            "Argument_Quality_Score": round(quality_score, 3), 
            "Rebuttal_Analysis": rebuttal_note,
            "Snippet": speech['argument'][:150] + "..."
        })
        
    return report, target_topic

# The old 'generate_final_judge_prompt' logic remains the same:
def generate_final_judge_prompt(debate_report: list, overall_topic: str) -> str:
    # ... (function body remains the same, using the debate_report and overall_topic)
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

client = None

def initialize_gemini_client():
    global client
    try:
        client = genai.Client()
        print("\nGemini Client initialized successfully.")
        return True
    except Exception as e:
        print("\n[CRITICAL ERROR] Failed to initialize Gemini Client.")
        print(f"Details: {e}")
        print("ACTION REQUIRED: Check if your API key is correct and valid, and that the Gemini API is enabled in your Google Cloud project.")
        return False

def get_gemini_judgment(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
    global client
    if not client:
        return "[API EXECUTION FAILED] Client was not initialized due to a critical API key/permission error."

    print(f"\n-> Requesting Judgment from {model_name}...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"\n[API ERROR] An error occurred during content generation: {e}"

# --- FINAL EXECUTION FLOW ---

# 1. Use the AUTOMATIC function
debate_report, overall_topic = generate_debate_report_automatic(arg_quality_df)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)

if initialize_gemini_client():
    # 2. Proceed with generation
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\nAborting final judgment due to client error.")