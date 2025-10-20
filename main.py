import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- 1. DATA LOADING ---

# The user-uploaded files are accessible via the file fetching tool
# We will use the following files:
# 1. Argument Quality Ranking (for training a quality scorer)
arg_quality_df = pd.read_csv("arg_quality_rank_30k.csv") 

# 2. Argument Component Detection (for training a component miner - used for evidence)
# This one contains sentences labeled as argumentative/non-argumentative.
arg_sentences_df = pd.read_csv("argumentative_sentences_in_spoken_language_with split.csv") 

# 3. Evidence Convincingness (for training an evidence quality ranker)
# We will combine train.csv and test.csv here
evidence_train_df = pd.read_csv("train.csv")
evidence_test_df = pd.read_csv("test.csv")
evidence_df = pd.concat([evidence_train_df, evidence_test_df], ignore_index=True)

print("--- Data Loading Complete ---")
print(f"Arg Quality Rank size: {len(arg_quality_df)}")
print(f"Arg Sentences size: {len(arg_sentences_df)}")
print(f"Evidence Convincingness size: {len(evidence_df)}")

# --- 2. ARGUMENT QUALITY SCORER (REGRESSION) ---

# Pre-process arg_quality_df for training
# Target variable: 'WA' (Weighted Average score for argument quality)
arg_quality_df = arg_quality_df.dropna(subset=['argument', 'WA'])
arg_quality_df['labels'] = arg_quality_df['WA'].astype(float) # Target label is the quality score

# Split the data
train_df_qual, val_df_qual = train_test_split(arg_quality_df, test_size=0.1, random_state=42)

# Tokenizer and Model Initialization (Conceptual: Choose a base model)
MODEL_NAME = "roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # Input is the argument text
    return tokenizer(examples['argument'], padding="max_length", truncation=True, max_length=128)

# Convert to HuggingFace Dataset objects
train_dataset = Dataset.from_pandas(train_df_qual[['argument', 'labels']]).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df_qual[['argument', 'labels']]).map(tokenize_function, batched=True)

# Define the Model (Conceptual: Full training loop omitted)
# model_qual = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1) 
# The num_labels=1 makes it a regression head

# --- Conceptual Scorer Class ---
class ArgumentQualityScorer:
    """Mock class simulating a trained model predicting argument quality (WA)."""
    def __init__(self, model_path=None):
        # In a real project: load the fine-tuned model here
        pass
        
    def score(self, text: str) -> float:
        """Mocks the prediction of a quality score (0.0 to 1.0)."""
        # For demo, returns a score based on text length (bad practice, but shows intent)
        score = min(1.0, len(text) / 200) # Longer arguments get higher mock scores
        return round(score, 3)

# Initialize the mock scorer
quality_scorer = ArgumentQualityScorer()

print("\n--- Argument Quality Scorer Initialized ---")
mock_score = quality_scorer.score(arg_quality_df.iloc[0]['argument'])
print(f"Mock Score for a sample argument: {mock_score}")


# --- 3. EVIDENCE CONVINCINGNESS RANKER (PAIRWISE CLASSIFICATION) ---

# Pre-process evidence_df for training
evidence_df['text_pair'] = evidence_df['evidence_1'] + tokenizer.sep_token + evidence_df['evidence_2']
evidence_df['labels'] = evidence_df['label'] - 1 # Convert label 1/2 to 0/1 for classification

# Conceptual: We would train a model on (topic, evidence_1, evidence_2) -> label (1 or 2)

class EvidenceRanker:
    """Mock class simulating a trained model predicting which evidence is stronger."""
    def __init__(self, model_path=None):
        # In a real project: load the fine-tuned model here
        pass
        
    def rank(self, text1: str, text2: str) -> int:
        """Mocks the prediction of the better evidence (1 or 2)."""
        # For demo, the longer text wins (bad practice, but shows intent)
        return 1 if len(text1) > len(text2) else 2

# Initialize the mock ranker
evidence_ranker = EvidenceRanker()

print("\n--- Evidence Convincingness Ranker Initialized ---")
sample_row = evidence_df.iloc[0]
mock_rank = evidence_ranker.rank(sample_row['evidence_1'], sample_row['evidence_2'])
print(f"Sample E1: {sample_row['evidence_1'][:50]}...")
print(f"Sample E2: {sample_row['evidence_2'][:50]}...")
print(f"Mock Ranker Result: Evidence {mock_rank} is stronger.")

# --- 4. FINAL JUDGE LLM PROMPT GENERATOR ---

def generate_debate_report(debate_speeches_data: pd.DataFrame, quality_scorer: ArgumentQualityScorer, debate_id):
    """
    Simulates the Argument Mining and Scoring phase to generate a Judge Report.
    NOTE: debate_speeches_data is assumed to be loaded from Hugging Face or another source.
    Here we use arg_sentences_df as a proxy for mining claims/evidence from a speech.
    """
    
    # 1. Structure the Debate (Mocked based on argument component data)
    # Since we don't have the IBM debate speeches file, we'll use a mock debate structure
    # based on the argument sentences dataset.
    mock_speeches = [
        {"side": "PRO", "text": arg_sentences_df.iloc[i]['sentence'] + " " + arg_sentences_df.iloc[i]['context'], "responds_to": None, "topic": arg_sentences_df.iloc[i]['topic']} 
        for i in range(5) # Use first 5 sentences as mock speeches
    ]
    
    report = []
    
    for i, speech in enumerate(mock_speeches):
        # 2. Argument Mining (Mocked: Treat the whole speech/sentence as the argument)
        argument_text = speech['text']
        
        # 3. Argument Scoring
        quality_score = quality_scorer.score(argument_text)
        
        # 4. Rebuttal Analysis (Mocked: Only the second speech responds to the first)
        rebuttal_note = "N/A"
        if i == 1:
            rebuttal_note = "Directly addresses and attempts to refute the main point of Speech 1."
        elif i > 1 and i % 2 == 1:
             rebuttal_note = f"Responds to the core claim of Speech {i-1} with counter-evidence."

        report.append({
            "Speech_ID": i + 1,
            "Side": speech['side'],
            "Topic": speech['topic'],
            "Argument_Quality_Score": quality_score,
            "Rebuttal_Analysis": rebuttal_note,
            "Snippet": argument_text[:150] + "..."
        })
        
    return report

def generate_final_judge_prompt(debate_report: list, overall_topic: str) -> str:
    """Constructs the final, detailed prompt for the Judge LLM."""
    
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

# --- EXECUTE THE JUDGING PIPELINE ---

overall_topic = arg_sentences_df['topic'].iloc[0] # Use a sample topic
debate_report = generate_debate_report(None, quality_scorer, 1)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)



# Assuming the previous code for data loading, mock scorers, and 
# final_prompt generation has run and the final_prompt variable is set.

# ----------------------------------------------------------------------
# NEW CODE FOR GEMINI API INTEGRATION
# ----------------------------------------------------------------------
# --- REVISED GEMINI API EXECUTION ---

import os
from google import genai
from google.genai.types import GenerationConfig
from google.genai.errors import APIError # Import specific error type for better debugging

# 1. Initialize the client globally before the function call
# We define a placeholder variable here
client = None

# 2. Function to safely initialize the client
def initialize_gemini_client():
    global client
    try:
        # This will raise an exception if the key is invalid or permissions are missing
        client = genai.Client()
        print("\nGemini Client initialized successfully.")
        return True
    except Exception as e:
        print("\n[CRITICAL ERROR] Failed to initialize Gemini Client.")
        print(f"Details: {e}")
        print("ACTION REQUIRED: Check if your API key is correct and valid, and that the Gemini API is enabled in your Google Cloud project.")
        return False
    



    
# --- 3. Function to request the judgment (MODIFIED) ---

# NOTE: The client initialization (initialize_gemini_client) remains the same.

def get_gemini_judgment(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
    """
    Sends the structured debate report prompt to the Gemini model and returns the judgment.
    The GenerationConfig object has been removed to fix the 'tools' attribute error.
    """
    global client
    
    # Check if client was successfully initialized
    if not client:
        return "[API EXECUTION FAILED] Client was not initialized due to a critical API key/permission error."

    print(f"\n-> Requesting Judgment from {model_name}...")
    
    # *** FIX: Call the API WITHOUT the problematic 'config' object ***
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            # config=config # REMOVED: This was the source of the error
        )
        return response.text
    except Exception as e:
        # Note: APIError is now handled generically as the library version is uncertain
        return f"\n[API ERROR] An error occurred during content generation: {e}"


# --- Execution Flow (Remains the same) ---
# The rest of your main script stays the same, including the calls to initialize_gemini_client()

if initialize_gemini_client():
    # 2. Only proceed with generation if initialization was successful
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\nAborting final judgment due to client error.")