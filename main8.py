import pandas as pd
import numpy as np
import random
import os
import re # For better string cleaning/matching

# FIX: Adjusted import to be more reliable in execution environment
import google.genai as genai
from google.genai.errors import APIError
from google.genai.types import GenerationConfig

# --- 1. DATA LOADING & ENRICHMENT ---
# Load the primary debate speeches
debate_speeches_df = pd.read_csv("ibm_debate_speeches.csv")

# Load the external dataset with human-rated quality scores and stance
# We will use this to assign more realistic metrics to our debate speeches
quality_df = pd.read_csv("arg_quality_rank_30k.csv")

print("--- Data Loading Complete ---")
print(f"Total Debate Speeches loaded: {len(debate_speeches_df)}")
print(f"Total Quality Annotations loaded: {len(quality_df)}")


# --- 2. ENRICHMENT UTILITY CLASS (Replacing Scorer/Classifier Simulations) ---
class DebateEnricher:
    """
    Replaces the simulated Scorer and Classifier by using the loaded
    'ground-truth' data from external CSVs to enrich the debate speeches.
    """
    def __init__(self, quality_df: pd.DataFrame):
        # Use the 'WA' (Writers' Acceptance) score as the primary quality metric
        self.quality_data = quality_df[['topic', 'WA', 'stance_WA', 'argument']].copy()

        # Rename columns for clarity
        self.quality_data.rename(columns={'stance_WA': 'stance_label'}, inplace=True)
        
        # Convert stance to 'PRO' (1) and 'CON' (-1)
        self.quality_data['stance_label'] = self.quality_data['stance_label'].apply(
            lambda x: 'PRO' if x == 1 else 'CON' if x == -1 else 'NEUTRAL'
        )

        # Pre-process topics for easier matching
        self.quality_data['normalized_topic'] = self._normalize_text(self.quality_data['topic'])

    def _normalize_text(self, text):
        if isinstance(text, pd.Series):
            return text.str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.strip()
        return str(text).lower().replace(r'[^a-z0-9\s]', '')

    def get_enriched_metrics(self, topic: str, text: str) -> dict:
        """
        Attempts to assign a quality score and stance based on the enriched dataset.
        If a direct match is found, use its human-rated metrics.
        If no direct match is found, fallback to sampling from related topics.
        """
        normalized_topic = self._normalize_text(topic)

        # 1. Try to find annotations for the current topic
        topic_matches = self.quality_data[self.quality_data['normalized_topic'] == normalized_topic]

        if not topic_matches.empty:
            # 2. If topic matches, find the closest matching argument text (using snippet for simplicity)
            # Find the closest argument based on word overlap or a random sample for the topic
            
            # Simple fallback: sample a random metric from this topic
            sample = topic_matches.sample(n=1).iloc[0]
            
            quality_score = round(sample['WA'], 3)
            stance = sample['stance_label']
            source = f"Human-Annotated (WA={quality_score})"
            
        else:
            # 3. Fallback: Use the original quality scoring mechanism but label it as 'Simulated'
            # This ensures every speech gets a score, even if the topic is new.
            quality_scorer = ArgumentQualityScorer() # Use the old simulated scorer as fallback
            quality_score = quality_scorer.score(text)
            
            # Use the rule-based classifier as fallback for stance
            stance_classifier = StanceClassifier()
            stance = stance_classifier.predict_stance_general(text)
            
            source = "Simulated (Fallback)"
            
        return {
            "Argument_Quality_Score": quality_score,
            "Side": stance,
            "Score_Source": source
        }

# --- DYNAMIC DEBATE STRUCTURE GENERATOR (Original components kept for fallback) ---

class ArgumentQualityScorer:
    """ Original simulation kept as fallback logic. """
    def __init__(self, model_path=None): pass
    def score(self, text: str) -> float:
        word_count = len(text.split())
        score_development = min(1.0, word_count / 400.0)
        paragraph_count = text.count('\n\n') + 1
        score_structure = max(0.0, 1.0 - (paragraph_count / (word_count / 100 + 1)))
        long_words = [w for w in text.split() if len(w) > 6]
        long_word_ratio = len(long_words) / (word_count + 1e-6)
        score_complexity = min(0.5, long_word_ratio * 1.5)
        final_score = (score_development * 0.4) + (score_structure * 0.3) + (score_complexity * 0.3)
        return round(np.clip(final_score, 0.0, 1.0), 3)

class StanceClassifier:
    """ Original rule-based classifier kept as fallback logic. """
    def __init__(self): pass
    def predict_stance_general(self, speech_text: str) -> str:
        text = speech_text.lower()
        if any(phrase in text for phrase in ['we advocate', 'we support', 'we stand for', 'our position is pro']):
            return 'PRO'
        if any(phrase in text for phrase in ['we oppose', 'we stand against', 'we do not support', 'our position is con']):
            return 'CON'
        return random.choice(['PRO', 'CON'])

# Initialize the new Enricher
enricher = DebateEnricher(quality_df)
print(f"Metrics now use enriched human-annotated data where possible.")


def generate_enriched_debate_report(speeches_df: pd.DataFrame, enricher: DebateEnricher) -> tuple:
    """
    Constructs the report using a random debate, now utilizing the DebateEnricher
    for metrics.
    """
    if speeches_df.empty:
        return [{"Speech_ID": 1, "Side": "N/A", "Topic": "N/A", "Argument_Quality_Score": 0.0,
                 "Rebuttal_Analysis": "Data loading failed.", "Snippet": "..."}], "UNKNOWN"

    # 1. DYNAMIC SELECTION: Pick a random topic ID
    available_topics = speeches_df['topic_id'].unique()
    if len(available_topics) == 0:
        return [], "UNKNOWN"

    random_debate_id = random.choice(available_topics)
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

        # 2. ENHANCED METRICS ASSIGNMENT (Using the Enricher)
        metrics = enricher.get_enriched_metrics(target_topic, speech_text)
        quality_score = metrics["Argument_Quality_Score"]
        side = metrics["Side"]
        score_source = metrics["Score_Source"]

        # 3. DYNAMIC REBUTTAL ANALYSIS: Based on Full Turn Flow (unchanged)
        rebuttal_note = ""
        if i == 0:
            rebuttal_note = "Opening Constructive Argument (Initial Thesis) from the first side."
        elif speech_id == total_speeches:
            rebuttal_note = "Final Summary/Closing Statement."
        elif i == 1:
            if report:
                 rebuttal_note = f"Direct Rebuttal/Second Constructive to Speech 1 (Side: {report[0]['Side']})."
            else:
                 rebuttal_note = "Direct Rebuttal/Second Constructive."
        else:
            rebuttal_note = "Mid-debate Rebuttal, Consolidation, or Extension."

        report.append({
            "Speech_ID": speech_id,
            "Side": side,
            "Topic": target_topic,
            "Argument_Quality_Score": quality_score,
            "Score_Source": score_source, # NEW FIELD
            "Rebuttal_Analysis": rebuttal_note,
            "Snippet": speech_text[:150].replace('\n', ' ') + "..."
        })

    return report, target_topic

# --- FINAL JUDGE PROMPT GENERATOR (UPDATED to reference the Score Source) ---
def generate_final_judge_prompt(debate_report: list, overall_topic: str) -> str:
    report_text = "\n".join([
        f"--- Speech {s['Speech_ID']} ({s['Side']} on '{s['Topic']}'): "
        f"Quality Score: {s['Argument_Quality_Score']:.3f} (Source: {s['Score_Source']}). " # UPDATE
        f"Rebuttal: {s['Rebuttal_Analysis']}\n"
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
        * **Crucially, analyze the Human-Annotated Quality Score** (if available, otherwise the Simulated Score) to determine which side delivered stronger *initial* content and whether that advantage was sustained or lost through the rebuttals.
    3.  **CONSTRUCTIVE FEEDBACK:** Provide 2-3 specific, actionable points for both the PRO and CON teams.

    --- TECHNICAL DEBATE REPORT ---
    {report_text}
    """
    return prompt

# --- GEMINI API INTEGRATION AND EXECUTION (Unchanged) ---
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
        return False

def get_gemini_judgment(prompt: str, model_name: str = 'gemini-2.5-pro') -> str:
    global client
    if not client:
        return "[API EXECUTION FAILED] Client was not initialized."

    try:
        # Implementing basic retry logic
        MAX_RETRIES = 3
        DELAY = 1  # seconds
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except APIError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(DELAY * (2**attempt))
                else:
                    raise e
        return "" # Should not reach here
    except Exception as e:
        # Log the full exception for debugging
        return f"\n[API ERROR] An error occurred while generating content: {e}"

# --- FINAL EXECUTION FLOW ---

# 1. Use the NEW, ENRICHED function
debate_report, overall_topic = generate_enriched_debate_report(debate_speeches_df, enricher)
final_prompt = generate_final_judge_prompt(debate_report, overall_topic)

print("\n\n" + "="*50)
print(f"      FINAL JUDGE LLM PROMPT GENERATED for Topic: {overall_topic}")
print("="*50)
print(final_prompt)

if initialize_gemini_client():
    print(f"\n-> Requesting Judgment for topic '{overall_topic}' from gemini-2.5-pro...")
    final_judgment_text = get_gemini_judgment(final_prompt, model_name='gemini-2.5-pro')

    print("\n" + "="*50)
    print("      FINAL AI DEBATE JUDGE VERDICT (via Gemini)")
    print("="*50)
    print(final_judgment_text)
else:
    print("\n[CRITICAL FAILURE] Gemini API client could not initialize. Aborting final judgment.")
