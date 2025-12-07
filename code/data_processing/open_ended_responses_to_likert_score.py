# Installations
# !pip install openai pandas
import openai
import pandas as pd
import time
import re
import sys
from tqdm import tqdm
sys.path.append("/localdisk1/PARK/park_vlm_finetuning/code/Prompt_Generation")
from generate_prompts import *

# Set OpenAI API key
with open("/localdisk1/PARK/park_vlm_finetuning/OPENAI_API_KEY.txt", "r") as f:
    api_key = f.read().strip()
openai.api_key = api_key  
model = "gpt-5.1"

# column name to statement
positive_statements = {
    1: "The person shows no signs of difficulty in facial expressions.",
    2: "The background is not overloaded, and the lighting conditions are appropriate (not too dark or overlit).",
    3: "The person’s eye blink rate is normal (no absence of blinking, reduced blink rate, or excessive blinking).",
    4: "The camera distance is appropriate, with the upper half of the subject’s body visible and thelower half not captured.",
    5: "The subject’s speech is coherent, easy to understand, and stays focused on the central topic(no unorganized detours).",
    6: "The subject followed all instructions accurately while completing the task (no deviationsobserved).",
    7: "The subject’s lips are never parted when not speaking (e.g., closed or slightly parted most ofthe time).",
    8: "The subject’s facial expressions are natural, emotionally appropriate, and symmetrical (noblank/emotionless face, weakness, asymmetry, or difficulty holding expressions).",
    9: "No abnormal signs are observed in body parts (e.g., no tremors, involuntary shaking, stiffness,rigidity, reduced arm swing, slowness, abnormal posture, or balance issues).",
    10: "The subject appears calm and alert, with no signs of exhaustion, confusion, or inappropriate energy levels.",
    11: "No other individuals are visible in the background during the task.",
    12: "The subject does not use complex or difficult-to-understand sentences.",
    13: "All critical body parts (e.g., unobstructed face) remain fully visible in the frame throughout the task."
}

def map_response_to_likert(row):
    """
    Maps an open-ended response to a Likert score (1-5).
    """
    qid = row['question-id']
    old_question_text = question[qid]
    llm_response = row['llm_response']
    new_question_text = positive_statements[qid]

    prompt = f"""
    Your task is to convert an open-ended response <OLD_RESPONSE> into a Likert score on a scale of 1 to 5.
    The open-ended response was collected for the question <OLD_QUESTION>, but we have reformulated the question into <NEW_QUESTION>, a positive statement for clarity.
    Assess how well the open-ended response aligns with the <NEW_QUESTION>, and assign it a score using a 5-point Likert scale (1 is strongly disagree and 5 is strongly agree).

    ### OLD_QUESTION:
    {old_question_text}

    ### OLD_RESPONSE:
    {llm_response}

    ### NEW_QUESTION:
    {new_question_text}

    ### Instructions:
    1. Carefully interpret the meaning of the open-ended response with respect to the <OLD_QUESTION>.
    2. Only rely on the content of the open-ended response to evaluate its alignment with the <NEW_QUESTION>.
    3. Provide a short but logical explanation for how the response aligns with the <NEW_QUESTION>.
    4. At the end of your explanation, clearly state the most appropriate Likert rating.
    5. End your final answer with: **Likert Score: X** (where X is 1 to 5).

    ### Begin Evaluation:
    """

    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    output = completion.choices[0].message.content
    match = re.search(r"Likert Score: (\d)", output)
    likert_score = int(match.group(1)) if match else None
    return (likert_score, output.strip())

def process_llm_responses(input_csv, output_csv):
    """
    Processes open-ended responses in the input CSV and maps them to Likert scores.
    Saves the results in the output CSV.
    """
    df = pd.read_csv(input_csv)
    df["likert_score"] = None
    df["likert_explanation"] = None

    for i, r in tqdm(df.iterrows()):
        likert_score, explanation = map_response_to_likert(r)
        df.at[i, "likert_score"] = likert_score
        df.at[i, "likert_explanation"] = explanation
        qid = r['question-id']
        new_question_text = positive_statements[qid]
        df.at[i, "positive_statement"] = new_question_text
        time.sleep(1)  # To avoid hitting rate limits
    
        # save partial output
        if i%10==0:
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"Processed responses saved to {output_csv}")

if __name__ == "__main__":
    # LLM_INPUT_FILES = [
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/internVL2_test_responses_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/llava_qwen_test_responses_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_SFT_test_responses_v0.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_test_responses_v0.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/minicpm_test_responses_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/phi3.5_test_responses_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/QwenVL_test_responses.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVAQwen_test_responses_v1.csv"
    # ]

    # LLM_OUTPUT_FILES = [
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/internVL2_test_responses_with_likert_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/llava_qwen_test_responses_with_likert_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_SFT_test_responses_with_likert_v0.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_test_responses_with_likert_v0.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/minicpm_test_responses_with_likert_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/phi3.5_test_responses_with_likert_v1.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/QwenVL_test_responses_with_likert.csv",
    #     "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVAQwen_test_responses_with_likert_v1.csv"
    # ]

    LLM_INPUT_FILES = [
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVAQwen_SFT_test_responses_v1.csv"
    ]

    LLM_OUTPUT_FILES = [
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVAQwen_SFT_test_responses_with_likert_v1.csv"
    ]

    for input_csv, output_csv in zip(LLM_INPUT_FILES, LLM_OUTPUT_FILES):
        print(f"Processing {input_csv}...")
        process_llm_responses(input_csv, output_csv)
        print(f"Saved processed responses with Likert scores to {output_csv}")

    print("END")