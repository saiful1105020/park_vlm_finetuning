import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, cohen_kappa_score
import pingouin as pg
import krippendorff
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

def compute_scores(hypothesis, ref1, ref2, ref3):
    H, R = [hypothesis], [ref1, ref2, ref3]
    bleu = corpus_bleu(H, R).score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    best_rouge1 = 0
    best_rougeL = 0

    for ref in R:
        scores = scorer.score(ref, hypothesis)
        best_rouge1 = max(best_rouge1, scores["rouge1"].fmeasure)
        best_rougeL = max(best_rougeL, scores["rougeL"].fmeasure)

    # print(bleu)
    # print(best_rouge1)
    # print(best_rougeL)
    # assert False

    return bleu, best_rouge1, best_rougeL
    
def main():
    llm_names = [
        "InternVL2", 
        "LlaVA-Qwen", 
        "LlaVA-Next",
        "LlaVA-Next-SFT",
        "MiniCPM",
        "Phi3.5",
        "QwenVL",
        "LlaVA-Qwen-MSI"
    ]
    llm_ratings_paths = [
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/internVL2_test_responses_with_likert_v1.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/llava_qwen_test_responses_with_likert_v1.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_test_responses_with_likert_v0.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVANext_SFT_test_responses_with_likert_v0.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/minicpm_test_responses_with_likert_v1.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/phi3.5_test_responses_with_likert_v1.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/QwenVL_test_responses_with_likert.csv",
        "/localdisk1/PARK/park_vlm_finetuning/model_outputs/LlaVAQwen_test_responses_with_likert_v0.csv"
    ]

    clinical_ratings = pd.read_csv("/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_test_groud_truth_likert.csv")
    clinical_ratings['video_path'] = clinical_ratings['video_path'].astype(str)

    for llm_name, llm_ratings_filename in zip(llm_names, llm_ratings_paths):
        llm_ratings = pd.read_csv(llm_ratings_filename)
        llm_ratings['video_path'] = llm_ratings['filename'].apply(lambda x: f'/localdisk1/PARK/park_vlm/Videos/{x}').astype(str)
        llm_ratings['question_id'] = llm_ratings['question-id']
        llm_ratings.drop(columns=['question-id'])

        combined_ratings = pd.merge(llm_ratings, clinical_ratings, on=['video_path', 'question_id'], how='inner')

        bleu_scores, R1_scores, RL_scores = [], [], []
        for i, r in combined_ratings.iterrows():
            bleu, R1, RL = compute_scores(hypothesis=r['llm_response'], ref1=r['answer1'], ref2=r['answer2'], ref3=r['answer3'])
            bleu_scores.append(bleu)
            R1_scores.append(R1)
            RL_scores.append(RL)

        final_results = {
            "Bleu_scores": bleu_scores,
            "Rouge1_scores": R1_scores,
            "RougeL_scores": RL_scores,
            "BLEU_mean": np.array(bleu_scores).mean(),
            "R1_mean": np.array(R1_scores).mean(),
            "RL_mean": np.array(RL_scores).mean()
        }
        
        results_base_path = "/localdisk1/PARK/park_vlm_finetuning/results"
        os.makedirs(results_base_path, exist_ok=True)
        results_path = os.path.join(results_base_path, f"llm_automated_scores_overall_{llm_name}.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()