import pandas as pd
import os
import sys
sys.path.append("/localdisk1/PARK/park_vlm_finetuning/code/Prompt_Generation")
from generate_prompts import *

QTYPES_MAP = {
    1: 'Motor Performance and Task Execution',
    2: 'Technical Quality and Environmental Context',
    3: 'Facial Expression and Ocular Analysis',
    4: 'Technical Quality and Environmental Context',
    5: 'Speech and Cognitive-Linguistic Assessment',
    6: 'Motor Performance and Task Execution',
    7: 'Facial Expression and Ocular Analysis',
    8: 'Facial Expression and Ocular Analysis',
    9: 'Motor Performance and Task Execution',
    10: 'Facial Expression and Ocular Analysis',
    11: 'Technical Quality and Environmental Context',
    12: 'Speech and Cognitive-Linguistic Assessment',
    13: 'Technical Quality and Environmental Context'
}

QTYPES = list(set(QTYPES_MAP.values()))

def create_dataset():
    human_label_dataset = []
    row_id = 1

    # Sample 15 data from clinical labels for each question types
    ratings = pd.read_csv("/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_test_likert.csv")
    for qtype_id in range(4):
        qtype = QTYPES[qtype_id]
        qids = [x for x in QTYPES_MAP.keys() if QTYPES_MAP[x]==qtype]
        ratings_subset = ratings[ratings["question_id"].astype(int).isin(qids)]

        for annot_id in range(1,4):
            df_annot = ratings_subset.sample(5)
            for i, r in df_annot.iterrows():
                item = {
                    'row_id': row_id,
                    'q_type': qtype,
                    'annotation_type': f"C-{r[f'annotator{annot_id}']}",
                    'old_q': r['question'],
                    'response': r[f'answer{annot_id}'],
                    'new_q': r['positive_statement'],
                    'gpt_likert_score': r[f'likert{annot_id}_score'],
                    'gpt_likert_explanation': r[f'likert{annot_id}_explanation'],
                    'A1_likert_score': None,
                    'A2_likert_score': None,
                    'A3_likert_score': None
                }
                human_label_dataset.append(item)
                row_id +=1

    # Sample 10 data from LlaVA-Qwen labels for each question types
    ratings = pd.read_csv("/localdisk1/PARK/park_vlm_finetuning/model_outputs/llava_qwen_test_responses_with_likert_v1.csv")
    for qtype_id in range(4):
        qtype = QTYPES[qtype_id]
        qids = [x for x in QTYPES_MAP.keys() if QTYPES_MAP[x]==qtype]
        ratings_subset = ratings[ratings["question-id"].astype(int).isin(qids)]
        df_annot = ratings_subset.sample(10)

        for i, r in df_annot.iterrows():
            item = {
                'row_id': row_id,
                'q_type': qtype,
                'annotation_type': "LlaVA-Qwen",
                'old_q': question[r['question-id']],
                'response': r['llm_response'],
                'new_q': r['positive_statement'],
                'gpt_likert_score': r['likert_score'],
                'gpt_likert_explanation': r[f'likert_explanation'],
                'A1_likert_score': None,
                'A2_likert_score': None,
                'A3_likert_score': None
            }
            human_label_dataset.append(item)
            row_id +=1

    df = pd.DataFrame.from_dict(human_label_dataset)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42) # random_state for reproducibility

    data_dir = os.path.join("/localdisk1/PARK/park_vlm_finetuning/data", "Likert_Human_Validation")
    os.makedirs(data_dir, exist_ok=True)

    df_shuffled.to_csv(os.path.join(data_dir, "validate_likert_score_combined.csv"), index=False)
    print("Dataset saved for human validation")

def prep_dataset_for_annotator(data_location, id=1):
    df = pd.read_csv(data_location)

    # remove llm ratings columns
    df = df.drop(columns=['q_type', 'annotation_type', 'gpt_likert_score', 'gpt_likert_explanation'])

    for i in range(1,4):
        if i==id:
            continue

        df = df.drop(columns=[f'A{i}_likert_score'])

    data_dir = f"{data_location[:-4]}_A{id}.csv"
    df.to_csv(data_dir, index=False)
    return

if __name__ == "__main__":
    create_dataset()
    data_location = "/localdisk1/PARK/park_vlm_finetuning/data/Likert_Human_Validation/validate_likert_score_combined.csv"
    for i in range(1,4):
        prep_dataset_for_annotator(data_location, i)