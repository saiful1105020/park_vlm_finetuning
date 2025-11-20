import pandas as pd
import numpy as np

clinical_ratings_file = "/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_test_likert.csv"

def merge_ratings(row):
    r1 = row["likert1_score"]
    r2 = row["likert2_score"]
    r3 = row["likert3_score"]

    # if there is majority voting, return that
    if r1==r2 or r1==r3:
        return r1
    
    if r2==r3:
        return r2
    
    return int((r1+r2+r3)/3)

def main():
    ratings = pd.read_csv(clinical_ratings_file)
    ratings["overall_ratings"] = ratings.apply(merge_ratings, axis=1)
    ratings.to_csv("/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_test_groud_truth_likert.csv", index=False)
    return

if __name__ == "__main__":
    main()
    print("END")