import argparse, json, os, random, csv
from pathlib import Path

SYSTEM_MESSAGE = "Imagine you are a clinician specializing in movement disorders. " \
"Rely on your knowledge of neurology and clinical care. " \
"Now, you are watching a home-recorded video of a person performing some tasks used to assess Parkinson's disease. " \
"No experts supervise the person, so there can be different types of noise, or the person may not follow the task instructions properly. " \
"The person can also show symptoms that may be associated with having Parkinson's disease. " \
"Focus on the noises, task instructions, user compliance, and possible symptoms of Parkinson's disease while answering the question."

def to_vlm_conversations(row, system=SYSTEM_MESSAGE, include_system=True):
    path, q, a = row
    conv = []
    if include_system and system:
        conv.append({"from": "system", "value": system})
    conv.append({"from": "human", "value": "<image>\n" + q.strip()})
    conv.append({"from": "gpt", "value": a.strip()})
    return {
        "video": path.strip(),
        "conversations": conv
    }

def read_rows(csv_path):
    rows = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect columns: video_filename, question, answer
        for r in reader:
            path = r["video_path"].strip()
            q = r["question"].strip()
            a = r["answer"].strip()
            rows.append((path, q, a))
    return rows

def read_test_rows(csv_path):
    rows = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect columns: video_filename, question, answer
        for r in reader:
            path = r["video_path"].strip()
            q = r["question"].strip()
            for i in range(3):
                a = r[f"answer{i+1}"].strip()
                rows.append((path, q, a))
    return rows

def write_jsonl(objs, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main(fold="train"):
    if fold == "train":
        input_csv = "/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_train.csv"
        output_jsonl = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_train.jsonl"
        rows = read_rows(input_csv)
        objs = []
        for row in rows:
            objs.append(to_vlm_conversations(row, system=SYSTEM_MESSAGE, include_system=True))
        write_jsonl(objs, output_jsonl)
        print(f"Wrote {len(objs)} examples to {output_jsonl}")

    elif fold == "test":
        input_csv = "/localdisk1/PARK/park_vlm_finetuning/data/QA_dataset/combined_qa_dataset_test.csv"
        output_jsonl = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test.jsonl"

        rows = read_test_rows(input_csv)
        objs = []
        for row in rows:
            objs.append(to_vlm_conversations(row, system=SYSTEM_MESSAGE, include_system=True))
        write_jsonl(objs, output_jsonl)
        print(f"Wrote {len(objs)} examples to {output_jsonl}")

if __name__ == "__main__":
    main(fold="test")