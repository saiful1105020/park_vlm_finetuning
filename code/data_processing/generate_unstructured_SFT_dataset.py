import argparse, json, os, random, csv
from pathlib import Path

# SYSTEM_MESSAGE = "Imagine you are a clinician specializing in movement disorders. " \
# "Rely on your knowledge of neurology and clinical care. " \
# "Now, you are watching a home-recorded video of a person performing some tasks used to assess Parkinson's disease. " \
# "No experts supervise the person, so there can be different types of noise, or the person may not follow the task instructions properly. " \
# "The person can also show symptoms that may be associated with having Parkinson's disease. " \
# "Focus on the noises, task instructions, user compliance, and possible symptoms of Parkinson's disease while answering the question."

SYSTEM_MESSAGE = "Now, You are reviewing a home-recorded video of a person performing tasks " \
"commonly used to assess Parkinson's disease. Since the video is unsupervised, there may be noise, " \
"imperfect task execution, or deviations from standard instructions. Your role is to carefully observe " \
"and describe the following aspects without making diagnostic judgments or conclusions regarding the diseases."

TASK_INSTRUCTION = "Task instructions: The person will talk about a recent book they have read or a movie " \
"or TV show they have watched. For this task, the face is crucial body part you should focus on. " \
"Additionally, you should also observe other body parts for relevant symptoms or signs of " \
"Parkinson's disease. However, do not give your judgement as clinician other than answering the question " \
"very carefully analyzing the video. Answer the question about what is happening in the video. " \
"Keep your response short and precise. Please only answer the question from the video. " \
"Keep your response precise and to the point."

SYSTEM_MESSAGE += "\n\n" + TASK_INSTRUCTION

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
    main(fold="train")
    main(fold="test")