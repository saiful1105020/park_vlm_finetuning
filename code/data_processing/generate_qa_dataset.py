'''
Given the annotations collected from Nami, Natalia, and Cayla,
format them into a simple, QA dataset.

Train: <video_path>\t<question>\t<answer>\t<annotator>\t<question_id>
Test: <video_path>\t<question>\t<answer1>\t<annotator1>\t<answer2>\t<annotator2>\t<answer3>\t<annotator3>\t<question_id>
'''

import csv
import os
import pandas as pd

BASE_PATH = "/localdisk1/PARK/park_vlm_finetuning/"

question = {
        1: 'Please describe whether the person demonstrates any difficulty through their facial expressions. Some examples of visible difficulty include furrowed brow, squinting eyes, clenched jaw, tight lips, head hanging low, sighing, wrinkled forehead, etc. Mention such specific details when found. End output with a final answer choice: \"Yes\" or \"No\".',
        2: 'Mention if the background is overloaded (i.e., too many things), or the lighting condition is inappropriate (i.e., too dark or overlit). Otherwise, just output \"normal background\".',
        3: 'Was there anything abnormal about the person’s eye blink rate? For example, they may not be blinking at all, or they may have reduced or increased blink rate compared to a normal person. If there is nothing abnormal, output \"normal blinking\".',
        4: 'How far is the camera? A good position of the camera would be when the upper half of the subject\'s body remains visible, while the lower half is not captured in the frame. If the upper body is only partly visible, the camera is too close. If the lower body is also visible, the camera is too far.',
        5: 'Is the subject coherent in what they are speaking? Are they delivering an easy to understand story, or are they deterring a lot from the central topic and delivering an unorganized speech?',
        6: 'Please indicate whether the subject was able to follow the instructions while completing the task. If the subject was doing something differently, please describe.',
        7: 'Indicate the extent of lips parting when the subject is not saying anything (i.e., always/most of the times/sometimes/very few times/never).',
        8: 'Indicate which of the following are true for the subject: (i) The individual\'s face appears blank and emotionless, even when they are trying to express an emotion. (ii) The expression is weak or asymmetrical, and the individual has difficulty holding an expression (e.g., smile) for an extended period. Also mention if you observe other facial expression abnormalities. If there is nothing abnormal, output \"nothing abnormal\".',
        9: 'Document any abnormal signs observed in body parts other than the face. This includes, but is not limited to, tremors in the hands, involuntary shaking or rhythmic movements of the upper or lower extremities, stiffness or rigidity in the limbs, reduced arm swing while speaking, or any signs of bradykinesia (slowness of movement). Additionally, note any abnormal postures, difficulty in maintaining balance, or other motor irregularities that may be indicative of Parkinson\'s disease.',
        10: 'Provide a brief description of the subject\'s perceived state of mind, noting whether they appear energetic, exhausted, calm, confused, or exhibit any other relevant emotional or cognitive cues.',
        11: 'Indicate whether any other individuals were present in the background. If so, provide a brief description (e.g., \"An older male is visible in the background\"). Conclude with a final answer: \"Yes\" or \"No\".',
        12: 'Indicate whether the subject is using complex sentences that are difficult to understand. Conclude with a final answer: \"Yes\" or \"No\".',
        13: 'Indicate if any body part critical for this task is partially visible (or invisible). For example, was the subject wearing a mask that may have obstructed important visual information? Or did the face go out of frame while the subject was completing the task?'
    }

question_id_to_column_names = {
    1: "Apparent difficulty completing the task",
    2: "Background and lighting",
    3: "Blink rate",
    4: "Camera Position",
    5: "Coherence",
    6: "Compliance with tasks instructions",
    7: "Lips parting when the mouth is at rest",
    8: "Masked facies",
    9: "Observations of other body parts not being directly assessed",
    10: "Overall appearance",
    11: "Presence of other persons",
    12: "Usage of complex sentence",
    13: "Visibility of significant body parts"
}

# Read annotations from CSV file (training fold)
def read_train_annotations(annotator_name="Cayla"):
    file_path = os.path.join("/localdisk1/PARK/park_vlm/Annotations/Clinical/", f"{annotator_name}/{annotator_name.lower()}_unique_final_with_pd_labels.csv")
    annotations = pd.read_csv(file_path)
    
    video_base_path = "/localdisk1/PARK/park_vlm/Videos"
    dataset = []

    for index, row in annotations.iterrows():
        data = {}
        data["video_path"] = os.path.join(video_base_path, row["Filename"])

        for qid in range(1, 14):
            col_name = question_id_to_column_names[qid]
            data[f"question"] = question[qid]
            data[f"answer"] = row[col_name]
            data["question_id"] = qid
            data[f"annotator"] = annotator_name
            dataset.append(data)

    return dataset

def read_test_annotations():
    test_annotation_path = "/localdisk1/PARK/park_vlm/Annotations/FinalData/all_responses_combined.csv"
    annotations = pd.read_csv(test_annotation_path)

    video_base_path = "/localdisk1/PARK/park_vlm/Videos"
    dataset = []

    for index, row in annotations.iterrows():
        data = {}
        data["video_path"] = os.path.join(video_base_path, row["filename"])
        qid = row["question-id"]
        data["question"] = question[qid]
        data["question_id"] = qid

        for index, annotator in enumerate(["Nami", "Natalia", "Cayla"]):
            data[f"answer{index+1}"] = row[f"{annotator.lower()}_response"]
            data[f"annotator{index+1}"] = annotator
        dataset.append(data)

    return dataset

if __name__ == "__main__":
    # Read raw annotations from all annotators (training fold)
    annotator_names = ["Nami", "Natalia", "Cayla"]
    train_dataset = []
    for annotator in annotator_names:
        subset = read_train_annotations(annotator_name=annotator)
        train_dataset.extend(subset)
        
    # Save the combined dataset to a CSV file
    output_file = os.path.join(BASE_PATH, "data/QA_dataset/combined_qa_dataset_train.csv")
    print(f"Size of train dataset: {len(train_dataset)}")
    pd.DataFrame(train_dataset).to_csv(output_file, index=False)

    # Read raw annotations for the test fold
    test_dataset = read_test_annotations()
    output_file = os.path.join(BASE_PATH, "data/QA_dataset/combined_qa_dataset_test.csv")
    print(f"Size of test dataset: {len(test_dataset)}")
    pd.DataFrame(test_dataset).to_csv(output_file, index=False)

    print("Completed creating the QA dataset.")