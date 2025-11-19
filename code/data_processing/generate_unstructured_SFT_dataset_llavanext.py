
import json
import os


def convert_llava_video_to_llava_next(sample):
    video_path = sample["video"]
    convs = sample["conversations"]

    conversation = []

    for turn in convs:
        role = turn["from"]
        text = turn["value"]

        if role == "system":
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": text}],
            })

        elif role == "human":
            # Strip the legacy <image> tag
            text = text.replace("<image>", "").strip()
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": text},
                ],
            })

        elif role in ["gpt", "assistant"]:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            })

    return conversation

if __name__ == "__main__":
    input_jsonl = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test.jsonl"
    output_jsonl = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test_llavanext.jsonl"

    with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            conversation = convert_llava_video_to_llava_next(sample)

            out_sample = {
                "conversation": conversation
            }
            fout.write(json.dumps(out_sample) + "\n")

    print(f"Converted dataset saved to {output_jsonl}")