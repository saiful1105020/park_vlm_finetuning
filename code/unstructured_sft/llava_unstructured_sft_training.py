"""
Version History:
- v0: Initial version, runs successfully, but converges too quickly. Loss plateaues around 3.75.
- v1: Current version
"""
import os
import av
import fsspec
import shutil
import json
import torch
import decord

import numpy as np

from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, AutoTokenizer, LlavaNextVideoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from datasets import load_dataset, concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from typing import List, Dict, Any, Optional
from decord import VideoReader, gpu, cpu

MAX_LENGTH = 256
BATCH_SIZE = 1
NUM_FRAMES = 32 # more frames -> more VRAM needed

DATASET_PATH = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured" # path where to save the dataset
OUTPUT_DIR = "/localdisk1/PARK/park_vlm_finetuning/checkpoints/unstructured_sft" # path where to save the checkpoints
MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
REPO_ID = "RaushanTurganbay/LLaVa-NeXT-Video-demo" # Change to your hf-hub repo

USE_LORA = False
USE_QLORA = True

def read_video_decord(video_path, num_frames=NUM_FRAMES):
    '''
    Decode the video with Decord decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


# And we also need to load the processor for collate_fn
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

def convert_llava_video_to_llava_next(sample):
    video_path = sample["video"]
    convs = sample["conversations"]

    conversation = []

    for turn in convs:
        src = turn["from"]
        text = turn["value"]

        if src == "system":
            conversation.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": text}
                    ],
                }
            )

        elif src == "human":
            text = text.replace("<image>", "").strip()
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_path},
                        {"type": "text", "text": text},
                    ],
                }
            )

        elif src in ["gpt", "assistant"]:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text}
                    ],
                }
            )

    # IMPORTANT: return the list, not wrapped in a dict
    return conversation

def build_assistant_only_labels(conversation, processor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    conversation: LLaVA-Next style list of messages (system/user/assistant)
    processor: LlavaNextVideoProcessor (processor.tokenizer is the LM tokenizer)
    input_ids: 1D tensor from processor.apply_chat_template(... tokenize=True ...).shape == (L,)

    Returns: labels tensor of shape (L,) with:
      - labels == input_ids on assistant text tokens
      - labels == -100 elsewhere (system, user, special tokens, padding later)
    """
    tokenizer = processor.tokenizer
    labels = torch.full_like(input_ids, -100)

    search_start = 0
    for msg in conversation:
        if msg["role"] != "assistant":
            continue

        # concatenate all text chunks in this assistant turn
        parts = [c["text"] for c in msg["content"] if c.get("type") == "text"]
        if not parts:
            continue
        msg_text = "".join(parts)

        # tokenize assistant text only, without extra special tokens
        msg_tok = tokenizer(
            msg_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        msg_ids = msg_tok["input_ids"][0]
        if msg_ids.numel() == 0:
            continue

        # naive subsequence search inside input_ids, starting from the last match
        found = False
        for i in range(search_start, input_ids.size(0) - msg_ids.size(0) + 1):
            window = input_ids[i : i + msg_ids.size(0)]
            if torch.equal(window, msg_ids):
                labels[i : i + msg_ids.size(0)] = window
                search_start = i + msg_ids.size(0)
                found = True
                break

        if not found:
            # This can happen if the template inserted extra formatting or truncation
            print("Warning: could not align assistant text to tokens")

    return labels


class LLaVAVideoJsonlDataset(Dataset):
    """
    Expects JSONL with items like:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "video", "path": video_path},  # <<< REAL PATH HERE
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]
    """
    def __init__(
        self,
        jsonl_path: str,
        video_root: str = "",
        image_processor=None,           # e.g., AutoProcessor.from_pretrained(...).image_processor
        tokenizer=None,                 # optional; pass if you want tokenized text returned
        num_frames: int = 32,
        resolution: int = 336,
        return_tokenized: bool = True, # set True if you want input_ids/labels here (simple single-turn)
        pad_to_max: Optional[int] = None
    ):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self.video_root = video_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.resolution = resolution
        self.return_tokenized = return_tokenized
        self.pad_to_max = pad_to_max

    def __len__(self):
        return len(self.items)

    def _resolve_path(self, p: str) -> str:
        p = p.strip()
        if not os.path.isabs(p):
            p = os.path.join(self.video_root, p) if self.video_root else p
        return p

    def __getitem__(self, idx):
        raw = self.items[idx]
        video_path = raw["video"]
        vr = decord.VideoReader(video_path, ctx=cpu(0))
        num_frames = min(len(vr), self.num_frames)

        conversation = convert_llava_video_to_llava_next(raw)

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            num_frames=num_frames,
        )

        sample = {k: v.squeeze(0) for k, v in inputs.items()}
        input_ids = sample["input_ids"]

        # build assistant-only labels on the already-tokenized sequence
        labels = build_assistant_only_labels(conversation, processor, input_ids)
        sample["labels"] = labels

        return sample


class LlavaNextSimpleVideoCollator:
    def __init__(self, processor):
        self.processor = processor
        self._dbg = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features[i]:
        #   "input_ids": (Li,)
        #   "attention_mask": (Li,)
        #   "labels": (Li,)
        #   "pixel_values_videos": (F, 3, H, W)

        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        labels_list = [f["labels"] for f in features]
        videos_list = [f["pixel_values_videos"] for f in features]

        # pad text fields with tokenizer
        batch = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            padding=True,
            return_tensors="pt",
        )

        # pad labels manually with -100
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        batch["labels"] = labels

        # stack videos: (B, F, 3, H, W)
        batch["pixel_values_videos"] = torch.stack(videos_list, dim=0)

        return batch

# print(f"Dataset size: {len(dataset)} examples.")
# print("Sample example keys:", dataset[0].keys())
# print("Sample example:\n")

# print(f"Video path: {dataset[0]['video_path']}")
# print(f"Pixel values shape: {dataset[0]['pixel_values'].shape}")  # (T,C,H,W)
# print(f"Number of frames: {dataset[0]['num_frames']}")
# print("Messages:")
# for msg in dataset[0]["messages"]:
#     print(f"  {msg['role']}: {msg['content']}")

# if "input_ids" in dataset[0]:
#     print(f"Input IDs: {dataset[0]['input_ids']}")
#     print(f"Attention Mask: {dataset[0]['attention_mask']}")
#     print(f"Labels: {dataset[0]['labels']}")

## Load model
# Two options for training:
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
# Standard LoRA:  model is loaded with standard LoRA adaptations.

if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
else:
    bnb_config = None # no quantization
    
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,  # or TaskType.SEQ_2_SEQ_LM for encoder-decoder models
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
# print(model)

image_processor = getattr(processor, "image_processor", processor)  # some processors expose .image_processor
# tokenizer = processor.tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer = processor
    
train_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_train.jsonl",
    video_root="",       # or "" if paths in JSONL are absolute
    image_processor=image_processor,
    tokenizer=tokenizer,                 # let your collator/trainer handle tokenization
    num_frames=NUM_FRAMES,
    resolution=336,
    return_tokenized=True
)

test_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test.jsonl",
    video_root="",       # or "" if paths in JSONL are absolute
    image_processor=image_processor,
    tokenizer=tokenizer,                 # let your collator/trainer handle tokenization
    num_frames=NUM_FRAMES,
    resolution=336,
    return_tokenized=True
)


# # Debug
# dl = DataLoader(train_dataset, batch_size=1, collate_fn=LlavaNextSimpleVideoCollator(processor))
# batch = next(iter(dl))
# print("input_ids shape:", batch["input_ids"].shape)
# print("labels shape:", batch["labels"].shape)
# print("num supervised tokens:", (batch["labels"] != -100).sum())
# print("first few tokens / labels:", batch["input_ids"][0][:40], batch["labels"][0][:40])
# print(batch["labels"][0])
# assert False

gradient_accumulation_steps = 2
args = TrainingArguments(
    # args related to training
    output_dir = OUTPUT_DIR,
    eval_strategy = 'steps',
    eval_steps=1000,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = gradient_accumulation_steps,
    learning_rate = 2e-05,
    max_steps = 20000, # adjust this depending on your dataset size
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.1,
    remove_unused_columns=False,

    # args related to eval/save
    logging_steps = 20,
    save_strategy = 'steps',
    save_steps=len(train_dataset)//(BATCH_SIZE*gradient_accumulation_steps), # save every epoch
    save_total_limit = 5,
    fp16 = True, # we have the model train and eval with fp16 precision
    fp16_full_eval = True,
    optim = 'adamw_bnb_8bit', # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
    report_to = "wandb", # install wand to use this
    hub_model_id = None,
    push_to_hub = False, # wel'll push the model to hub after each epoch

    # model that was wrapped for QLORA training with peft will not have arguments listed in its signature
    # so we need to pass lable names explicitly to calculate val loss
    label_names=["labels"],
    dataloader_num_workers=4, # let's get more workers since iterating on video datasets might be slower in general
)

data_collator = LlavaNextSimpleVideoCollator(processor=processor)
trainer = Trainer(
    model = model,
    tokenizer = processor,
    # data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args=args,
)

trainer.train()

