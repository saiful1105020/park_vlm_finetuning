"""
Version History:
- v0: Initial version, runs successfully, but converges too quickly. Loss plateaues around 3.75.
"""
import os
import av
import fsspec
import shutil
import numpy as np
import json
import copy
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from llava.mm_utils import process_images, tokenizer_image_token

import torch
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any, Optional
import decord

# from llava.model.builder import load_pretrained_model
MAX_LENGTH = 256
BATCH_SIZE = 1
NUM_FRAMES = 32 # more frames -> more VRAM needed
DATASET_PATH = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured" # path where to save the dataset
OUTPUT_DIR = "/localdisk1/PARK/park_vlm_finetuning/checkpoints/unstructured_sft" # path where to save the checkpoints

USE_LORA = False
USE_QLORA = True
MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    MODEL_ID,
    None,
    "llava_qwen",
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="sdpa",
)

# optional: still wrap with LoRA
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    bnb_config = None

# your LoRA config as before
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
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# Video Reader
from decord import VideoReader, gpu, cpu

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

# We collate to save everything in tensor format to speed-up dataloading process
# Saving the whole video clip (array) along with caption (string) will slow down iteration
# because unprocessed video clip will take up more memory due to higher resolution
# The processed video on the other hand is always 336x336 in size and fixed frame count per clip
# see: https://discuss.huggingface.co/t/slow-iteration-speed-with-and-without-keep-in-memory-true/33587
def collate_fn(example):
    video_file = example["video"]
    video_clip = read_video_decord(f'{video_file}') # change to the video decoder you want
    conversation = example["conversations"]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

    batch = processor(
        text=prompt,
        videos=video_clip,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return batch

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

def build_qwen_inputs(conversation, video_frames, tokenizer, image_processor, conv_template="qwen_1_5"):
    # conversation: list of {role, content} with <image> handled via DEFAULT_IMAGE_TOKEN
    # 1. preprocess video -> pixel_values_videos
    video = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]  # (F, C, H, W)
    pixel_values_videos = video  # or video.unsqueeze(0) depending on how you want batching

    # 2. build chat template
    conv = conv_templates[conv_template].copy()
    for turn in conversation:
        conv.append_message(turn["role"], turn["content"])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 3. tokenize and insert image tokens
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
    }


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
        self.conv_template = "qwen_1_5"   # the standard template for llava_qwen
        self.model_config = None          # we will set this after init


    def __len__(self):
        return len(self.items)

    def _resolve_path(self, p: str) -> str:
        p = p.strip()
        if not os.path.isabs(p):
            p = os.path.join(self.video_root, p) if self.video_root else p
        return p

    def __getitem__(self, idx):
        raw = self.items[idx]
        video_path = self._resolve_path(raw["video"])

        vr = decord.VideoReader(video_path, ctx=cpu(0))
        n_total = len(vr)
        n_use = min(n_total, self.num_frames)
        frame_indices = np.linspace(0, n_total - 1, n_use, dtype=int)

        # 1) Get numpy frames
        np_frames = [vr[i].asnumpy() for i in frame_indices]  # (H, W, 3) uint8

        # 2) Convert to PIL images so .size returns (width, height), not a scalar
        pil_frames = [Image.fromarray(frame) for frame in np_frames]

        # 3) Process with LLaVA helper
        image_tensors = process_images(pil_frames, self.image_processor, self.model_config)

        # Depending on your setup, image_tensors is usually a list of tensors.
        # For now, you can keep using the first element as your video tensor,
        # or adjust if you want to stack all frames.
        pixel_values_videos = image_tensors[0]

        # build conversation using LLaVA's conv_templates
        # your raw JSONL format already has role / content, but you were
        # previously converting via convert_llava_video_to_llava_next(raw)
        conversation = convert_llava_video_to_llava_next(raw)

        # assume single turn: user then assistant
        user_msg = conversation[0]
        assistant_msg = conversation[1]

        # extract user text and assistant text
        user_text = ""
        for c in user_msg["content"]:
            if c["type"] == "text":
                user_text += c["text"]
        assistant_text = ""
        for c in assistant_msg["content"]:
            if c["type"] == "text":
                assistant_text += c["text"]

        # LLaVA-Qwen chat format
        conv = copy.deepcopy(conv_templates[self.conv_template])
        # user prompt gets the <image> token
        question = DEFAULT_IMAGE_TOKEN + "\n" + user_text.strip()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], assistant_text.strip())
        prompt = conv.get_prompt()

        # tokenize, inserting IMAGE_TOKEN_INDEX at the right place
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )[0]  # remove batch dim

        attention_mask = torch.ones_like(input_ids)

        # standard LM labels with ignore index on padding
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": pixel_values_videos,  # (F, 3, H, W)
        }



class LlavaNextSimpleVideoCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features[i] has:
        #  input_ids: (seq_len_i,)
        #  attention_mask: (seq_len_i,)
        #  labels: (seq_len_i,)
        #  pixel_values_videos: (F, 3, H, W)

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # labels = [f["labels"] for f in features] if "labels" in features[0] else None

        # pad text
        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        # Create labels = input_ids, mask padding as -100
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        # stack videos: (B, F, 3, H, W)
        pixel_values_videos = torch.stack(
            [f["images"] for f in features], dim=0
        )
        batch["images"] = pixel_values_videos

        # Debug first batch: check shapes and placeholder count
        # if not hasattr(self, "_dbg"):
        #     print({k: v.shape for k, v in batch.items()})
        #     image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        #     print("placeholder count:", (batch["input_ids"] == image_token_id).sum())
        #     self._dbg = True

        return batch

# class LlavaNextVideoDataCollatorWithPadding:
#     def __init__(self, processor):
#         self.processor = processor

#     def __call__(self, features):
#         padded_inputs = self.processor.tokenizer.pad(
#             {
#                 "input_ids": [feat['input_ids'][0] for feat in features], # each element is one batch only so we slice [0]
#                 "attention_mask": [feat['attention_mask'][0] for feat in features],
#             },
#             padding=True,
#             return_tensors="pt",
#         )
        
#         labels = padded_inputs["input_ids"].clone()
#         labels[labels == self.processor.tokenizer.pad_token_id] = -100
#         padded_inputs["labels"] = labels
#         padded_inputs["pixel_values_videos"] = torch.stack([feat['pixel_values_videos'] for feat in features], dim=0)

#         return padded_inputs

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

# pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
# model_name = "llava_qwen"
# device = "cuda"
# device_map = "auto"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

# model.eval()
    
# model = LlavaNextVideoForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config,
#     device_map="auto",
# )

# image_processor = getattr(processor, "image_processor", processor)  # some processors expose .image_processor
# tokenizer = processor.tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
# tokenizer = processor
    
# train_dataset = LLaVAVideoJsonlDataset(
#     jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_train.jsonl",
#     video_root="",       # or "" if paths in JSONL are absolute
#     image_processor=image_processor,
#     tokenizer=tokenizer,                 # let your collator/trainer handle tokenization
#     num_frames=NUM_FRAMES,
#     resolution=336,
#     return_tokenized=True
# )

train_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_train.jsonl",
    video_root="",
    image_processor=image_processor,
    tokenizer=tokenizer,
    num_frames=NUM_FRAMES,
    resolution=336,
)

# let the dataset see the model config for process_images
train_dataset.model_config = model.config


test_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test.jsonl",
    video_root="",       # or "" if paths in JSONL are absolute
    image_processor=image_processor,
    tokenizer=tokenizer,                 # let your collator/trainer handle tokenization
    num_frames=NUM_FRAMES,
    resolution=336
)
# let the dataset see the model config for process_images
test_dataset.model_config = model.config

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

data_collator = LlavaNextSimpleVideoCollator(tokenizer=tokenizer)
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    # data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args=args,
)

# batch = next(iter(DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)))
# print(batch.keys())
# assert False
trainer.train()

