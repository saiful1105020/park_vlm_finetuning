"""
Version History:
- v0: Initial version, runs successfully, but converges too quickly. Loss plateaues around 3.75.
"""
import os

# Must come BEFORE any torch/transformers imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from torch.nn.utils.rnn import pad_sequence
from torch import bfloat16

from typing import List, Dict, Any, Optional
import decord

# from llava.model.builder import load_pretrained_model
MAX_LENGTH = 256
BATCH_SIZE = 1
NUM_FRAMES = 1 # more frames -> more VRAM needed
DATASET_PATH = "/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured" # path where to save the dataset
OUTPUT_DIR = "/localdisk1/PARK/park_vlm_finetuning/checkpoints/unstructured_sft_llava_qwen" # path where to save the checkpoints

USE_LORA = False
USE_QLORA = True
MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"
device = torch.device("cuda:0")
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
dtypes = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16"
}
# print(torch_dtype) -- torch.bfloat16

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path = MODEL_ID,
    model_base = None,
    model_name = "llava_qwen",
    torch_dtype = dtypes[torch_dtype],
    device_map = {"": 0},
    attn_implementation = "sdpa"
)

# Ensured that the model is fully using the bf16 data type
model_config = model.config

# Disable anyres + unpad, use flat patches and square images instead
model_config.image_aspect_ratio = "pad"      # or "square"
model_config.mm_patch_merge_type = "flat"

# Make sure the inner model config sees the same values
if hasattr(model, "get_model"):
    inner = model.get_model()
    inner.config.image_aspect_ratio = "pad"
    inner.config.mm_patch_merge_type = "flat"
else:
    # fallback if no get_model()
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type = "flat"

core = model.get_base_model() if hasattr(model, "get_base_model") else model
vt = core.get_vision_tower()

# vision encoder
vt.to(device=device, dtype=torch_dtype)

# mm projector (name can vary a bit, so check both common ones)
if hasattr(core, "mm_proj"):
    core.mm_proj.to(device=device, dtype=torch_dtype)

if hasattr(vt, "config"):
    vt.config.image_aspect_ratio = "pad"

if hasattr(model.config, "image_aspect_ratio"):
    model.config.image_aspect_ratio = "pad"

# optional: still wrap with LoRA
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
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

if USE_QLORA:
    model = get_peft_model(model, lora_config)

# So far, all weights are in bfloat16, only Lora weights are in float32

model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

for p in model.parameters():
    p.requires_grad = False
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad = True

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
        resolution: int = 224,
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

        # Always sample exactly self.num_frames frames
        if n_total >= self.num_frames:
            # uniform sampling over the video
            frame_indices = np.linspace(0, n_total - 1, self.num_frames, dtype=int)
        else:
            # not enough frames: repeat last frame to reach self.num_frames
            base_indices = np.linspace(0, n_total - 1, n_total, dtype=int)
            pad_count = self.num_frames - n_total
            pad_indices = np.full(pad_count, base_indices[-1], dtype=int)
            frame_indices = np.concatenate([base_indices, pad_indices])

    
        # 1) Get numpy frames
        np_frames = [vr[i].asnumpy() for i in frame_indices]  # (H, W, 3) uint8

        # 2) Convert to PIL images so .size returns (width, height), not a scalar
        pil_frames = [Image.fromarray(frame) for frame in np_frames]
        image_sizes = [img.size for img in pil_frames]  # each is (W, H)

        # 3) Process with LLaVA helper
        image_tensors = process_images(pil_frames, self.image_processor, self.model_config)
        
        # Depending on your setup, image_tensors is usually a list of tensors.
        # For now, you can keep using the first element as your video tensor,
        # or adjust if you want to stack all frames.
        # pixel_values_videos = image_tensors[0]
        pixel_values_videos = image_tensors
        
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
        question = DEFAULT_IMAGE_TOKEN + "\n" + user_text.strip()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], assistant_text.strip())
        prompt = conv.get_prompt()

        # 3. tokenize and insert IMAGE_TOKEN_INDEX
        # IMPORTANT: do NOT use return_tensors="pt" here
        input_ids_list = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
        )  # this is a plain list[int]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)  # (seq_len,)
        attention_mask = torch.ones_like(input_ids)

        # standard LM labels (we'll re-pad in collator)
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        pixel_values_videos = pixel_values_videos.to(dtype=torch_dtype)
    
        return {
            "input_ids": input_ids,          # (L,)
            "attention_mask": attention_mask,
            "labels": labels,
            "images": pixel_values_videos,         # (F, 3, H, W), float32
            "image_sizes": image_sizes,  # list of (W, H) per frame
        }

class LlavaNextSimpleVideoCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features[i]:
        #   input_ids: (L_i,)
        #   attention_mask: (L_i,)
        #   labels: (L_i,)
        #   images: (F, 3, H, W)

        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        images = [f["images"] for f in features]
        image_sizes_lists = [f["image_sizes"] for f in features]  # list of lists

        # 1) pad input_ids to (B, L_max)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )

        # 2) pad attention_mask to (B, L_max)
        attention_masks = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        # 3) pad labels to (B, L_max) with ignore_index=-100
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        # 4) stack images -> (B, F, 3, H, W)
        # images = torch.stack(images, dim=0)
        images = torch.stack([f["images"] for f in features], dim=0) # (B, F, 3, H, W)
        
        # flatten all sizes into one list (for all images across batch)
        flat_sizes = [size for sizes in image_sizes_lists for size in sizes]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "images": images,
            "image_sizes": flat_sizes,
        }

        return batch


train_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_train.jsonl",
    video_root="",
    image_processor=image_processor,
    tokenizer=tokenizer,
    num_frames=NUM_FRAMES,
    resolution=224,
)

# let the dataset see the model config for process_images
train_dataset.model_config = model.config


test_dataset = LLaVAVideoJsonlDataset(
    jsonl_path="/localdisk1/PARK/park_vlm_finetuning/data/SFT_unstructured/vlm_conversations_test.jsonl",
    video_root="",       # or "" if paths in JSONL are absolute
    image_processor=image_processor,
    tokenizer=tokenizer,                 # let your collator/trainer handle tokenization
    num_frames=NUM_FRAMES,
    resolution=224
)
# let the dataset see the model config for process_images
test_dataset.model_config = model.config

gradient_accumulation_steps = 4

args = TrainingArguments(
    # args related to training
    output_dir = OUTPUT_DIR,
    eval_strategy = 'steps',
    eval_steps=1000,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = gradient_accumulation_steps,
    learning_rate = 2e-05,
    max_steps = 2000, # adjust this depending on your dataset size
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.1,
    remove_unused_columns=False,

    # args related to eval/save
    logging_steps = 5,
    save_strategy = 'steps',
    # save_steps=len(train_dataset)//(BATCH_SIZE*gradient_accumulation_steps), # save every epoch
    save_steps = 100,
    save_total_limit = 20,
    # fp16=(torch_dtype == torch.float16),
    # bf16=(torch_dtype == torch.bfloat16),
    fp16=False,
    bf16=False,
    # fp16_full_eval = True,
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

# print(model_config.mm_patch_merge_type, model_config.image_aspect_ratio)

# print(model.get_model().config.mm_patch_merge_type, model.get_model().config.image_aspect_ratio)

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    # data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args=args
)

trainer.train()