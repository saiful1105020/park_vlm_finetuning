import os
os.environ['XDG_CACHE_HOME'] = '/localdisk1/saiful_cache/.cache'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("/localdisk1/PARK/park_vlm_finetuning/code/Prompt_Generation")
import torch
import av
import numpy as np

# generate_text_prompt
from generate_prompts import *

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
warnings.filterwarnings("ignore")

base_path = "/localdisk1/PARK/park_vlm/Videos"
def get_video_path(filename):
    return os.path.join(base_path, filename)

pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map, attn_implementation = "sdpa")  # Add any other thing you want to pass in llava_model_args
model.eval()

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)

    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

def get_LlaVAVideo_response(video_path, text_prompt, question_id=None):
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{text_prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    try:
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    except Exception as e:
        text_outputs = ""
        print(f"Exception in parsing LlaVAQwen response for question id {question_id}: {e}")

    return text_outputs

if __name__ == "__main__":
    filename = "2024-01-31T17%3A14%3A45.278Z_ZdL4aKmd5uSJLFc5S0ndhX4EFc92_speech.mp4"
    video_path = get_video_path(filename)
    
    text_prompt = generate_text_prompt(question_id=1)    
    model_response = get_LlaVAVideo_response(video_path, text_prompt, question_id=1)
    print(model_response)

    text_prompt = generate_text_prompt(question_id=2)
    model_response = get_LlaVAVideo_response(video_path, text_prompt, question_id=2)
    print(model_response)