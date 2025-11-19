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

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

base_path = "/localdisk1/PARK/park_vlm/Videos"
def get_video_path(filename):
    return os.path.join(base_path, filename)

# Load model from checkpoint
# from peft import PeftModel

# model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
# ADAPTER_PATH = "/localdisk1/PARK/park_vlm_finetuning/checkpoints/unstructured_sft_run_1/checkpoint-20"

# base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True,
#     device_map="auto" 
# )
# model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
# model.eval()

# Load LlaVA-Next base model
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto" 
)
model.eval()
processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def get_LlaVANext_response(video_path, text_prompt, question_id=None):
    # max_pixels_per_question = {
    #     1: 256*144,
    #     2: 320*240,
    #     3: 256*144,
    #     4: 320*240,
    #     5: 320*240,
    #     6: 320*240,
    #     7: 320*240,
    #     8: 256*144,
    #     9: 320*240,
    #     10: 320*240,
    #     11: 320*240,
    #     12: 320*240,
    #     13: 320*240
    # }

    # fps_per_question = {
    #     1: 2,
    #     2: 1,
    #     3: 2,
    #     4: 1,
    #     5: 1,
    #     6: 1,
    #     7: 1,
    #     8: 2,
    #     9: 1,
    #     10: 1,
    #     11: 1,
    #     12: 1,
    #     13: 1
    # }

    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    num_frames = 32
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=256, do_sample=False)
    raw_output = processor.decode(output[0][2:], skip_special_tokens=True)
    try:
        response = raw_output.split("ASSISTANT:")[-1].strip().split("\n")[0].strip()
    except Exception as e:
        response = raw_output.strip()
        print(f"Exception in parsing LlaVANext response for question id {question_id}: {e}")

    return response

if __name__ == "__main__":
    filename = "2024-01-31T17%3A14%3A45.278Z_ZdL4aKmd5uSJLFc5S0ndhX4EFc92_speech.mp4"
    video_path = get_video_path(filename)
    
    text_prompt = generate_text_prompt(question_id=1)
    model_response = get_LlaVANext_response(video_path, text_prompt, question_id=1)
    print(model_response)

    text_prompt = generate_text_prompt(question_id=2)
    model_response = get_LlaVANext_response(video_path, text_prompt, question_id=2)
    print(model_response)
