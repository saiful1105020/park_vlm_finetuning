import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from peft import PeftModel

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
ADAPTER_PATH = "/localdisk1/PARK/park_vlm_finetuning/checkpoints/unstructured_sft_run_1/checkpoint-20"

base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto" 
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
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


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Does this person in the video demonstrate signs of Parkinson's disease?"},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
video_path = "/localdisk1/PARK/park_vlm/Videos/2024-04-15T14%3A27%3A53.507Z_RyMSyFtxxqPWzGGhc1YUBvSm2xy2_speech.mp4"

container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)
inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

output = model.generate(**inputs_video, max_new_tokens=256, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
