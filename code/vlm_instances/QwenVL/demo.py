import os
os.environ['XDG_CACHE_HOME'] = '/localdisk1/saiful_cache/.cache'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Messages containing a local video path and a text query
long_video_path = "/localdisk1/PARK/park_vlm/Videos/2023-04-24T18%3A04%3A47.883Z_c7SEeGMrE7Xtvwe2HkZ3S959mvw1_resting_face.mp4"
short_video_path = "/localdisk1/PARK/colearning/data/finger_tapping/videos/2017-10-12T20-07-10-147Z53-finger_tapping.mp4"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": long_video_path,
                "max_pixels": 240 * 320,
                "fps": 2.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)