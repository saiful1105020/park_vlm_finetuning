import os
os.environ['XDG_CACHE_HOME'] = '/localdisk1/saiful_cache/.cache'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("/localdisk1/PARK/park_vlm/Annotations/LLM_Prompting/Prompt_Generation/")
from generate_prompts import *
# generate_text_prompt

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def get_model_response(video_path, text_prompt, question_id):
    max_pixels_per_question = {
        1: 256*144,
        2: 320*240,
        3: 256*144,
        4: 320*240,
        5: 320*240,
        6: 320*240,
        7: 320*240,
        8: 256*144,
        9: 320*240,
        10: 320*240,
        11: 320*240,
        12: 320*240,
        13: 320*240
    }

    fps_per_question = {
        1: 2,
        2: 1,
        3: 2,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 2,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1
    }
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": max_pixels_per_question[question_id],
                    "fps": fps_per_question[question_id],
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

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

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    # generated_ids = model.generate(**inputs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return response

if __name__ == "__main__":
    filename = "2024-01-31T17%3A14%3A45.278Z_ZdL4aKmd5uSJLFc5S0ndhX4EFc92_speech.mp4"
    video_path = get_video_path(filename)
    
    text_prompt = generate_text_prompt(video_path, question_id=1)
    model_response = get_model_response(video_path, text_prompt, question_id=1)
    print(model_response)
    assert False

    text_prompt = generate_text_prompt(video_path, question_id=2)
    model_response = get_model_response(video_path, text_prompt, question_id=2)
    print(model_response)
