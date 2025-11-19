import os
import pandas as pd
import sys

# Huggingface cache directory
os.environ['XDG_CACHE_HOME'] = '/localdisk1/saiful_cache/.cache'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
import cv2
import subprocess
import whisper

# Already developed modules
sys.path.append("/localdisk1/PARK/park_vlm_finetuning/code/Prompt_Generation")
sys.path.append("/localdisk1/PARK/park_vlm_finetuning/code/vlm_instances")

# text prompt generation
from generate_prompts import *

# VLM inference functions
from QwenVL.Qwen25_7B_VL import *
from LlaVANext.LLaVA_Next_VL import *
from LlaVANext.LlaVA_Next_SFT import *

base_folder = "/localdisk1/PARK/park_vlm/Annotations/Clinical"
annotators = ["Cayla", "Nami", "Natalia"]
base_path = "/localdisk1/PARK/park_vlm/Videos"

def get_video_path(filename):
    return os.path.join(base_path, filename)

def extract_audio_with_ffmpeg(video_path, audio_path=None):
    audio_path = f"{video_path[:-4]}_audio.wav"
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # sample rate
        "-ac", "1",      # mono channel
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def extract_transcription(video_path, model_size="base"):
    audio_path = extract_audio_with_ffmpeg(video_path)
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    os.remove(audio_path)
    return transcript

def trim_video(input_path):
    '''
    If the video is longer than 60 seconds, trim it to 60 seconds
    '''
    output_path = input_path[:-4]+"_trimmed.mp4"

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate max number of frames for 60 seconds
    max_frames = int(fps * 60)

    # Use original frame count if less than 60 seconds
    frames_to_write = min(max_frames, total_frames)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write frames up to 60 seconds
    frame_count = 0
    while frame_count < frames_to_write:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    # Release everything
    cap.release()
    out.release()
    return output_path

def get_test_responses(llm_name="QwenVL", output_path="./"):
    # store the responses as list of dictionary here
    dataset = []

    question_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    for annotator in annotators[0:1]:
        # this has the filenames we want to look at
        annotation_file = os.path.join(base_folder, annotator, f"{annotator.lower()}_common_final.csv")
        df = pd.read_csv(annotation_file)
        filenames = df["video"]
        
        for file in tqdm(filenames):
            video_path = get_video_path(file)
            transcript = extract_transcription(video_path, model_size="base") # collect speech transcript
            video_path = trim_video(video_path) # make sure its max 60 seconds
            
            for qid in tqdm(question_ids):
                prompt = generate_text_prompt(question_id=qid, transcript=transcript)
                
                if llm_name == "QwenVL":
                    response = get_model_response(video_path, prompt, qid)
                elif llm_name == "LlaVANext":
                    response = get_LlaVANext_response(video_path, prompt, qid)
                elif llm_name == "LlaVANext_SFT":
                    response = get_LlaVANextSFT_response(video_path, prompt, qid)
                elif llm_name == "Other":
                    raise Exception(f"{llm_name} not implemented")

                # construct the dictionary -- filename, question-id, prompt, llm_name, llm_response
                item = {
                    "filename": file,
                    "question-id": qid,
                    "prompt": prompt,
                    "llm_name": llm_name,
                    "llm_response": response
                }

                dataset.append(item)

        # intentional; not a debugging break
        # As we are just evaluating on the test set which is available for all annotators, no need to go through all annotators
        break

    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_path = os.path.join(output_path, f"{llm_name}_test_responses_v0.csv")
    dataset_df.to_csv(dataset_path, index=False)
    return

if __name__ == "__main__":
    # pass it as a function param later
    # llm_name = "QwenVL"
    # llm_name = "LlaVANext"
    llm_name = "LlaVANext_SFT"
    output_path = "/localdisk1/PARK/park_vlm_finetuning/model_outputs"
    get_test_responses(llm_name, output_path)

    