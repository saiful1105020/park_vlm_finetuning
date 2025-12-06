import os

question = {
        1: 'Please describe whether the person demonstrates any difficulty through their facial expressions. Some examples of visible difficulty include furrowed brow, squinting eyes, clenched jaw, tight lips, head hanging low, sighing, wrinkled forehead, etc. Mention such specific details when found. End output with a final answer choice: \"Yes\" or \"No\".',
        2: 'Mention if the background is overloaded (i.e., too many things), or the lighting condition is inappropriate (i.e., too dark or overlit). Otherwise, just output \"normal background\".',
        3: 'Was there anything abnormal about the person’s eye blink rate? For example, they may not be blinking at all, or they may have reduced or increased blink rate compared to a normal person. If there is nothing abnormal, output \"normal blinking\".',
        4: 'How far is the camera? A good position of the camera would be when the upper half of the subject\'s body remains visible, while the lower half is not captured in the frame. If the upper body is only partly visible, the camera is too close. If the lower body is also visible, the camera is too far.',
        5: 'Is the subject coherent in what they are speaking? Are they delivering an easy to understand story, or are they deterring a lot from the central topic and delivering an unorganized speech?',
        6: 'Please indicate whether the subject was able to follow the instructions while completing the task. If the subject was doing something differently, please describe.',
        7: 'Indicate the extent of lips parting when the subject is not saying anything (i.e., always/most of the times/sometimes/very few times/never).',
        8: 'Indicate which of the following are true for the subject: (i) The individual\'s face appears blank and emotionless, even when they are trying to express an emotion. (ii) The expression is weak or asymmetrical, and the individual has difficulty holding an expression (e.g., smile) for an extended period. Also mention if you observe other facial expression abnormalities. If there is nothing abnormal, output \"nothing abnormal\".',
        9: 'Document any abnormal signs observed in body parts other than the face. This includes, but is not limited to, tremors in the hands, involuntary shaking or rhythmic movements of the upper or lower extremities, stiffness or rigidity in the limbs, reduced arm swing while speaking, or any signs of bradykinesia (slowness of movement). Additionally, note any abnormal postures, difficulty in maintaining balance, or other motor irregularities that may be indicative of Parkinson\'s disease.',
        10: 'Provide a brief description of the subject\'s perceived state of mind, noting whether they appear energetic, exhausted, calm, confused, or exhibit any other relevant emotional or cognitive cues.',
        11: 'Indicate whether any other individuals were present in the background. If so, provide a brief description (e.g., \"An older male is visible in the background\"). Conclude with a final answer: \"Yes\" or \"No\".',
        12: 'Indicate whether the subject is using complex sentences that are difficult to understand. Conclude with a final answer: \"Yes\" or \"No\".',
        13: 'Indicate if any body part critical for this task is partially visible (or invisible). For example, was the subject wearing a mask that may have obstructed important visual information? Or did the face go out of frame while the subject was completing the task?'
    }

def template(question_id, transcript=None):
    preamble = "Imagine you are a clinician specializing in movement disorders. Rely on your knowledge of neurology and clinical care. Now, you are watching a home-recorded video of a person performing some tasks used to assess Parkinson's disease. No experts supervise the person, so there can be different types of noise, or the person may not follow the task instructions properly. The person can also show symptoms that may be associated with having Parkinson's disease. Focus on the noises, task instructions, user compliance, and possible symptoms of Parkinson's disease while answering the question."

    task_instruction = "The person will talk about a recent book they have read or a movie or TV show they have watched. The person will speak for approximately one minute. They should be front-facing the camera, and their face must be visible in the recording frame. There should not be any other person visible in the recording frame. The background should not be dark or overlit and should have good contrast against the person\'s face. For this task, the face is the most crucial body part you should focus on. However, you should also observe other body parts for relevant symptoms or signs of Parkinson's disease."

    if question_id in [1,2,3,4,7,8,9,10,11,13]:
        nudge_instruction = "Answer the question about what is happening in the video. Keep your response short and precise."
    elif question_id in [5,6,12]:
        nudge_instruction = "Analyze the provided text transcription of the person’s speech and answer the question about what is happening in the video."

    text = ""
    if question_id in [1,2,3,4,7,8,9,10,11,13]:
        text = f"{preamble}\n\nTask instructions: {task_instruction}\n\n{nudge_instruction}\n\nQuestion: {question[question_id]}"
    elif question_id in [5,6,12]:
        text = f"{preamble}\n\nTask instructions: {task_instruction}\n\n{nudge_instruction}\n\nTranscript: {transcript}\n\nQuestion: {question[question_id]}"
    return text

def generate_text_prompt(question_id, transcript=None):
    text_prompt = template(question_id, transcript)
    return text_prompt

def main():
    prompt = generate_text_prompt(question_id=1)
    print(prompt)
    return

if __name__ == "__main__":
    main()