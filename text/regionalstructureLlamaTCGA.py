import json
import os
import random
import time
import openai
from pathlib import Path
from typing import List

import pandas as pd
import tenacity
from tqdm import tqdm
from transformers import AutoTokenizer

import regional_structure_few_shot_examples_tcga

# Lambda function to convert conversations to string format
conv_to_str = lambda conv: "\n\n".join([("User: " if x["from"] == "human" else "Assistant: ") + x["value"] for x in conv])

def read_json(file_path: str):
    "Read a JSON file."
    with open(file_path, "r") as f:
        return json.load(f)

def write_jsonl(file_path: str, data):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

# @tenacity.retry(wait=tenacity.wait_fixed(30), stop=tenacity.stop_after_attempt(8))
# def get_response(client, model, messages):
#     """
#     Function to get response from OpenAI API.
#     """
#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages
#     )
#     return completion.choices[0].message.content
@tenacity.retry(wait=tenacity.wait_fixed(30), stop=tenacity.stop_after_attempt(8))
def get_response(model, messages):
    """
    Function to get a response from the OpenAI API.
    """
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

class PromptGenerator:
    @staticmethod
    def few_shot_messages_gen(query_context):
        starters = [
            "Can you describe the pattern of tumor infiltration?",
            "What observations can you make about the tissue architecture on this slide?",
            "What are the notable features of the cellular morphology in this slide?",
        ]
        random.shuffle(starters)
        starters = "- " + "\n- ".join(starters)

        messages = [
    {"role": "system", "content": f"""You are an AI assistant specialized in histopathology slide interpretation. I will provide you with descriptions and diagnostic results related to histopathology slides. Your task is to create a dialogue as if you are directly observing and analysing the slide.

Guidelines:

Observation: Assume you are directly viewing the slide and provide detailed descriptions of the pathological features when answering the questions. Ensure each respose is directly related to the specific morphological feature being asked about, and do not mention any diagnosis, prognosis, or grading.
Tone: Maintain a professional and informative conversational style, emulating the perspective of a visual AI assistant specialising in histopathology.
Diaglogue Structure: Questions must strictly focus on the following three morphological aspects and should only be asked if relevant information is provided in the pathology report. Avoid generating any questions or comments outside of these areas.
The questions should include:
1. Tumor Infiltration (Including Vascular and Neural Invasion): If the pathology report mentions tumor infiltration, ask a question regarding how the tumor invades surrounding tissues. For example: "Can you describe the pattern of tumor infiltration?" The answer should describe patterns such as local tissue invasion, perineural invasion, or vascular involvement, excluding nuclear features.
2. Tissue Architecture Observation: If the pathology report mentions tissue architecture, ask a question regarding the general structural arrangement of cells and tissues. For example: "What observations can you make about the tissue architecture on this slide?" The answer should focus on features such as glandular formations, solid sheets of cells, or stromal alterations.
3. Cellular Morphology: If the pathology report includes observations about cellular and nuclear characteristics, ask a question about these features. For example: "What are the notable features of the cellular morphology in this slide?" The answer should describe aspects like cell size, shape, arrangement, nuclear size, nuclear shape, chromatin texture, the presence of nucleoli, and mitotic figures (including any abnormal mitoses), without linking these features to a diagnosis or including any grading.
     
Scope Limiations: Do not generate questions outside of three specified categories. Only ask questions if the corresponding information is explicitly mentioned in the pathology report. If certain information is missing, omit the question for that category.
Avoid Additional Questions: Do not introduce questions beyong the given categories, even if they seem relevant to a pathologist's typical inquiries.
Ensure Direct Relevence: Make sure all questions and answers are directly connected to the provided pathology report. If certain information is not available, omit questions that would require that information.
Consistent Question Phrasing: You may vary the wording to reflect a professional's questioning style but ensure the essence of the questions remains within the specified categories.
Length: Ensure the entire dialogue does not exceed 600 words, providing accurate morphological description. 
        {starters}
  - Include at least one question and answer pair. 
  """},
    ]
        for ex in regional_structure_few_shot_examples_tcga.fs:
            messages += [
                {"role": "user", "content": "Here is a great example conversation:\n"},
                {"role": "assistant", "content": conv_to_str(ex["conversations"])},
            ]
        messages.append({"role": "user", "content": "Here is the pathology report:\n" + query_context})
        return messages

    @staticmethod
    def context_gen(sample_text):
        return sample_text.strip()

    @staticmethod
    def wrap_gen_message(sample):
        text = PromptGenerator.context_gen(sample)
        context = PromptGenerator.few_shot_messages_gen(text)
        return context

def main(
    in_directory="/path/to/TCGA_Reports.csv",
    max_tries=7,
    out_dir="/path/to/output",
    skip_until=None,
    model="gpt-4-1106-preview",
    max_tokens_allowed=3000,
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",
):
    """
    Main function to process data and generate responses using the OpenAI API.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    tcga = pd.read_csv(in_directory)
    output = []

    # Initialize OpenAI client
    # client = openai.OpenAI(
    #     base_url=base_url,
    #     api_key=api_key,
    # )
    openai.api_base = base_url
    openai.api_key = api_key

    save_path = Path(out_dir) / "regionalstructureTCGA.jsonl"

    with open(save_path, "a") as file:
        for idx, row in tqdm(tcga.iterrows(), total=len(tcga)):
            if isinstance(skip_until, int) and idx < skip_until:
                continue

            sample_description = row["text"]
            context = PromptGenerator.wrap_gen_message(sample_description)
            tokens = tokenizer.tokenize(str(context))

            if len(tokens) > max_tokens_allowed:
                continue

            n_tries = 0
            #response = get_response(client, model, context)
            response = get_response(model, context)

            while not response and n_tries < max_tries:
                #response = get_response(client, model, context)
                response = get_response(model, context)
                n_tries += 1

            if idx % 10 == 0:
                print(response)

            output_data = {"file_path": row["patient_filename"], "result": response}
            file.write(json.dumps(output_data) + "\n")
            file.flush()

            time.sleep(10)  # Adjust delay as needed
    print("Processing complete.")

if __name__ == "__main__":
    main(
        in_directory="/mnt/bulk-io/vidhya/VLM/vidhya/VQA/TCGA_Reports.csv",
        max_tries=3,
        out_dir="/mnt/bulk-io/vidhya/VLM/vidhya/VQA/InstructData/TCGA",
        skip_until=None,
        model="gpt-4-1106-preview",
        max_tokens_allowed=5000, #TODO
        base_url="http://localhost:8080/v1",
        api_key="sk-no-key-required",
    )
