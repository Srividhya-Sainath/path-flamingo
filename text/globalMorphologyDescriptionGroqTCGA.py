""" Modified from LLaVa Med Source Cde"""
## TODO: Add correct credits before we release our code

import json
import random
import textwrap
import os
from typing import List
from pathlib import Path
from openai import OpenAI
import json
import ast
import random
from tqdm import tqdm
import pandas as pd
from groq import Groq
import time
import tenacity

from transformers import AutoTokenizer

import instruct_few_shot_examples_tcga

### 
# 1. ..
# 2. get Groq API key
# 3. export GROQ_API_KEY=<your-api-key-here> via Terminal
# 4. modify main function below


conv_to_str = lambda conv: "\n\n".join([("User: " if x["from"] == "human" else "Assistant: ") + x["value"] for x in conv])

def read_json(file_path: str):
    "Read a JSON file."
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_json(file_path: str, data):
    "Write a list of dictionaries to a JSON file."
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def read_jsonl(file_path: str):
    "Read a JSONL file and return a list of dictionaries."
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(file_path: str, data):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


@tenacity.retry(wait=tenacity.wait_fixed(30), stop=tenacity.stop_after_attempt(8))
def get_response(client, model, messages):

    completion = client.chat.completions.create(
        #model='mixtral:8x7b-instruct-v0.1-q8_0', # OLLAMA
        # model="mixtral-8x7b-32768", # GROQ
        model=model, # GROQ
        #model="llama3-8b-8192", # GROQ
        #model="mistralai/Mixtral-8x7B-Instruct-v0.1", # VLLM
        messages=messages
    )
    return completion.choices[0].message.content



class PromptGenerator:
  @staticmethod
  def few_shot_messages_gen(query_context):

    starters = [
        "What global morphological features do you observe in the image?",
        "Can you describe the major morphological characteristics seen in the image?",
        "What prominent features can be observed in this image?",
        "What key morphological traits stand out in the image?",
        "What overall structural patterns are visible in the image?",
        ]

    random.shuffle(starters)
    starters = "- " + "\n- ".join(starters)

    messages = [
    {"role": "system", "content": f"""You are an AI assistant specialized in pathology.

Your task is to generate a conversation between a person (User) inquiring about the image and you (Pathology Assistant) responding to their questions. The conversation should proceed as though both the User and Assistant are viewing the image directly.

- Focus on the morphological aspects of the image.
- Avoid quoting or referencing specific terms from the text report.
- Phrase your answers as if you actually see the image.
- Prioritize features directly related to the diagnosis.
- In the answer, do not mention any diagnosis or grading.
- Do not introduce questions beyond the given category, even if they seem relevant to a pathologist’s typical inquiries.
- Good starter questions are:
        {starters}
  - Include at least one question and answer pair that discuss the exact diagnosis of the microscopic image (if the diagnosis is available from the pathology report). 
  """},
    ]
    for ex in instruct_few_shot_examples_tcga.fs: #### TODO REMOVE IF MORE SAMPLES SHALL BE ADDED
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
  
def is_valid_json(json_str):
    try:
        json.dumps(json_str)
        return True
    except Exception as e:
        return False

def contains_conversation(message):
    #starts_with_user = message.strip().lower().startswith("User:".lower())
    contains_user = "User:".lower() in message.lower()
    contains_assistant = "Assistant:".lower() in message.lower()
    #return starts_with_user and contains_assistant
    return contains_user and contains_assistant

def not_contains_kw(message):
    kws = ["context", "caption", "information provided"]
    return not any(kw.lower() in message.lower() for kw in kws)


def is_valid(message):
    try:
        return is_valid_json(message) and contains_conversation(message) and not_contains_kw(message)
    except Exception as e:
        return False

def main(
    in_directory="/TCGA/TCGA_Reports.csv", 
    max_tries=7,
    out_dir="/mnt/bulk/dferber/LLM_Inference/Generate_VLM_Instruction_Data/TCGA/Instruction/Data",
    skip_until=None,
    model=None,
    max_tokens_allowed=3000,
):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

    tcga = pd.read_csv(in_directory)

    output = []

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY") #dykexxxx@gmail
    )
    
    save_path = Path(out_dir) / "morphologyDescription.jsonl"

    with open(save_path, "a") as file:

        for idx, row in tqdm(tcga.iterrows(), total=len(tcga)):

            if isinstance(skip_until, int) and idx < skip_until:
                continue
            
            sample_description = row["text"]
            context = PromptGenerator.wrap_gen_message(sample_description)
            
            tokens = tokenizer.tokenize(str(context)) # only approximate

            if len(tokens) > max_tokens_allowed:
                continue

            n_tries = 0
            response = get_response(client, model, context)

            # Returns the last "non-valid" response if it fails
            while not is_valid(response) and n_tries < max_tries:
                response = get_response(client, model, context)
                n_tries += 1

            if idx % 10 == 0:
                print(response)

            output = {"file_path": row["patient_filename"], "result": response}
            file.write(json.dumps(output) + "\n")
            file.flush()

            time.sleep(10)
            
            print("Done.")

if __name__ == "__main__":
    main(
        in_directory="/mnt/bulk-io/vidhya/VLM/vidhya/VQA/TCGA_Reports.csv",
        max_tries=3,
        out_dir="/mnt/bulk-io/vidhya/VLM/vidhya/VQA/InstructData/TCGA/",
        skip_until=235, # set to 5000
        model="llama3-70b-8192",
        max_tokens_allowed=3000, # adjust for each model from here: https://console.groq.com/settings/limits -> Tokens per Minute
    )
