import json
import os
import random
import time
from pathlib import Path
from typing import List

import pandas as pd
import tenacity
from tqdm import tqdm
from transformers import AutoTokenizer

import instruct_few_shot_examples_tcga

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

@tenacity.retry(wait=tenacity.wait_fixed(30), stop=tenacity.stop_after_attempt(8))
def get_response(client, model, messages):
    """
    Function to get response from OpenAI API.
    """
    completion = client.chat.completions.create(
        model=model,
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
- Avoid mentioning the Pathology report at all.
- Phrase your answers as if you actually see the image.
- In the answer, DO NOT mention any diagnosis or grading.
- Do not introduce questions beyond the given category, even if they seem relevant to a pathologist’s typical inquiries.
- Good starter questions are:
        {starters}
  - Include at least one question and answer pair that discuss the exact morphology of the microscopic image (if the morphology description is available from the pathology notes). 
  """},
    ]
        for ex in instruct_few_shot_examples_tcga.fs:
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
    import openai
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    save_path = Path(out_dir) / "morphologyDescription.jsonl"

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
            response = get_response(client, model, context)

            while not response and n_tries < max_tries:
                response = get_response(client, model, context)
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
