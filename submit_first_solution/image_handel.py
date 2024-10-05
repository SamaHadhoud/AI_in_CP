import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
import os, re
def initialize_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

def generate_description(image_path, problem_description, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"In the context of the following problem description:\n\n{problem_description}\n\nDescribe this image in detail"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def process_and_edit_problems(problem_list, directory, model, processor):
    for problem_name in problem_list:
        md_file_path = os.path.join(directory, f"{problem_name}.md")
        
        with open(md_file_path, 'r') as file:
            content = file.read()

        # Find all image placeholders
        image_placeholders = re.finditer(r'{{PHOTO_ID:(\d+)\|WIDTH:\d+}}', content)
        new_content = content

        for match in image_placeholders:
            image_id = match.group(1)
            image_path = os.path.join(directory, f"{image_id}.jpg")
            
            # Get the context up to this image placeholder or constraints, whichever comes first
            context = get_context(content, match.start())
            
            if os.path.exists(image_path):
                description = generate_description(image_path, context, model, processor)
                
                # Replace the image placeholder with the description
                new_content = new_content.replace(
                    match.group(0),
                    f"Image Description: {description}",
                    1  # Replace only the first occurrence
                )
                
                print(f"Processed image {image_id} for problem: {problem_name}")
            else:
                print(f"Image {image_id} not found for problem: {problem_name}")

        # Write the updated content back to the file
        with open(md_file_path, 'w') as file:
            file.write(new_content)
        
        print(f"Updated problem file: {problem_name}")

def get_context(content, image_start):
    constraints_index = content.find("# Constraints")
    if constraints_index == -1 or constraints_index > image_start:
        return content[:image_start].strip()
    else:
        return content[:constraints_index].strip()

# Initialize the model
model, processor = initialize_model()

# Example usage
problem_list = ["cheeseburger_corollary_ch1", "cheeseburger_corollary_ch2", "dim_sum_delivery", "two_apples_a_day", "road_to_nutella"]  # Add your problem names here
directory = "./dataset/2023/practice_Qwen/Qwen2-VL-2B-Instruct"  # Replace with your actual directory path

process_and_edit_problems(problem_list, directory, model, processor)