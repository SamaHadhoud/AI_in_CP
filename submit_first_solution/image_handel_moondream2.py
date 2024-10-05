import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def initialize_model():
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

def generate_description(image_path, problem_description, model, tokenizer):
    image = Image.open(image_path)
    enc_image = model.encode_image(image)
    
    prompt = f"""
    In the context of the following problem description:

{problem_description}

Describe this image in detail"""

    description = model.answer_question(enc_image, prompt, tokenizer)
    return description

def get_context(content, image_start):
    constraints_index = content.find("# Constraints")
    if constraints_index == -1 or constraints_index > image_start:
        return content[:image_start].strip()
    else:
        return content[:constraints_index].strip()

def process_and_edit_problems(problem_list, directory, model, tokenizer):
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
                description = generate_description(image_path, context, model, tokenizer)
                
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


# Initialize the model
model, tokenizer = initialize_model()

# Example usage
problem_list = ["dim_sum_delivery"]  # Add your problem names here
directory = "./dataset/2023/practice"  # Replace with your actual directory path

process_and_edit_problems(problem_list, directory, model, tokenizer)