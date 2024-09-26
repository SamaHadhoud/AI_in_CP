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

def generate_description(image_path, model, tokenizer):
    # Load the image
    image = Image.open(image_path)

    # Encode the image
    enc_image = model.encode_image(image)
    
    # prompt = "Describe this image in detail"
    # Prepare the prompt
    prompt = """
    In the context of the following problem description:

You and your friends have drawn a really big connected graph in sidewalk chalk with \(N\) nodes (numbered from \(1..N\)) and \(M\) edges. \(Q\) times, there will be a race from node \(a_i\) to node \(b_i\)​, with a chance to win a coveted hazelnut chocolate snack. By the unbreakable rules of hop-scotch, everyone must travel along a path from node \(a_i\) to node \(b_i\) using edges in the graph, alternating which foot touches the ground at each node, starting each race with their left foot on \(a_i\).

Your friends will make a mad dash for the chocolate along the **shortest path** from \(a_i\) to \(b_i\)​. You on the other hand are looking for a more interesting challenge, and are allowed to take *any* path, potentially including any nodes (even \(b_i\)) multiple times. You want to end at node \(b_i\)​, but with the following conditions:
 - You must finish on a different foot from everyone who took the shortest path.
- To make things interesting, you'd like to minimize the number of edges you travel through more than once.

*An illustration of the first sample. Your friends take a shortest path (blue), and you can take the path in red. The path in red travels through \(1\) edge multiple times: the edge connecting nodes \(6\) and \(8\).*

For each query, is it possible to fulfill your two conditions? If so, add the minimum number of edges you have to travel through multiple times to your answer. If not, add \(-1\) to your answer.

Describe this image in detail"""

    # Generate the description
    description = model.answer_question(enc_image, prompt, tokenizer)
    return description

# Color formatting for console output
class color:
   BLUE = '\033[94m'
   BOLD = '\033[1m'
   END = '\033[0m'
model, tokenizer = initialize_model()
# Example usage
image_path = "/home/sama.hadhoud/Documents/NLP701/project/nlp_project/submit_first_solution/dataset/2023/practice/903178538089777.jpg"
description = generate_description(image_path, model, tokenizer)
print(f"Image description:\n{description}")