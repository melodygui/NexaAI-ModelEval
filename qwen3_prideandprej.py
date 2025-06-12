import re
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# === settings ===
MODEL_NAME = "Qwen/Qwen3-0.6B" 
MAX_GEN_TOKENS = 4096
CONTEXT_WINDOW = 32768
CHUNK_SIZE = 30000 # leave ~2k tokens for prompt + safety 
OVERLAP = 500 

# === load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto", 
    torch_dtype=torch.float16
)

# === load text ===
with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"\n\s*\n", "\n\n", text) # optional: collapse consecutive newlines into 2 newlines

# === tokenize full text ===
tokens = tokenizer.encode(text)
print(f"Total tokens in book: {len(tokens)}")

# === chunking ===
chunks = []
for i in range(0, len(tokens), CHUNK_SIZE - OVERLAP):
    chunk_tokens = tokens[i:i + CHUNK_SIZE]
    chunk_text = tokenizer.decode(chunk_tokens)
    chunks.append(chunk_text)

print(f"Number of chunks: {len(chunks)}")

# === process each chunk ===
for i, chunk_text in enumerate(chunks):
    print(f"\n=== Processing chunk {i+1}/{len(chunks)} ===")
    messages = [
        {
            "role": "system",
            "content": "Your task is to retain useful information and details that can help to answer this question. Don't use your base knowledge, summarize solely based on the provided content. No need to format the response, the extraction of information is the top priority."
        },
        {
            "role": "user",
            "content": (
                "What are the main themes of Pride and Prejudice, and how are they illustrated through the plot?\n\n"
                "Context:\n\n"
                f"{chunk_text}"
            )
        }
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate 
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_GEN_TOKENS,
        temperature=0.7,
        do_sample=True
    )
    end_time = time.time()

    print(f"Generation for chunk {i+1} took {end_time - start_time:.2f} seconds.")

    # Only decode generated new tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print(f"\n--- Output for chunk {i+1} ---")
    print(output_text)
    
    with open(f"qwen3_chunk_{i+1:02d}.txt", "w", encoding="utf-8") as out_f:
        out_f.write(output_text)
    
    print(f"\nSaved output for chunk {i+1}")
