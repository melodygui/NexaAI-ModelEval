import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "microsoft/bitnet-b1.58-2B-4T"
MAX_CHUNK_TOKENS = 2500
CHUNK_OVERLAP = 200
MAX_GEN_TOKENS = 4096

QUERY_PROMPT = "What are the main themes of Pride and Prejudice, and how are they illustrated through the plot?"

def chunk_tokens(tokens, max_tokens=2000, overlap_tokens=200):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(chunk)
        start += max_tokens - overlap_tokens
    return chunks

def run_book_extraction_bitnet():
    # === Load local text ===
    with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"Loaded Pride and Prejudice ({len(full_text.split())} words)")

    # === Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32
    )
    print(f"Model device map: {model.hf_device_map}")

    # === Tokenize full text ===
    tokens = tokenizer.encode(full_text)
    print(f"Total tokens in book: {len(tokens)}")

    # === Chunk into token chunks ===
    token_chunks = chunk_tokens(tokens, max_tokens=MAX_CHUNK_TOKENS, overlap_tokens=200)
    print(f"Total number of token chunks: {len(token_chunks)}")

    extracted_chunks = []

    for i, chunk_token_ids in enumerate(token_chunks):
        print(f"\n>>> Processing chunk {i+1}/{len(token_chunks)} ...")

        # Decode token chunk to text
        chunk_text = tokenizer.decode(chunk_token_ids)

        # Build plain prompt (BitNet is not chat-based)
        prompt = (
            f"{QUERY_PROMPT}\n\n"
            f"Context:\n\n"
            f"{chunk_text}\n\n"
            f"Extracted Information:"
        )

        # Tokenize prompt
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Input token length: {len(model_inputs.input_ids[0])}")

        # Generate output
        start_time = time.time()

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            temperature=0.8,
            do_sample=True
        )

        end_time = time.time()

        elapsed = end_time - start_time
        print(f">>> Chunk {i+1} took {elapsed:.2f} seconds ({elapsed/60:.2f} min)")

        # Extract only generated tokens (past the input length)
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        extracted_chunks.append(f"=== Chunk {i+1} ===\n{generated_text}")

        print(f"\n--- Extracted Info for chunk {i+1} ---\n{generated_text}\n")

    # Save all outputs
    aggregated_text = "\n\n".join(extracted_chunks)
    with open("bitnet_pride_and_prejudice_query_output.txt", "w", encoding="utf-8") as f:
        f.write(aggregated_text)

    print(f"\nSaved full document extraction to: bitnet_pride_and_prejudice_query_output.txt")

if __name__ == "__main__":
    run_book_extraction_bitnet()
