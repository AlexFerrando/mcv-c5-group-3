import os

import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch
import time
from tqdm.auto import tqdm  # For progress bar
from huggingface_hub import login

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-3B"
CSV_INPUT_PATH = "/ghome/c5mcv03/mcv-c5-group-3/week3/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
CSV_OUTPUT_PATH = "/ghome/c5mcv03/mcv-c5-group-3/week5/recipes_classified.csv"
TITLE_COLUMN = "Title"
INGREDIENTS_COLUMN = "Cleaned_Ingredients"
BATCH_SIZE = 16  # Process rows in batches for potentially better performance (adjust based on GPU memory)

# Retrieve the token from the environment variable
token = os.getenv("HF_TOKEN")

# Log in to Hugging Face
login(token=token)

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a recipe classifier. Your task is to determine if a recipe, based on its title and ingredients, is a 'drink' or 'other'. Respond with only the single word label 'drink' or 'other'.

--- Example 1 ---
Recipe Title: Classic Mojito
Ingredients: white rum, sugar, lime juice, soda water, mint
Classification: drink

--- Example 2 ---
Recipe Title: Simple Baked Chicken Breast
Ingredients: chicken breast, olive oil, salt, black pepper, paprika, garlic powder
Classification: other

--- Recipe to Classify ---
Recipe Title: {title}
Ingredients: {ingredients}
Classification:"""

def format_prompt(title, ingredients):
    """Formats the prompt with the given title and ingredients."""
    # Basic cleaning: replace potential newlines in ingredients that might break formatting
    ingredients_str = str(ingredients).replace('\n', ', ')
    title_str = str(title)
    return PROMPT_TEMPLATE.format(title=title_str, ingredients=ingredients_str)

def clean_output(generated_text, prompt):
    """Extracts the classification ('drink' or 'other') from the model's full output."""
    # The pipeline output includes the prompt, so we remove it
    output_only = generated_text[len(prompt):].strip()
    # Take the first word, convert to lowercase, and remove potential punctuation
    first_word = output_only.split()[0].lower().strip(".,!?;:")
    if first_word in ["drink", "other"]:
        return first_word
    else:
        # Fallback or indicate uncertainty if the output is unexpected
        print(f"Warning: Unexpected model output: '{output_only}'. Returning 'other'.")
        return "other" # Defaulting to 'other' on unexpected output


print(f"Loading model: {MODEL_NAME}")
# --- Load Model and Tokenizer using pipeline ---
# Using bfloat16 for potentially faster inference and lower memory if supported
# Use device_map="auto" to leverage GPU(s) if available
try:
    # Llama models might require explicit trust_remote_code=True
    classifier_pipeline = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME, # Usually same as model, pipeline can handle it
        torch_dtype=torch.float16, # Use float16 or remove if causing issues
        device_map="auto", # Automatically use GPU if available, else CPU
        trust_remote_code=True # Often needed for Llama models
    )
except Exception as e:
    print(f"Error loading model pipeline: {e}")
    print("Ensure you have accepted the model's license on Hugging Face Hub and are logged in (huggingface-cli login).")
    exit()

# Set pad_token_id if not already set (important for batching)
if classifier_pipeline.tokenizer.pad_token is None:
    classifier_pipeline.tokenizer.pad_token = classifier_pipeline.tokenizer.eos_token
    classifier_pipeline.model.config.pad_token_id = classifier_pipeline.model.config.eos_token_id


print("Loading CSV data...")
try:
    df = pd.read_csv(CSV_INPUT_PATH)
    if TITLE_COLUMN not in df.columns or INGREDIENTS_COLUMN not in df.columns:
        raise ValueError(f"CSV must contain columns named '{TITLE_COLUMN}' and '{INGREDIENTS_COLUMN}'")
except FileNotFoundError:
    print(f"Error: Input CSV file not found at '{CSV_INPUT_PATH}'")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()


print(f"Starting classification for {len(df)} rows...")

results = []
# Process in batches
for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    prompts = [
        format_prompt(row[TITLE_COLUMN], row[INGREDIENTS_COLUMN])
        for _, row in batch_df.iterrows()
    ]

    # Generate classifications for the batch
    # max_new_tokens is important to limit output length
    # We only expect 'drink' or 'other', so 5-10 tokens is plenty.
    # do_sample=False ensures deterministic output (most likely token)
    start_time = time.time()
    try:
        batch_outputs = classifier_pipeline(
            prompts,
            max_new_tokens=10,
            num_return_sequences=1,
            eos_token_id=classifier_pipeline.tokenizer.eos_token_id,
            pad_token_id=classifier_pipeline.tokenizer.pad_token_id,
            do_sample=False # Get the most likely prediction
        )
    except Exception as e:
        print(f"\nError during pipeline generation (batch starting row {i}): {e}")
        # Add None or error markers for rows in the failed batch
        results.extend([f"ERROR: {e}"] * len(batch_df))
        continue # Skip to the next batch

    end_time = time.time()
    batch_time = end_time - start_time
    print(f"Processed batch {i//BATCH_SIZE + 1}/{(len(df) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_df)} rows) in {batch_time:.2f}s")

    # Clean and store results for the batch
    for idx, output in enumerate(batch_outputs):
        original_prompt = prompts[idx]
        # The pipeline might return a list of dicts, get the generated text
        generated_text = output[0]['generated_text']
        print("generated_text:", generated_text)
        try:
            classification = clean_output(generated_text, original_prompt)
        except Exception as e:
            print(f"Error cleaning output for prompt '{original_prompt}': {e}")
            print("Generated text:", generated_text)
            classification = "ERROR: Cleaning failed"
        results.append(classification)

# Add results to the DataFrame
df['Classification'] = results[:len(df)] # Ensure results list matches df length

print("\nClassification complete.")
print(df['Classification'].value_counts())

# Save the results
try:
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Results saved to '{CSV_OUTPUT_PATH}'")
except Exception as e:
    print(f"Error saving results to CSV: {e}")

print("Done.")