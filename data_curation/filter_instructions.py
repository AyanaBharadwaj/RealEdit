from tqdm import tqdm
import requests
import argparse
import pandas as pd

api_key = "<replace_with_your_api_key>"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

FIXED_OUTPUT_TOKENS = 70
DEFAULT_TIMEOUT = 5

def build_prompt(instruction):
    """
    Build the prompt that would be used for the API call.
    
    Args:
        instruction (str): The original instruction
        
    Returns:
        str: The complete prompt
    """
    return f"""Simplify the given image editing instruction. Remove URLs, typos, irrelevant details, and expressions of gratitude.
Summarize the main task into a concise image editing request. Some special cases: if you think the request is asking to combine multiple input images,
describe it as 'multi-input'; if the request is unclear or ambiguous, describe it as 'ambiguous'.

Examples of good inputs and outputs:

Input instruction:  Can someone remove the text? I wanna use it as a mobile wallpaper.
Output instruction: Remove the text.

Input instruction:  My friend's mom has a birthday coming up and I was hoping to get her childhood photo restored.
Output instruction: Restore this photo.

Input instruction:  I've been asked for a headshot--can you make this look like one? (please!)
Output instruction: Turn this image into a professional headshot.

Input instruction:  Make me look funny however you want.
Output instruction: Make a humorous edit to the photo.

Input instruction:  Me and the boys - 15 bucks for top 3 ideas!
Output instruction: Add a creative edit to the photo.

Input instruction:  Color ideas for this table?
Output instruction: Recolor the table creatively.

Input instruction:  I need help.
Output instruction: ambiguous

Input instruction:  Combine the people and couches from one image with the jungle in the second image.
Output instruction: multi-input

Input instruction:  Add a baby monkey into the child's arms, similar to the provided reference image.
Output instruction: multi-input

Input instruction:  Merge the people in the five photos into one group photo.
Output instruction: multi-input

Input instruction:  {instruction}
Output instruction:"""

def clean_instruction(raw_instruction: str, model="gpt-4o-2024-11-20", timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Uses the OpenAI API via a POST request to simplify the given image editing instruction.
    In case of error, returns the original instruction.
    
    Args:
        raw_instruction (str): The original image editing instruction.
        timeout (int): Timeout in seconds for the API call.
    
    Returns:
        str: The simplified instruction (or the original instruction on error).
    """
    prompt = build_prompt(raw_instruction)
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": FIXED_OUTPUT_TOKENS
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        cost = calculate_request_cost(response.json(), model)
        return result, cost
    except Exception as e:
        print(f"Error processing instruction: {raw_instruction}\nError: {e}")
        return raw_instruction, 0

def calculate_request_cost(response_json, model="gpt-4o-2024-11-20"):
    """
    Calculate the cost of an API request based on the response JSON.
    
    Args:
        response_json (dict): The JSON response from OpenAI API
        model (str): The model used for the request
        
    Returns:
        float: The cost in USD
    """
    usage = response_json.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    pricing = {
        "gpt-4o-2024-11-20": {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000},
    }
    model_pricing = pricing.get(model, pricing["gpt-4o-2024-11-20"])
    input_cost = prompt_tokens * model_pricing["input"]
    output_cost = completion_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost
    return total_cost
 
def clean_instructions(input_csv: str, output_csv: str, timeout: int = DEFAULT_TIMEOUT) -> None:
    """
    Reads an input CSV file (which must contain an 'instruction' column),
    passes each instruction through the GPT-4 simplification prompt (via a POST request),
    and writes out a new CSV file with the original instructions plus an additional
    'simplified_instruction' column containing the cleaned version.
   
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        timeout (int): Timeout in seconds for each API call.
    """
    try:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(input_csv)
        
        if 'instruction' not in df.columns:
            print("Error: 'instruction' column not found in CSV")
            return
    except Exception as e:
        print(f"Error reading CSV {input_csv}: {e}")
        return
    
    # Process each instruction
    total_cost = 0.0
    simplified_instructions = []
    costs = []
    
    for raw_inst in tqdm(df['instruction'], desc="Simplifying instructions"):
        simplified, cost = clean_instruction(raw_inst, timeout=timeout)
        simplified_instructions.append(simplified)
        costs.append(cost)
        total_cost += cost
    
    # Add new columns to DataFrame
    df['clean_instruction'] = simplified_instructions
    
    try:
        # Write to output CSV
        df.to_csv(output_csv, index=False)
        print(f"CSV with simplified instructions saved to {output_csv}")
        print(f"Total API cost: ${total_cost:.2f}")
    except Exception as e:
        print(f"Error writing CSV {output_csv}: {e}")

def final_filter(input_csv: str, output_csv: str, verbose: bool = True) -> None:
    """
    Reads an input CSV file and filters out entries where the 'instruction' column contains
    one of the specified keywords.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        verbose (bool): Whether to print progress and statistics.
    """
    filter_keywords = ["ambiguous", "multi-input"]
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        
        if verbose:
            print(f"Loaded {len(df)} entries from {input_csv}")
        
        initial_count = len(df)
        
        # Check if 'instruction' column exists
        if 'instruction' not in df.columns:
            if verbose:
                print("Warning: CSV file missing 'instruction' column")
            return
        
        # Count filtered entries for each keyword
        removed_count = {keyword: 0 for keyword in filter_keywords}
        filtered_df = df.copy()
        
        # Filter entries
        for keyword in filter_keywords:
            if keyword:  # Skip empty keyword
                mask = df['clean_instruction'].str.lower().str.contains(keyword.lower(), na=False)
                removed_count[keyword] = mask.sum()
                filtered_df = filtered_df[~mask]
                
        # Write filtered data to CSV
        filtered_df.to_csv(output_csv, index=False)
        
        if verbose:
            print(f"\nFiltering complete:")
            print(f"Initial entries: {initial_count}")
            print(f"Filtered entries: {len(filtered_df)}")
            print(f"Removed entries: {initial_count - len(filtered_df)}")
            for keyword, count in removed_count.items():
                print(f"  - Removed due to '{keyword}': {count}")
            print(f"Output saved to: {output_csv}")
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and filter image editing instructions.")
    parser.add_argument("--input", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, help="Path to the output JSONL file.")
    clean_instructions(parser.input_jsonl, parser.output_jsonl)
    final_filter(parser.input_jsonl, parser.output_jsonl)
    