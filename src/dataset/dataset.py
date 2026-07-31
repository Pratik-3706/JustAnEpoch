import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm



def download_and_save_dataset(dataset_name="teknium/OpenHermes-2.5", save_dir=None, subset_size=None):
    """
    Downloads the dataset from HuggingFace and saves it locally to disk.
    For a 4GB GPU, testing with a smaller subset first is highly recommended.
    """
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "reasoning_dataset"
    else:
        save_dir = Path(save_dir)

    print(f"Loading dataset '{dataset_name}' from HuggingFace...")
    
    # Load the dataset (The dataset uses the 'train' split)
    dataset = load_dataset(dataset_name, split="train")
    
    # If the user wants a smaller chunk to test on their 4GB GPU
    if subset_size is not None and subset_size < len(dataset):
        print(f"Selecting a subset of {subset_size} rows for faster testing...")
        dataset = dataset.select(range(subset_size))
        
    print(f"Dataset loaded with {len(dataset)} rows.")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the dataset to local disk as a pure raw text file!
    save_path = os.path.join(save_dir, "dataset_train.txt")
    print(f"Saving dataset to {save_path} as raw text...")
    
    with open(save_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Writing to file"):
            # Flatten the JSON conversation into a single raw text block
            text_block = ""
            for turn in item["conversations"]:
                role = turn["from"].upper()
                value = turn["value"]
                text_block += f"<{role}>\n{value}\n"
            
            f.write(text_block + "\n<|endoftext|>\n\n")
    
    print("Download and save complete!")
    print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    
    return dataset

if __name__ == "__main__":

    download_and_save_dataset(subset_size=None)
