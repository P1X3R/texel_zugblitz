import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import math

# Configuration
MAX_LEN = 32
PAD_TOKEN = 6  # Typically used to represent "No Piece" or Padding
CHUNK_SIZE = 4096
# The folder where train.npz and val.npz are located
IN_DIR = "." 
# The folder where the padded .pt chunks will be saved
OUT_DIR = "dataset_pad"

def pad_block(arr, offsets, max_len, pad_val):
    n = len(offsets) - 1
    out = np.full((n, max_len), pad_val, dtype=arr.dtype)
    for i in range(n):
        start, end = offsets[i], offsets[i + 1]
        length = min(end - start, max_len)
        out[i, :length] = arr[start : start + length]
    return out

class ChunkWriter:
    def __init__(self, output_dir, process_id, chunk_size=CHUNK_SIZE):
        self.output_dir = output_dir
        self.process_id = process_id
        self.chunk_size = chunk_size
        self.chunk_id = 0
        self.index = []

        self.pieces_buf = []
        self.squares_buf = []
        self.colors_buf = []
        self.results_buf = []

    def add(self, pieces, squares, colors, result):
        self.pieces_buf.append(pieces)
        self.squares_buf.append(squares)
        self.colors_buf.append(colors)
        self.results_buf.append(result)
        if len(self.pieces_buf) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.pieces_buf:
            return

        chunk_name = f"proc{self.process_id}_chunk_{self.chunk_id:05d}.pt"
        chunk_path = os.path.join(self.output_dir, chunk_name)

        torch.save(
            {
                "pieces": torch.stack(self.pieces_buf),
                "squares": torch.stack(self.squares_buf),
                "colors": torch.stack(self.colors_buf),
                "result": torch.tensor(self.results_buf, dtype=torch.float32),
            },
            chunk_path,
        )

        for row in range(len(self.results_buf)):
            # Store relative path for index portability
            self.index.append((chunk_name, row))

        self.pieces_buf.clear()
        self.squares_buf.clear()
        self.colors_buf.clear()
        self.results_buf.clear()
        self.chunk_id += 1

def process_single_file(filepath, split_out, process_id):
    """Processes one large .npz file and breaks it into padded .pt chunks."""
    writer = ChunkWriter(split_out, process_id)

    data = np.load(filepath)
    offsets = data["offsets"]
    results = data["results"]

    # Apply padding
    pieces_padded = pad_block(data["pieces"], offsets, MAX_LEN, PAD_TOKEN)
    squares_padded = pad_block(data["squares"], offsets, MAX_LEN, PAD_TOKEN)
    colors_padded = pad_block(data["colors"], offsets, MAX_LEN, 2) # Color padding often 2 (none)

    pieces_tensor = torch.from_numpy(pieces_padded).to(torch.uint8)
    squares_tensor = torch.from_numpy(squares_padded).to(torch.uint8)
    colors_tensor = torch.from_numpy(colors_padded).to(torch.uint8)

    for i in range(len(results)):
        writer.add(
            pieces_tensor[i], squares_tensor[i], colors_tensor[i], float(results[i])
        )

    writer.flush()
    return writer.index

def convert_dataset_parallel(split_name, num_workers=4):
    """Handles the conversion of train.npz or val.npz into padded chunks."""
    input_file = os.path.join(IN_DIR, f"{split_name}.npz")
    split_out = os.path.join(OUT_DIR, split_name)
    os.makedirs(split_out, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found.")
        return

    print(f"Loading {input_file} for padding...")
    # For a single large file, we divide the indices among workers 
    # but for simplicity and to avoid memory overhead, we process the file 
    # sequentially or pass it to one worker. 
    # If the file is massive (>10GB), we process in one block here:
    
    indices = process_single_file(input_file, split_out, 0)

    # Save the consolidated index
    index_path = os.path.join(split_out, "index.pt")
    torch.save(indices, index_path)
    print(f"{split_name} complete: {len(indices):,} positions saved to {split_out}")

def main():
    # Process both splits
    for split in ["train", "val"]:
        convert_dataset_parallel(split)

if __name__ == "__main__":
    main()
