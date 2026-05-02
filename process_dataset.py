import chess
import numpy as np
import random
from tqdm import tqdm

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_arrays(board: chess.Board):
    pieces, squares, colors = [], [], []
    for square, piece in board.piece_map().items():
        pieces.append(PIECE_TO_INDEX[piece.piece_type])
        squares.append(square)
        colors.append(0 if piece.color == chess.WHITE else 1)
    return (
        np.asarray(pieces, dtype=np.int8),
        np.asarray(squares, dtype=np.int8),
        np.asarray(colors, dtype=np.int8),
    )

def result_to_value(result_str: str) -> float:
    if "1-0" in result_str:
        return 1.0
    if "0-1" in result_str:
        return 0.0
    return 0.5

def pack_samples(samples):
    if not samples:
        return None
    pieces_all, squares_all, colors_all, results = [], [], [], []
    offsets = [0]
    
    for pieces, squares, colors, result in samples:
        pieces_all.append(pieces)
        squares_all.append(squares)
        colors_all.append(colors)
        offsets.append(offsets[-1] + len(pieces))
        results.append(result)
        
    return {
        "pieces": np.concatenate(pieces_all).astype(np.int8),
        "squares": np.concatenate(squares_all).astype(np.int8),
        "colors": np.concatenate(colors_all).astype(np.int8),
        "offsets": np.asarray(offsets, dtype=np.uint32),
        "results": np.asarray(results, dtype=np.float32),
    }

def process_epd_to_npz(input_path, train_path, val_path, split_ratio=0.8):
    samples = []
    
    # 1. Count lines for the progress bar
    print("Counting lines...")
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    # 2. Parse EPD file
    with open(input_path, mode="r") as file:
        print(f"Processing {total_lines} positions...")
        for line in tqdm(file, total=total_lines, desc="Parsing EPD"):
            line = line.strip()
            if not line:
                continue
                
            board = chess.Board()
            try:
                # EPD usually uses 'ce' or 'c9' opcodes. 
                # python-chess set_epd splits the line at the first opcode.
                ops = board.set_epd(line)
                
                # Check for 'c9' specifically as in your original snippet
                raw_result = ops.get("c9", "1/2-1/2") 
                outcome = result_to_value(str(raw_result))
                
                pieces, squares, colors = board_to_arrays(board)
                samples.append((pieces, squares, colors, outcome))
                
            except (ValueError, KeyError):
                continue

    # 3. Shuffle and Split
    print("Shuffling and splitting dataset...")
    random.shuffle(samples)
    
    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # 4. Pack and Save
    for data, path, label in [(train_samples, train_path, "Train"), 
                              (val_samples, val_path, "Validation")]:
        packed = pack_samples(data)
        if packed:
            np.savez_compressed(path, **packed)
            print(f"Successfully saved {label} set ({len(data)} positions) to {path}")

if __name__ == "__main__":
    process_epd_to_npz(
        input_path="./dataset.epd", 
        train_path="./train.npz", 
        val_path="./val.npz",
        split_ratio=0.8
    )
