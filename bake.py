import torch
import os

# Configuration matching your heatmap script
CHECKPOINT_PATH = "checkpoints/epoch_070.pt"
PIECE_NAMES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]


def bake_to_c():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return

    # Load data exactly like your heatmap script
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    # Convert tensors to integers (centipawns)
    mg_tables = state_dict["psqt_mg"][:6].round().detach().numpy().astype(int)
    eg_tables = state_dict["psqt_eg"][:6].round().detach().numpy().astype(int)

    with open("psqt_weights.h", "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("typedef struct { int16_t mg; int16_t eg; } score_t;\n\n")
        f.write("const score_t PSQTS[6][64] = {\n")

        for i, name in enumerate(PIECE_NAMES):
            f.write(f"  // --- {name} ---\n  {{\n")

            # Reshape to 8x8 for rank-by-rank formatting
            mg_grid = mg_tables[i].reshape(8, 8)
            eg_grid = eg_tables[i].reshape(8, 8)

            for rank in range(8):
                f.write("    ")
                for file in range(8):
                    mg_val = mg_grid[rank][file]
                    eg_val = eg_grid[rank][file]
                    f.write(f"{{{mg_val:3}, {eg_val:4}}}, ")
                f.write(f" // Rank {rank + 1}\n")

            f.write("  },\n")

        f.write("};\n\n")

    print("Successfully baked weights into psqt_weights.h")


if __name__ == "__main__":
    bake_to_c()
