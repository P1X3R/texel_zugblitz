# PSQT Texel Tuning Pipeline

A full training pipeline for a **tapered PSQT (Piece-Square Table) evaluation function** for a chess engine using a Texel-style approach with PyTorch.

What started as a “quick 400 LOC script” evolved into a complete system for:

* dataset generation from EPD
* efficient packed representations
* padded tensor datasets
* constrained PSQT learning
* checkpointed training with validation and early stopping

---

## Overview

This project implements a **modernized Texel tuning pipeline**:

1. Parse EPD positions with game outcomes
2. Convert positions into compact array representations
3. Pack and pad data for efficient batching
4. Train a **tapered midgame/endgame PSQT model**
5. Learn at the same time:

   * piece-square values
   * sigmoid scaling factor (K) for WDL prediction

---

## Key Ideas

* **Tapered Evaluation**
  Separate midgame (MG) and endgame (EG) PSQTs blended by phase.

* **Learned Logistic Scaling**
  A trainable parameter `K` converts centipawn evaluations into win/draw/loss probabilities.

* **Material Anchoring**
  PSQTs are regularized to stay consistent with base material values (avoid values to explode in one direction, e.g finding out a pawn at e4 is worth 7000 cp).

* **Smooth Constraints**
  A soft penalty prevents PSQT values from drifting too far from reasonable ranges (and look better on the heatmap).

* **Use base `e` for sigmoid function**  
  Inspired by Ethereal's tuning method, this training uses a sigmoid with base `e`, letting the constant `K` absorb the `-1/400` term from the original formula.

  Peter Österlund's formula:

  ```
  sigmoid = 1 / (1 + 10^(-K * E / 400))
  ```

  Modified (Ethereal-style):

  ```
  sigmoid = 1 / (1 + e^(-K * E))
  ```

> [!NOTE]
> Here E is an evaluation, and K is a coefficient computed to minimize an error function. 

---

## Pipeline

### 1. Dataset Creation (EPD -> NPZ)

```bash
python process_dataset.py
```

* Parses EPD positions (from Zurichess dataset)
* Extracts:

  * pieces
  * squares
  * colors
  * results (W/D/L -> float)
* Outputs:

  * `train.npz`
  * `val.npz`

---

### 2. Padding & Chunking (NPZ -> PT)

```bash
python pad.py
```

* Converts variable-length positions into fixed-size tensors
* Applies padding for batching
* Splits into chunked `.pt` files
* Produces:

  * `dataset_pad/train/*.pt`
  * `dataset_pad/val/*.pt`

---

### 3. Training

```bash
python main.py
```

* Loads full dataset into memory
* Trains PSQT model using:

  * MSE loss vs WDL targets
  * material consistency regularization
  * PSQT deviation penalty
* Uses:

  * Adagrad optimizer
  * checkpointing
  * early stopping

---

### 4. Baking

```bash
python bake.py
```
* Writes the model parameters into a C array (6 pieces * 64 squares) where `0 = A1` and `63 = H8`.
* Uses the custom `score_t` structure to encode MG and EG score.
* Set the checkpoint to be baked by updating `CHECKPOINT_PATH`.

---

## Model

The evaluation function:

* Computes MG and EG scores from PSQTs
* Blends them using a **phase function** (calculates the current stage of the game)
* Converts evaluation -> whites winning probability via:

```
P(win) = sigmoid(K * E)
```

Where:

* `K` is learned during training
* `E` is the model's evaluation in Centipawns (CP)

---

## Architecture

### Inputs per position

* `pieces`: piece types
* `squares`: board indices
* `colors`: white/black

### Core steps

1. Index PSQT tables (in centipawns)
2. Apply color symmetry (flip for black)
3. Compute MG / EG scores
4. Compute phase
5. Blend -> final evaluation
6. Apply sigmoid

---

## Constraints & Regularization

To stabilize training:

### Material Consistency Loss

Keeps average PSQT values aligned with base material.

### PSQT Deviation Loss

Soft constraint preventing extreme square values using a smooth penalty.

---

## Training Features

* Batch training (PyTorch)
* Checkpointing per epoch
* Resume support
* Logging
* Early stopping

---

## Why this exists

Traditional Texel tuning (Peter Österlund's original idea) is often:

* Minimalist
* Slow due to local search limitations
* Non-viable for a large number of parameters

This project uses **modern tools** like PyTorch and Python to **boost the performance of the Texel tuning method.** This potentially allows for:

* A higher parameter count.
* Better utilization of modern hardware, such as GPUs.
* Drastically faster convergence to a local optimum.
* The use of larger datasets due to the increase in performance.

> [!NOTE]
> This project is currently used only for PSQTs (Piece-Square Tables) and utilizes the Zurichess dataset (a classic benchmark dataset) due to computational constraints.

---

## Project Structure

```
.
├── process_dataset.py # EPD -> NPZ dataset
├── pad.py             # NPZ -> padded PT chunks
├── main.py            # Training loop
├── bake.py            # Model -> C Array
├── heatmap.py         # Model visualizer tool
├── dataset_pad/       # Processed dataset
├── checkpoints/       # Saved models (PSQTs)
```

---

## Notes

* Entire dataset is loaded into RAM during training
* Just an experiment, not meant for production deployment even though Zugblitz uses it.
* Assumes reasonably clean EPD input (look at Zurichess's format)
* Dataset padding could be done directly on EPD processing

---

## Origin Story 

This was supposed to be a simple, vibe-coded Texel tuning script.

It wasn’t.

The code written by the AI was painfully slow and, most of the time, didn't even work. Therefore, I had to get my hands dirty. With my limited knowledge of Python and even less of PyTorch and NumPy, I somehow made it work. If you noticed that the dataset padding could have been done directly on the EPD, this design choice was taken precisely due to the incorrect use of AI to "speed up the development process." However, it works effectively and remains stable.

Tuning speed is surprising, converging on a local minimum within roughly 2 hours on an Intel Pentium Silver N5030, compared to the 6 hours it takes using CPW's tuning method on a 16-core Dell T620. This was an expected effect, but not at such a significant scale.

This side project is more an experiment in machine learning and the correct use of AI to actually accelerate development. I’ve learned that it is not a replacement for engineering and critical thinking, but rather a reasoning copilot.
