# TinyGPT — Mini Language Model from Scratch

A lightweight GPT-style language model built entirely from scratch using PyTorch. The model learns from a small text corpus (a poem) and generates new text by predicting the next token given a user-provided starting word or phrase. It uses a real Transformer architecture with multi-head self-attention, positional embeddings, and BPE tokenization — the same core ideas behind large language models like GPT.

---

## Project Structure

```
project/
│
├── program.py              # Main training and generation script
├── transformer_block.py    # Transformer architecture (attention, FFN, blocks)
├── corpus.txt              # Training dataset (poem used as text corpus)
├── tokenizer.model         # SentencePiece trained tokenizer model (auto-generated)
├── tokenizer.vocab         # Tokenizer vocabulary file (auto-generated)
└── README.md               # This file
```

---

## How It Works

1. **Tokenization** — The corpus is tokenized using SentencePiece BPE (Byte Pair Encoding), which breaks text into subword units and builds a vocabulary of a fixed size.
2. **Training Data Preparation** — The tokenized sequence is split into overlapping windows of `block_size` tokens. Each window forms an (input, target) pair where the target is the input shifted one position to the right.
3. **Model Training** — The TinyGPT model is trained using cross-entropy loss and the AdamW optimizer to predict the next token at each position.
4. **Text Generation** — The user provides a starting word or phrase. The model encodes it, then autoregressively samples the next token using softmax probabilities, continuing until the desired length is reached.

---

## File Details

### `program.py`
The main entry point of the project. It handles the full pipeline:

- **SentencePiece training** — Trains a BPE tokenizer on `corpus.txt` with a vocabulary size of 60 and saves `tokenizer.model` and `tokenizer.vocab`.
- **Data preparation** — Reads and encodes the corpus into integer token IDs, then converts them into a PyTorch tensor.
- **Hyperparameters** — Defines all model and training settings:
  - `block_size = 8` — context window length (number of tokens the model sees at once)
  - `embedding_dim = 64` — size of token and position embedding vectors
  - `n_heads = 4` — number of attention heads in multi-head attention
  - `n_layers = 3` — number of stacked Transformer blocks
  - `lr = 1e-3` — learning rate for AdamW optimizer
  - `epochs = 3000` — number of training steps
- **Batch generation (`get_batch`)** — Randomly samples starting positions from the data and builds batches of input-target pairs.
- **TinyGPT model class** — Defines the full model with token embeddings, positional embeddings, stacked Transformer blocks, layer norm, and a linear output head.
- **Training loop** — Runs for the specified number of epochs, printing loss every 500 steps.
- **Interactive generation loop** — After training, prompts the user to enter a starting word or phrase in a loop. Encodes the input, runs the model's `generate()` method, decodes the output, and prints the generated text. Type `quit` to exit.

### `transformer_block.py`
Contains all the neural network building blocks that make up the Transformer architecture:

- **`SelfAttentionHead`** — A single causal (masked) self-attention head. Computes Key, Query, and Value projections from the input. Uses a lower-triangular mask to prevent the model from attending to future tokens. Scales attention scores by `1/sqrt(head_size)`.
- **`MultiHeadAttention`** — Runs multiple `SelfAttentionHead` instances in parallel, concatenates their outputs, and projects back to `embedding_dim` using a linear layer.
- **`FeedForward`** — A two-layer MLP applied position-wise after attention: `Linear → ReLU → Linear`, with a 4x hidden dimension expansion.
- **`Block`** — A full Transformer block combining multi-head attention and feed-forward layers, each wrapped with a residual connection and Pre-LayerNorm (`x = x + sublayer(LayerNorm(x))`).

### `corpus.txt`
The training dataset. Contains the poem *"The Unseen Architecture"* — 16 lines of poetic text used as the model's learning material. The model learns the statistical patterns and word sequences present in this text.

### `tokenizer.model` / `tokenizer.vocab`
Auto-generated files produced by SentencePiece during the first run of `program.py`. The `.model` file is the trained BPE tokenizer used to encode and decode text. The `.vocab` file lists all vocabulary tokens and their log-probabilities.

---

## Technologies & Libraries Used

| Library / Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.x | Core programming language |
| **PyTorch** (`torch`) | Latest | Tensor operations, neural network layers, autograd, training loop |
| **torch.nn** | (part of PyTorch) | `nn.Module`, `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, `nn.Sequential` |
| **torch.nn.functional** | (part of PyTorch) | `F.softmax`, `F.cross_entropy` for loss and sampling |
| **SentencePiece** (`sentencepiece`) | Latest | BPE tokenizer training and encoding/decoding |

---

## Model Architecture Summary

```
Input tokens (B, T)
       │
       ├─► Token Embedding      (vocab_size → embedding_dim)
       ├─► Position Embedding   (block_size → embedding_dim)
       │
       └─► Sum of both embeddings
               │
        ┌──────┴──────┐
        │  Block × 3  │  (n_layers stacked Transformer blocks)
        │             │
        │  LayerNorm  │
        │      +      │
        │  Multi-Head │  (n_heads = 4 parallel attention heads)
        │  Attention  │
        │      +      │
        │  LayerNorm  │
        │      +      │
        │  FeedForward│  (Linear → ReLU → Linear, 4x expansion)
        └──────┬──────┘
               │
          LayerNorm (final)
               │
         Linear Head  (embedding_dim → vocab_size)
               │
            Logits  →  Softmax  →  Sample next token
```

---

## Setup & Installation

### Prerequisites
Make sure you have Python 3.x installed. Then install the required libraries:

```bash
pip install torch sentencepiece
```

### Running the Project

1. Place `corpus.txt`, `program.py`, and `transformer_block.py` in the same folder.
2. Run the main script:

```bash
python program.py
```

3. The script will:
   - Train the BPE tokenizer on `corpus.txt`
   - Train the TinyGPT model for 3000 steps (loss printed every 500 steps)
   - Prompt you to enter a starting word

4. Example interaction after training:

```
Enter a starting word or phrase (or 'quit' to exit): the

Generated text:
the air is doing the hard work holding up the birds carrying the scent
```

---

## Customization Tips

| Setting | Where | What to change |
|---|---|---|
| Training data | `corpus.txt` | Replace with any `.txt` file |
| Vocabulary size | `program.py` → `vocab_size=60` | Increase for larger corpora |
| Context length | `program.py` → `block_size=8` | Larger = more context, slower |
| Model size | `program.py` → `embedding_dim`, `n_heads`, `n_layers` | Bigger = smarter, needs more data |
| Training duration | `program.py` → `epochs=3000` | More epochs = lower loss |
| Output length | `program.py` → `max_new_tokens=25` | Change in the generate call |

---

## Notes

- Since the corpus is a single short poem (~16 lines), the model will primarily recombine and extend phrases it has seen during training. This is expected behavior for a mini-LLM on a tiny dataset.
- For richer generation, try replacing `corpus.txt` with a larger text (a full book, song collection, or article set) and increasing `vocab_size` and `epochs` accordingly.
- GPU acceleration is supported automatically — if CUDA is available on your machine, PyTorch will use it.
