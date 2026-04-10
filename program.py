import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


import sentencepiece as spm

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=60,      # increased a bit for the richer poem vocabulary
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
    
ids = sp.encode(text, out_type=int) 
data = torch.tensor(ids, dtype=torch.long) 

print(data)

vocab_size = sp.get_piece_size() 
print("Vocab size:", vocab_size)


block_size = 8        # slightly larger context window for the poem
embedding_dim = 64    # richer embeddings for poetic language
n_heads = 4  
n_layers = 3 
lr = 1e-3 
epochs = 3000         # more epochs since poem dataset is small


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("\nTraining the model on the poem...\n")
for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

print("\nTraining complete!\n")


# -----------------------------------------------
# User input: type a word/phrase to start generation
# -----------------------------------------------
sp2 = spm.SentencePieceProcessor()
sp2.load("tokenizer.model")

while True:
    user_input = input("\nEnter a starting word or phrase (or 'quit' to exit): ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    if not user_input:
        print("Please enter something.")
        continue

    encoded = sp2.encode(user_input, out_type=int)
    if not encoded:
        print("Could not encode input. Try another word.")
        continue

    context = torch.tensor([encoded], dtype=torch.long)
    out = model.generate(context, max_new_tokens=25)

    generated_ids = out[0].tolist()
    generated_text = sp2.decode(generated_ids)

    print(f"\nGenerated text:\n{generated_text}")
