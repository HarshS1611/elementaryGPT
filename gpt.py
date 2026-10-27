import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_PATH = "sherlock.txt"

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

# block_size is the context length in characters; n_embd must be divisible by n_head
CONFIG = dict(
    batch_size=32,
    block_size=128,
    n_embd=256,
    n_head=8,
    n_layer=6,
    dropout=0.2,
    lr=3e-4,
    max_iters=10000,
    eval_interval=1000,
    eval_iters=100,
)


# ------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------

def read_text(path=DATA_PATH):
    return open(path, encoding="utf-8").read()


def build_vocab(text):
    # character-level model: every distinct character is one token
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return (chars, stoi, itos)


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join([itos[i] for i in ids])


def split_data(data, train_frac=0.9):
    # contiguous split, not shuffled: validation text is never seen during training
    n = int(train_frac * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(data, batch_size, block_size, generator=None):
    # random start positions; generator=None uses PyTorch's global random state,
    # so every call returns a different batch
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    # targets are the inputs shifted one character ahead: at every position the model predicts the next character
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ------------------------------------------------------------
# 3. Baseline: bigram model
# ------------------------------------------------------------

class BigramLanguageModel(nn.Module):
    # predicts the next character from the current one only, with no context
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        idx = idx.long()
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross_entropy expects (N, C) scores and (N,) targets, so flatten batch and time
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # sample one character at a time and feed it back in as the next input
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ------------------------------------------------------------
# 4. Self-attention
# ------------------------------------------------------------

class Head(nn.Module):
    # one head of causal self-attention: each position forms a query and compares it
    # with the keys of the positions before it to decide which values to read
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        key = nn.Linear(n_embd, head_size, bias=False)
        query = nn.Linear(n_embd, head_size, bias=False)
        value = nn.Linear(n_embd, head_size, bias=False)
        self.key, self.query, self.value = key, query, value
        # the mask is a buffer: saved with the model but not trained
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # scale by 1/sqrt(head_size) so large dot products do not saturate the softmax
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # causal mask: a position may only attend to itself and earlier positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # each output is a weighted average of the value vectors
        out = wei @ self.value(x)
        return out


class MultiHeadAttention(nn.Module):
    # several heads in parallel can pick up different patterns; their outputs are
    # concatenated and mixed by a linear projection
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, block_size, dropout) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


# ------------------------------------------------------------
# 5. Transformer block
# ------------------------------------------------------------

class FeedForward(nn.Module):
    # applied to each position independently; the 4x hidden width is the standard transformer choice
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # attention lets positions exchange information, the feed-forward layer then
    # processes each position on its own
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # pre-norm residual connections: normalize, apply the sublayer, add the input back.
        # the residual path lets gradients reach early layers in a deep stack
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ------------------------------------------------------------
# 6. The GPT model
# ------------------------------------------------------------

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # the token embedding says what each character is, the position embedding says
        # where it is: attention alone has no sense of order
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # small initial weights make the untrained model start near a uniform guess
        # over the vocabulary (loss close to ln(vocab_size)) instead of a confident wrong one
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        # one score per vocabulary character at every position
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop context to block_size
            logits, _ = self(idx_cond)
            # only the prediction at the last position matters for the next character
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ------------------------------------------------------------
# 7. Training
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters):
    # average over several batches for a steadier number than a single batch gives;
    # eval() switches dropout off while measuring
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, cfg, log=print):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    history = []
    for step in range(cfg['max_iters']):
        # a growing gap between train and val loss means the model is memorizing the text
        if step % cfg['eval_interval'] == 0 or step == cfg['max_iters'] - 1:
            losses = estimate_loss(model, train_data, val_data, cfg['batch_size'], cfg['block_size'], cfg['eval_iters'])
            log(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            history.append((step, losses['train'], losses['val']))

        xb, yb = get_batch(train_data, cfg['batch_size'], cfg['block_size'])
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history


# ------------------------------------------------------------
# 8. Running it
# ------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return "mps"
    return "cpu"


if __name__ == "__main__":
    torch.manual_seed(1337)
    device = pick_device()

    text = read_text()
    chars, stoi, itos = build_vocab(text)
    # the data lives on the device, so batches need no copying
    data = torch.tensor(encode(text, stoi), dtype=torch.long, device=device)
    train_data, val_data = split_data(data)

    c = CONFIG
    model = GPTLanguageModel(len(chars), c["block_size"], c["n_embd"],
                             c["n_head"], c["n_layer"], c["dropout"]).to(device)

    print("=" * 60)
    print("elementaryGPT")
    print("=" * 60)
    print(f"device       : {device}")
    print(f"dataset      : {len(text):,} characters, vocab {len(chars)}")
    print(f"split        : {len(train_data):,} train / {len(val_data):,} val tokens")
    print(f"model        : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"({c['n_layer']} layers, {c['n_head']} heads, n_embd {c['n_embd']}, block_size {c['block_size']})")
    print(f"training     : {c['max_iters']} steps, batch {c['batch_size']}, lr {c['lr']}")
    print("-" * 60)

    history = train(model, train_data, val_data, c)

    print("-" * 60)
    print(f"final        : train {history[-1][1]:.4f} | val {history[-1][2]:.4f}")
    # only the weights are saved; generate.py rebuilds the model from CONFIG and loads them
    torch.save(model.state_dict(), "elementarygpt.pt")
    print("checkpoint   : elementarygpt.pt")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=2000)[0].tolist(), itos)
    with open("samples.txt", "w") as f:
        f.write(sample)
    print("samples      : samples.txt (2000 characters)")
    print("=" * 60)
    print(sample[:600])
