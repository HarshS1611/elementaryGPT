import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from gpt import (
    build_vocab, encode, decode, split_data, get_batch,
    BigramLanguageModel, Head, MultiHeadAttention, FeedForward, Block,
    GPTLanguageModel, estimate_loss, train,
)

TEXT = ("It is a capital mistake to theorize before one has data. "
        "Elementary, my dear Watson. ") * 40

TINY = dict(block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def _data():
    chars, stoi, itos = build_vocab(TEXT)
    data = torch.tensor(encode(TEXT, stoi), dtype=torch.long)
    return chars, stoi, itos, data


# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------

def test_vocab_sorted_unique():
    chars, stoi, itos = build_vocab("cabba")
    assert chars == ["a", "b", "c"]
    assert stoi == {"a": 0, "b": 1, "c": 2}
    assert itos == {0: "a", 1: "b", 2: "c"}


def test_encode_decode_roundtrip():
    chars, stoi, itos = build_vocab(TEXT)
    s = "Elementary, my dear Watson."
    assert decode(encode(s, stoi), itos) == s


def test_split_data_no_shuffle():
    data = torch.arange(100)
    tr, va = split_data(data)
    assert len(tr) == 90 and len(va) == 10
    assert torch.equal(tr, torch.arange(90)), "train must be the first 90%, in order"
    assert torch.equal(va, torch.arange(90, 100))


def test_get_batch_returns_a_different_batch_each_call():
    # a generator seeded inside get_batch returned the same batch on every step
    data = torch.arange(1000)
    a, _ = get_batch(data, 8, 16)
    b, _ = get_batch(data, 8, 16)
    assert not torch.equal(a, b)


def test_get_batch_shapes_and_shift():
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size=4, block_size=8, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1), "y must be x shifted left by one position"


# ------------------------------------------------------------
# 2. Bigram baseline
# ------------------------------------------------------------

def test_bigram_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    chars, stoi, itos, data = _data()
    V = len(chars)
    m = BigramLanguageModel(V)
    x, y = get_batch(data, 4, 8)
    logits, loss = m(x, y)
    assert logits.shape == (4, 8, V)
    assert loss.dim() == 0
    # untrained: should be in the ballpark of -log(1/V), not wildly off
    assert loss.item() < 3 * torch.log(torch.tensor(float(V))).item()


def test_bigram_forward_without_targets():
    m = BigramLanguageModel(10)
    logits, loss = m(torch.zeros((2, 5), dtype=torch.long))
    assert loss is None and logits.shape == (2, 5, 10)


def test_bigram_generate_length():
    m = BigramLanguageModel(10)
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)


# ------------------------------------------------------------
# 3. Attention
# ------------------------------------------------------------

def test_head_output_shape():
    h = Head(n_embd=16, head_size=4, block_size=8, dropout=0.0)
    assert h(torch.randn(2, 8, 16)).shape == (2, 8, 4)


def test_head_mask_registered_as_buffer_not_parameter():
    h = Head(16, 4, 8, 0.0)
    names = [n for n, _ in h.named_parameters()]
    assert not any("tril" in n for n in names), "tril must be a buffer, not a learned parameter"
    assert "tril" in dict(h.named_buffers())


def test_head_is_causal():
    torch.manual_seed(0)
    h = Head(16, 4, 8, 0.0)
    h.eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[0, 5:] = torch.randn(3, 16)  # change positions 5, 6, 7 only
    out1, out2 = h(x), h(x2)
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-6), (
        "positions 0-4 changed when only positions 5-7 changed: "
        "information is leaking from the future"
    )


def test_head_handles_shorter_sequence_than_block_size():
    h = Head(16, 4, block_size=8, dropout=0.0)
    assert h(torch.randn(2, 3, 16)).shape == (2, 3, 4)


def test_multihead_shape_and_head_count():
    mha = MultiHeadAttention(n_embd=16, n_head=4, block_size=8, dropout=0.0)
    assert mha(torch.randn(2, 8, 16)).shape == (2, 8, 16)
    heads = [m for m in mha.modules() if isinstance(m, Head)]
    assert len(heads) == 4


def test_multihead_is_causal():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4, 8, 0.0)
    mha.eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[0, 6:] = torch.randn(2, 16)
    assert torch.allclose(mha(x)[0, :6], mha(x2)[0, :6], atol=1e-6)


# ------------------------------------------------------------
# 4. Transformer block
# ------------------------------------------------------------

def test_feedforward_shape():
    ff = FeedForward(16, 0.0)
    assert ff(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_block_shape_and_layernorms():
    b = Block(16, 2, 8, 0.0)
    assert b(torch.randn(2, 8, 16)).shape == (2, 8, 16)
    lns = [m for m in b.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(lns) == 2


def test_block_has_residual_connection():
    b = Block(16, 2, 8, 0.0)
    b.eval()
    with torch.no_grad():
        for p in b.parameters():
            p.zero_()
    x = torch.randn(2, 8, 16)
    assert torch.allclose(b(x), x, atol=1e-6), "Block must add its input back: x = x + f(x)"


# ------------------------------------------------------------
# 5. GPT model
# ------------------------------------------------------------

def _gpt(V):
    return GPTLanguageModel(V, **TINY)


def test_gpt_forward_shapes():
    torch.manual_seed(0)
    chars, stoi, itos, data = _data()
    m = _gpt(len(chars))
    x, y = get_batch(data, 4, TINY["block_size"])
    logits, loss = m(x, y)
    assert logits.shape == (4, TINY["block_size"], len(chars))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_gpt_is_causal_end_to_end():
    torch.manual_seed(0)
    m = _gpt(20)
    m.eval()
    x = torch.randint(0, 20, (1, 8))
    x2 = x.clone()
    x2[0, 4:] = (x2[0, 4:] + 1) % 20
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[0, :4], l2[0, :4], atol=1e-5)


def test_gpt_uses_position_embedding():
    torch.manual_seed(0)
    m = _gpt(20)
    m.eval()
    x = torch.full((1, 8), 3, dtype=torch.long)
    logits, _ = m(x)
    assert not torch.allclose(logits[0, 0], logits[0, 7], atol=1e-4)


def test_gpt_generate_beyond_block_size():
    m = _gpt(20)
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=3 * TINY["block_size"])
    assert out.shape == (1, 1 + 3 * TINY["block_size"])


# ------------------------------------------------------------
# 6. Training
# ------------------------------------------------------------

def test_estimate_loss_keys_and_restores_train_mode():
    torch.manual_seed(0)
    chars, stoi, itos, data = _data()
    tr, va = split_data(data)
    m = _gpt(len(chars))
    m.train()
    out = estimate_loss(m, tr, va, batch_size=4, block_size=TINY["block_size"], eval_iters=3)
    assert set(out) == {"train", "val"}
    assert all(isinstance(v, float) for v in out.values())
    assert m.training, "estimate_loss must call model.train() before returning"


def test_train_loss_decreases():
    torch.manual_seed(0)
    chars, stoi, itos, data = _data()
    tr, va = split_data(data)
    m = _gpt(len(chars))
    cfg = dict(TINY, batch_size=16, lr=3e-3, max_iters=300, eval_interval=100, eval_iters=10)
    history = train(m, tr, va, cfg, log=lambda *a, **k: None)
    first, last = history[0][1], history[-1][1]
    assert last < first - 0.5, f"train loss should drop clearly: {first:.3f} -> {last:.3f}"
