import argparse
import os
import re

import torch
import torch.nn.functional as F

from gpt import CONFIG, GPTLanguageModel, build_vocab, decode, encode, read_text

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------

# the training text was normalized to ASCII, so prompts must be too
ASCII_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "--", "–": "-"}

OPTION = re.compile(r"/(temp|len|topk)\s+(-?\d*\.?\d+)")
SETTINGS = ("/temp", "/len", "/topk")

HELP = """commands (they also work at the end of a prompt, e.g.  The case of the /temp 0.5):
  /temp 0.5   randomness: lower is safer and more repetitive, higher is wilder (0 = always the likeliest letter)
  /len 400    how many characters to write
  /topk 40    only sample from the 40 likeliest letters at each step (0 = no limit)
  /help       show this
  /quit       leave
anything else is the opening of a passage, and the model continues it"""


# ------------------------------------------------------------
# 2. Prompt handling and sampling
# ------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_prompt(text, stoi):
    for fancy, plain in ASCII_QUOTES.items():
        text = text.replace(fancy, plain)
    dropped = sorted({ch for ch in text if ch not in stoi})
    return "".join(ch for ch in text if ch in stoi), dropped


def parse_line(line):
    return OPTION.sub("", line), OPTION.findall(line)


def apply_options(options, temperature, n_chars, top_k):
    for name, value in options:
        if name == "temp":
            temperature = float(value)
        elif name == "len":
            n_chars = int(float(value))
        else:
            top_k = int(float(value))
    return temperature, n_chars, top_k


def sample_next(logits, temperature=0.5, top_k=40):
    # temperature 0 always picks the likeliest character; higher values flatten the distribution
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    # top-k: ignore everything except the k likeliest characters, which avoids rare junk
    if top_k and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))


@torch.no_grad()
def generate_stream(model, ids, n_chars, block_size, temperature=0.5, top_k=40):
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(n_chars):
        logits, _ = model(idx[:, -block_size:])
        nxt = sample_next(logits[0, -1], temperature, top_k)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=idx.device)], dim=1)
        yield nxt


# ------------------------------------------------------------
# 3. Loading the model
# ------------------------------------------------------------

def load_model(checkpoint, device):
    # the vocabulary and architecture must match the ones used in training
    chars, stoi, itos = build_vocab(read_text(os.path.join(HERE, "sherlock.txt")))
    c = CONFIG
    model = GPTLanguageModel(len(chars), c["block_size"], c["n_embd"], c["n_head"], c["n_layer"], c["dropout"])
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    except FileNotFoundError:
        raise SystemExit(f"no checkpoint at {checkpoint}. Run `python3 gpt.py` first; it saves elementarygpt.pt")
    except RuntimeError:
        raise SystemExit("the checkpoint does not match CONFIG in gpt.py (n_embd, n_head, n_layer or block_size changed since training)")
    return model.to(device).eval(), stoi, itos


# ------------------------------------------------------------
# 4. Interactive session
# ------------------------------------------------------------

def write(model, stoi, itos, prompt, n_chars, temperature, top_k):
    prompt, dropped = clean_prompt(prompt, stoi)
    if dropped:
        print(f"(the model does not know these characters and skipped them: {' '.join(map(repr, dropped))})")
    ids = encode(prompt or "\n", stoi)
    print(prompt, end="", flush=True)
    for ix in generate_stream(model, ids, n_chars, CONFIG["block_size"], temperature, top_k):
        print(itos[ix], end="", flush=True)
    print("\n")


def main():
    p = argparse.ArgumentParser(description="Continue a passage in the style of the Sherlock Holmes stories")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "elementarygpt.pt"))
    p.add_argument("--prompt", help="write one continuation and exit instead of starting the interactive session")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int)
    a = p.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
    model, stoi, itos = load_model(a.checkpoint, pick_device())

    if a.prompt is not None:
        write(model, stoi, itos, a.prompt, a.tokens, a.temperature, a.top_k)
        return

    n_chars, temperature, top_k = a.tokens, a.temperature, a.top_k
    print("elementaryGPT: type the opening of a passage and the model continues it.")
    print("It writes in the style of the Holmes stories; it does not answer questions. /help for commands.\n")
    # settings such as /temp 0.5 are accepted anywhere in a line and removed from the prompt
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = line.split()
        if cmd and cmd[0] == "/quit":
            break
        if cmd and cmd[0] == "/help":
            print(HELP)
            continue

        prompt, options = parse_line(line)
        if cmd and cmd[0].startswith("/") and cmd[0] not in SETTINGS:
            print("unknown command; /help lists them")
        elif cmd and cmd[0] in SETTINGS and not options:
            print(f"usage: {cmd[0]} <number>")
        else:
            if options:
                temperature, n_chars, top_k = apply_options(options, temperature, n_chars, top_k)
                print(f"temperature {temperature}, length {n_chars}, top-k {top_k}")
            if prompt.strip():
                write(model, stoi, itos, prompt, n_chars, temperature, top_k)


if __name__ == "__main__":
    main()
