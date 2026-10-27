import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from generate import apply_options, clean_prompt, generate_stream, parse_line, sample_next
from gpt import GPTLanguageModel

STOI = {ch: i for i, ch in enumerate('abc "\n')}


# ------------------------------------------------------------
# 1. Sampling
# ------------------------------------------------------------

def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 2.0, 0.5, 1.9])
    assert all(sample_next(logits, temperature=0) == 1 for _ in range(20))


def test_top_k_one_is_greedy():
    logits = torch.tensor([0.1, 2.0, 0.5, 1.9])
    assert all(sample_next(logits, temperature=1.0, top_k=1) == 1 for _ in range(50))


def test_top_k_never_samples_outside_the_k_likeliest():
    logits = torch.tensor([5.0, 4.0, 0.0, -1.0, -2.0])
    seen = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(300)}
    assert seen <= {0, 1}


def test_higher_temperature_spreads_the_samples():
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 1.0, 0.5, 0.0])
    cold = {sample_next(logits, temperature=0.1, top_k=0) for _ in range(300)}
    hot = {sample_next(logits, temperature=3.0, top_k=0) for _ in range(300)}
    assert len(hot) > len(cold)


# ------------------------------------------------------------
# 2. Prompt handling
# ------------------------------------------------------------

def test_clean_prompt_converts_curly_quotes_and_dashes():
    text, dropped = clean_prompt("“ab”", STOI)
    assert text == '"ab"' and dropped == []


def test_clean_prompt_drops_and_reports_unknown_characters():
    text, dropped = clean_prompt("abZc#", STOI)
    assert text == "abc" and dropped == ["#", "Z"]


# ------------------------------------------------------------
# 3. Generation loop
# ------------------------------------------------------------

def test_generate_stream_yields_the_requested_number_of_valid_ids():
    torch.manual_seed(0)
    model = GPTLanguageModel(len(STOI), 8, 16, 2, 2, 0.0).eval()
    out = list(generate_stream(model, [0, 1, 2], 25, block_size=8))
    assert len(out) == 25 and all(0 <= i < len(STOI) for i in out)


def test_generate_stream_handles_a_prompt_longer_than_the_context_window():
    torch.manual_seed(0)
    model = GPTLanguageModel(len(STOI), 8, 16, 2, 2, 0.0).eval()
    assert len(list(generate_stream(model, [1] * 40, 5, block_size=8))) == 5


# ------------------------------------------------------------
# 4. Settings typed in the session
# ------------------------------------------------------------

def test_parse_line_removes_a_setting_typed_at_the_end_of_a_prompt():
    prompt, options = parse_line("The case of the /temp 0.5")
    assert prompt == "The case of the " and options == [("temp", "0.5")]
    assert "temp" not in prompt and "/" not in prompt


def test_parse_line_handles_several_settings_anywhere():
    prompt, options = parse_line("/topk 10 The man /len 80 with the /temp 0.3")
    assert prompt.split() == ["The", "man", "with", "the"]
    assert options == [("topk", "10"), ("len", "80"), ("temp", "0.3")]


def test_parse_line_leaves_plain_prompts_alone():
    assert parse_line('"Watson," said Holmes, "') == ('"Watson," said Holmes, "', [])


def test_parse_line_ignores_a_setting_with_a_bad_value():
    assert parse_line("/temp abc")[1] == []


def test_apply_options_updates_only_what_was_given():
    assert apply_options([("temp", "0.3")], 0.8, 400, 40) == (0.3, 400, 40)
    assert apply_options([("len", "80"), ("topk", "10")], 0.8, 400, 40) == (0.8, 80, 10)


def test_apply_options_last_value_wins():
    assert apply_options([("temp", "0.2"), ("temp", "0.9")], 0.8, 400, 40)[0] == 0.9
