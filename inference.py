# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/llama/llama.py
# Copyright © 2023 Apple Inc.

import argparse
import json
import time
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from classes import Llama

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    norm_eps: float
    vocab_size: int
    multiple_of: int
    model_type: str
    quantization: dict[str, int]

def tic():
    return time.time()

def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"

def generate(args):
    input("Press enter to start generation")
    print("------")
    print(args.prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(args.prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x, args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.max_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)

def few_shot_generate(args):
    def possible_end(s):
        word = "[Instruction]"
        for i in range(len(word) - 1, 0, -1):
            if s[-i:] == word[:i]:
                return 0
        if s[-len(word) :] == word:
                return 1
        return -1

    def generate(question):
        x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(question)])
        skip = 0
        prompt_processing = None
        tokens = []
        start = tic()
        for token in model.generate(x, args.temp):
            tokens.append(token)

            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)
                prompt_processing = toc("Prompt processing", start)

            if len(tokens) >= args.max_tokens:
                break

            mx.eval(tokens)
            token_list = [t.item() for t in tokens]
            s = tokenizer.decode(token_list)

            end = possible_end(s)
            if end == 0:
                continue
            if end == 1:
                skip = len(s)
                break

            print(s[skip:], end="", flush=True)
            skip = len(s)
            if token_list[-1] == tokenizer.eos_id():
                break

        mx.eval(tokens)
        full_gen = toc("Full generation", start)
        s = tokenizer.decode([t.item() for t in tokens])
        print(s[skip:], end="", flush=True)

    print("[INFO] Loading few-shot examples from: {}".format(args.few_shot))
    prompt = open(args.few_shot).read().strip()
    while True:
        question = input("Ask a question: ")
        generate(prompt.replace("{}", question))
        print()

def sanitize_config(config, weights):
    if 'quantization' in config:
        if 'group_size' in config['quantization'] and 'bits' in config['quantization']:
            config['quantization']['group_size'] = int(config['quantization']['group_size'])
            config['quantization']['bits'] = int(config['quantization']['bits'])
        else:
            raise ValueError("Both 'group_size' and 'bits' must be present in 'quantization'")
    else:
        config['quantization'] = None
    return config

def load_model(model_path):
    model_path = Path(model_path)

    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        print("[INFO] Loading model from {}.".format(unsharded_weights_path))
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)
        print("[INFO] Loading model from {}.".format(sharded_weights_glob))

        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())

    with open(model_path / "config.json", "r") as f:
        config = sanitize_config(json.loads(f.read()), weights)
    
    print(config)
    model_args = ModelArgs(**config)
    model = Llama(n_layers=model_args.n_layers, vocab_size=model_args.vocab_size, dim=model_args.dim, multiple_of=model_args.multiple_of, n_heads=model_args.n_heads)
    quantization = config.get("quantization", None)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument(
        "--model-path",
        help="Path to the model weights and tokenizer",
        default="mlx_model",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--few-shot",
        help="Read a few shot prompt from a file (as in `sample_prompt.txt`).",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)
    if args.few_shot:
        few_shot_generate(args)
    else:
        generate(args)
