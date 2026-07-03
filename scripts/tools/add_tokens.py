#!/usr/bin/env python3
"""Add reasoning special tokens to a Hugging Face tokenizer directory."""

import argparse

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Tokenizer/model directory to update.")
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=["<think>", "</think>", "<answer>", "</answer>"],
        help="Special tokens to add.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    old_vocab_size = len(tokenizer)
    existing = [token for token in args.tokens if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id]
    missing = [token for token in args.tokens if token not in existing]

    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        tokenizer.save_pretrained(args.model_path)

    print(f"Existing tokens: {existing}")
    print(f"Added tokens: {missing}")
    print(f"Vocabulary size: {old_vocab_size} -> {len(tokenizer)}")


if __name__ == "__main__":
    main()
