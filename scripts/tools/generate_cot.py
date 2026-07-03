#!/usr/bin/env python3
"""Generate chain-of-thought training targets with an OpenAI-compatible API."""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=os.environ.get("COT_INPUT_JSON", "examples/data/sample_train.json"))
    parser.add_argument("--output-json", default=os.environ.get("COT_OUTPUT_JSON", "outputs/cot/sample_train_cot.json"))
    parser.add_argument("--image-base-path", default=os.environ.get("IMAGE_BASE_PATH", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("COT_MODEL", "gpt-4o"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def encode_image(image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {path}")
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def process_item(client, args, item):
    image_path = Path(args.image_base_path) / item["image"]
    image = encode_image(image_path)
    if image is None:
        return item

    conversations = item["conversations"]
    response = None
    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a remote sensing assistant. Given a question and answer, "
                            "write concise reasoning in <think>...</think>, then the final answer "
                            "in <answer>...</answer>."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "question: " + conversations[0]["value"]},
                            {"type": "text", "text": "answer: " + conversations[1]["value"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=args.temperature,
            )
            break
        except Exception as exc:
            print(f"Attempt {attempt + 1} failed for {item.get('id', '<unknown>')}: {exc}")
            time.sleep(1.5 * (attempt + 1))

    if response is None:
        print(f"Failed all attempts for {item.get('id', '<unknown>')}")
        return item

    item["conversations"].append({"from": "gpt", "value": response.choices[0].message.content})
    return item


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running this script.")

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, client, args, item) for item in raw_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
