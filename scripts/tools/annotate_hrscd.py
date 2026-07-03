#!/usr/bin/env python3
"""
HRSCD / DOTA Dataset Annotation Script using DashScope API (v1.24+ compatible)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import dashscope
import urllib3
import mimetypes
from dashscope import MultiModalConversation
# ✅ 新接口导入
from dashscope import Files, Generation

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hrscd_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HRSCDAnnotator:
    def __init__(self, api_key: str):
        dashscope.api_key = api_key

        self.caption_prompt = (
            "Please describe the content of the following remote sensing image in detail. "
            "First, give an overall summary of the geographical and functional types (e.g., city, forest, airport, farmland, port, etc.). "
            "Then divide the image conceptually into nine parts (from top-left to bottom-right) and describe the main objects, structures, "
            "and their colors, densities, and spatial layouts in each part. "
            "Finally, summarize the overall spatial composition and provide high-level reasoning, including possible environmental, "
            "functional, or socioeconomic implications that can be inferred from the image (e.g., human activity, traffic importance, "
            "urban planning, ecological balance)."
        )

    # ✅ 使用 Files.upload (DashScope v1.23+)
    def upload_image(self, image_path: str, max_retries: int = 3) -> str:
        """Upload image to DashScope and return its file_id"""
        for attempt in range(max_retries):
            try:
                result = Files.upload(image_path)
                if hasattr(result, "id"):
                    return result.id
                else:
                    logger.warning(f"⚠️ Unexpected upload result: {result}")
            except Exception as e:
                logger.warning(f"❌ Image upload failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        return None

    # ✅ 使用 Generation.call (新版多模态生成接口)
    def call_dashscope_api(self, image_path: str, max_retries: int = 3) -> str:
        """Call DashScope multimodal API using uploaded file_id"""
        # file_id = self.upload_image(image_path)
        # if not file_id:
        #     return None

        for attempt in range(max_retries):
            try:
                response = MultiModalConversation.call(
                    model="qwen-vl-max",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"text": self.caption_prompt},
                                {"image": image_path}
                            ],
                        }
                    ],
                    timeout=300
                )

                if hasattr(response, "output") and hasattr(response.output, "choices"):
                    content = response.output.choices[0].message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", "")
                return None

            except Exception as e:
                logger.warning(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        return None

    def collect_image_paths(self, dataset_root: str) -> List[str]:
        """Recursively collect all image files under dataset_root."""
        image_paths = []
        dataset_path = Path(dataset_root)
        exts = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}

        for file_path in dataset_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in exts:
                image_paths.append(str(file_path))

        logger.info(f"Found {len(image_paths)} images in dataset.")
        return image_paths

    def annotate_dataset(self, dataset_root: str, output_file: str, resume_from: int = 0):
        """Annotate all images and save results incrementally."""
        image_paths = self.collect_image_paths(dataset_root)
        if resume_from > 0:
            image_paths = image_paths[resume_from:]

        results = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception:
                pass

        for idx, image_path in enumerate(tqdm(image_paths, desc="Annotating images")):
            caption = self.call_dashscope_api(image_path)
            if caption:
                result_entry = {
                    "id": f"HRSCD_{idx + resume_from}",
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n[caption] {self.caption_prompt}"},
                        {"from": "gpt", "value": caption}
                    ]
                }
                results.append(result_entry)

                # 每 10 张保存一次
                if (idx + 1) % 10 == 0:
                    self.save_results(results, output_file)
                    logger.info(f"💾 Progress saved: {len(results)} images processed")

            else:
                logger.warning(f"⚠️ Skipped image due to failure: {image_path}")

            time.sleep(1)

        self.save_results(results, output_file)
        logger.info(f"✅ Annotation completed. Results saved to {output_file}")

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save JSON results."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Annotate remote sensing images with DashScope.")
    parser.add_argument("--dataset-root", required=True, help="Image dataset directory.")
    parser.add_argument("--output-file", required=True, help="Output JSON file.")
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running this script.")

    annotator = HRSCDAnnotator(api_key)
    logger.info("🚀 Starting annotation (DashScope v1.24 API mode)...")
    annotator.annotate_dataset(args.dataset_root, args.output_file, args.resume_from)


if __name__ == "__main__":
    main()
