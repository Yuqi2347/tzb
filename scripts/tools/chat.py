"""
Name chat.py
Date 2025/5/6 11:20
Version 1.0
TODO: Test multi-image mode (1 overview + 4 quadrants)
"""

import torch
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoProcessor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def create_multi_view_images(image, max_overview_size=10000):
    """
    将大图像切分为5张图：
    1. overview（缩小的全局图）
    2-5. 四个象限的高清图
    
    Args:
        image: PIL Image对象
        max_overview_size: overview图的最大边长
    
    Returns:
        list of PIL Images: [overview, top_left, top_right, bottom_left, bottom_right]
    """
    width, height = image.size
    print(f"Original image size: {width} x {height}")
    
    images = []
    
    # 1. 创建overview（缩小的全图）
    if max(width, height) > max_overview_size:
        scale = max_overview_size / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        overview = image.resize(new_size, Image.LANCZOS)
        print(f"Overview size: {new_size[0]} x {new_size[1]}")
    else:
        overview = image.copy()
        print(f"Overview size: {width} x {height} (no resize)")
    
    images.append(overview)
    
    # 2-5. 切分为4个象限
    mid_x = width // 2
    mid_y = height // 2
    
    # 添加一些重叠区域（overlap），避免目标被切断
    overlap = min(200, min(width, height) // 10)  # 10%重叠或200px
    
    quadrants = [
        (0, 0, mid_x + overlap, mid_y + overlap, "Top-Left"),
        (mid_x - overlap, 0, width, mid_y + overlap, "Top-Right"),
        (0, mid_y - overlap, mid_x + overlap, height, "Bottom-Left"),
        (mid_x - overlap, mid_y - overlap, width, height, "Bottom-Right")
    ]
    
    for x1, y1, x2, y2, name in quadrants:
        # 确保坐标在合法范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        quadrant = image.crop((x1, y1, x2, y2))
        print(f"{name} quadrant: {quadrant.size[0]} x {quadrant.size[1]}")
        images.append(quadrant)
    
    print(f"Created {len(images)} images total")
    return images


if __name__ == '__main__':
    prompt = f"""
    You are a remote sensing assistant. For each question, think step by step using <think>...</think> and then give your answer in <answer>...</answer>.
    """

    model_file = os.environ.get("MODEL_PATH", "models/FM9G4B-V")
    
    print(f"Loading merged model from: {model_file}")
    # 直接加载合并后的模型（不需要 base_model，因为已经是完整模型）
    model = AutoModel.from_pretrained(
        model_file, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True  # 只使用本地文件
    )
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_file, trust_remote_code=True)
    # 添加 token（和训练时一样）
    tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
    for token in tokens_to_add:
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    
    model = model.eval().cuda()
    print("✅ Model loaded successfully!\n")

    # ========== 配置测试模式 ==========
    USE_MULTI_IMAGE = True  # 设置为True测试多图模式，False测试单图模式
    TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH", "examples/images/sample.tif")
    TEST_QUESTION = "[mcq] What is the direction of the building with the larger blue rectangular roof in the lower left corner relative to the building with the white rectangular roof in the residential area in the lower left corner?"  # 测试问题
    TEST_CHOICES = ["(A) Below", "(B) Above", "(C) On the left side", "(D) Right Side"]
    # ==================================

    # 加载原始图像
    original_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    print(f"\n{'='*60}")
    print(f"Testing {'MULTI-IMAGE' if USE_MULTI_IMAGE else 'SINGLE-IMAGE'} mode")
    print(f"{'='*60}\n")

    if USE_MULTI_IMAGE:
        # 多图模式：生成5张图（1个overview + 4个象限）
        images = create_multi_view_images(original_image, max_overview_size=10000)
        
        # 构建多图prompt（每张图一个IMAGE_TOKEN）
        image_tokens = "".join([DEFAULT_IMAGE_TOKEN] * len(images))
        question_text = (
            f"{prompt}"
            f"{image_tokens}{TEST_QUESTION}{TEST_CHOICES}\n"
            f"Note: You are provided with {len(images)} images:\n"
            f"- Image 1: Overview of the entire scene\n"
            f"- Image 2-5: High-resolution quadrants (top-left, top-right, bottom-left, bottom-right)\n"
            f"Please analyze all images and provide your answer."
        )
        
        # 使用apply_chat_template格式
        msgs = [{"role": "user", "content": question_text}]
        prompt = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        print(f"Prompt:\n{prompt}\n")
        print(f"Processing {len(images)} images...")
        
        # 传入多张图像
        inputs = processor(text=prompt, images=images, return_tensors="pt").to('cuda')
        
    else:
        # 单图模式（原来的方式）
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + TEST_QUESTION)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f"Prompt:\n{prompt}\n")
        
        inputs = processor(text=prompt, images=original_image, return_tensors="pt").to('cuda')

    # 打印输入信息
    print(f"Input shapes:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            if value and isinstance(value[0], torch.Tensor):
                print(f"    - First item shape: {value[0].shape}")
    
    print(f"\nGenerating output...")
    
    # 生成输出
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        tgt_sizes=inputs["tgt_sizes"],
        image_bound=inputs["image_bound"],
        tokenizer=tokenizer,
        max_new_tokens=512,
        repetition_penalty=1.1,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    
    output_text = tokenizer.batch_decode(output_ids)[0].strip()
    print("\n", "="*100, "\n")
    print("OUTPUT:")
    print(output_text)
    print("\n", "="*100, "\n")


    # # 第二轮聊天，传递多轮对话的历史信息
    # msgs.append({"role": "assistant", "content": [res]})
    # msgs.append({"role": "user", "content": ["图中有几个箱子?"]})

    # answer = model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer
    # )
    # print("\n", "="*100, "\n")
    # print(answer)


    ## 流式输出，设置：
    # sampling=True
    # stream=True
    ## 返回一个生成器
    # msgs = [{'role': 'user', 'content': [image, prompt]}]
    # res = model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer,
    #     sampling=True,
    #     stream=True
    # )
    # print("\n", "="*100, "\n")
    # generated_text = ""
    # for new_text in res:
    #     generated_text += new_text
    #     print(new_text, flush=True, end='')
