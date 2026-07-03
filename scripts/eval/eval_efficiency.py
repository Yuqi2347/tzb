
# -*- coding: utf-8 -*-
import argparse
import os
import warnings
import re
warnings.filterwarnings("ignore", category=FutureWarning, message=".*image_processor_class.*")
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str,
                    default=os.environ.get("MODEL_PATH", "models/FM9G4B-V"))
parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
parser.add_argument("--image-folder", type=str, default=None)
parser.add_argument("--question-file", type=str, default="")
parser.add_argument("--data-name", type=str, default="")
parser.add_argument("--answers-file", type=str, default="", help="./answer_cap_v2.json")
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--eval-type", type=str, default="", help="ref or None")
# 添加新的控制参数
parser.add_argument("--no-merge", action="store_true",default=False,help="不合并结果，每个进程单独输出文件")
parser.add_argument("--output-dir", type=str, default="./eval_results", help="输出目录")
args = parser.parse_args()

import json
# from peft import PeftModel
from tqdm import tqdm
# import shortuuid
import base64
from PIL import Image
from io import BytesIO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.distributed as dist
import math
# from torchinfo import summary
import sys
import time
# from datetime import timedelta
import tifffile
import warnings
import numpy as np
MAX_PIXELS = 1000000000  # Pillow 默认限制

def load_and_resize_safe(image_file):
    try:
        ext = os.path.splitext(image_file)[-1].lower()
        
        # TIFF 图像处理（用 tifffile）
        if ext in ['.tif', '.tiff']:
            img_array = tifffile.imread(image_file)
            
            # 灰度图转3通道
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] > 3:
                img_array = img_array[..., :3]  # 保留前3通道

            # 确保类型为 uint8
            if img_array.dtype != np.uint8:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            total_pixels = img_array.shape[0] * img_array.shape[1]
            if total_pixels > MAX_PIXELS:
                scale = (MAX_PIXELS / total_pixels) ** 0.5
                new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
                print(f"[TIFF过大] {image_file}，原始尺寸 {img_array.shape[1]}x{img_array.shape[0]}，缩放到 {new_size}")
                image = Image.fromarray(img_array).resize(new_size, Image.LANCZOS).convert("RGB")
            else:
                image = Image.fromarray(img_array).convert("RGB")
            return image
        
        # 非TIFF图像（用 PIL 处理）
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(image_file).convert("RGB")
                return image

    except Image.DecompressionBombWarning:
        with Image.open(image_file) as img:
            width, height = img.size
            total_pixels = width * height
            print(f"[警告] 图像过大：{image_file}，分辨率 {width}x{height}，像素总数 {total_pixels}")

            scale = (MAX_PIXELS / total_pixels) ** 0.5
            new_size = (int(width * scale), int(height * scale))
            print(f"--> 正在缩放到安全范围：{new_size[0]}x{new_size[1]}")
            resized_img = img.resize(new_size, Image.LANCZOS).convert("RGB")
            return resized_img

    except Exception as e:
        print(f"[错误] 无法加载图像 {image_file}：{e}")
        return None


def setup_distributed():
    """简化的分布式初始化"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()



from ptflops import get_model_complexity_info

def compute_gflops_precise(model, input_shape=(3, 224, 224)):
    model.eval()
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
    gflops = macs / 1e9
    print(f"模型参数: {params / 1e6:.2f} M")
    print(f"计算复杂度: {gflops:.2f} GFLOPs")

def compute_gflops_gpu(model, input_shape=(1, 3, 224, 224), device=None):
    """在GPU上计算GFLOPs和推理时间"""
    model.eval()
    # 确保模型在GPU上
    model = model.to(device=device, dtype=torch.float16)
    
    # 创建测试输入
    dummy_input = torch.randn(input_shape, device=device, dtype=torch.float16)
    
    # 预热
    print("🔥 GPU推理预热中...")
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            torch.cuda.synchronize()  # 确保GPU计算完成
        except Exception as e:
            print(f"⚠️ 预热失败: {e}")
            return
    
    # 正式测试
    num_runs = 5
    times = []
    
    print(f"🔄 开始GPU推理速度测试 ({num_runs} 次运行)...")
    for i in range(num_runs):
        # 使用CUDA事件进行更精确的时间测量
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"⚠️ 推理失败 (运行 {i+1}): {e}")
                continue
        
        end_event.record()
        torch.cuda.synchronize()  # 确保GPU计算完成
        
        # 获取GPU时间（毫秒）
        gpu_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        times.append(gpu_time)
        print(f"   运行 {i+1}: {gpu_time:.4f} 秒")
    
    if not times:
        print("❌ 所有推理运行都失败了")
        return
    
    # 计算统计信息
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 估算FLOPs
    total_params = sum(p.numel() for p in model.parameters())
    flops_per_forward = total_params * 2  # 粗略估算
    gflops = (flops_per_forward / avg_time) / 1e9
    
    print(f"\n⚡ GPU推理性能结果:")
    print(f"   设备: {device}")
    print(f"   平均推理时间: {avg_time:.4f} 秒")
    print(f"   最快推理时间: {min_time:.4f} 秒")
    print(f"   最慢推理时间: {max_time:.4f} 秒")
    print(f"   参数量: {total_params / 1e6:.2f} M")
    print(f"   估算GFLOPs: {gflops:.2f}")
    print(f"   成功运行次数: {len(times)}/{num_runs}")
    
    # 显示GPU内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        print(f"   GPU内存使用: {memory_allocated:.2f} GB (已分配) / {memory_reserved:.2f} GB (已保留)")


# Dataset classes (保持不变)
class VRSDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt
        self.eval_type = eval_type

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = self.image_folder + line["image_id"]
        qs = line["question"]
        qs = DEFAULT_IMAGE_TOKEN + f"[{self.eval_type}]" + qs

        qs = qs + self.prompt
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        image = Image.open(image_file).convert("RGB")
        return qs, image

    def __len__(self):
        return len(self.questions)

class ValidDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file):

        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt
        self.eval_type = eval_type
        self.question_file = question_file


    def __getitem__(self, index):
        if self.eval_type == 'caption':
            line = self.questions[index]
            image_file = self.image_folder + line["Image"]
            
            qs = DEFAULT_IMAGE_TOKEN + f"[{self.eval_type}]"
            if "zh" in self.question_file:
                qs = qs + '''您是遥感图像描述方面的专家。请根据以下要求详细描述图像内容。
                描述步骤：1、简要分析遥感图像区域。2、将图像平均分为九份，从左上角到右下角逐区域描述区域内内容。3、根据各区域内容总结推理该区域的特征和功能，阐述理由。
                请将上述描述以纯文本段落回复，无结构化输出。
                '''
            else:
                qs = qs + '''You are an expert in remote sensing image description. Please describe the image content in detail according to the following requirements. 
                Description steps: 1. Briefly analyze the remote sensing image area. 2. Divide the image into nine equal parts and describe the content within the competition area from the top left corner to the bottom right corner. 3. Summarize and infer the characteristics and functions of each region based on its content, and explain the reasons.
                Please reply to the above description in plain text paragraphs without structured output.
                '''


        if self.eval_type == 'mcq':
            line = self.questions[index]
            choices = line['Answer choices']
            image_file = self.image_folder + line["Image"]
            qs = line["Text"]
            qs = DEFAULT_IMAGE_TOKEN + f"[{self.eval_type}]" + qs
            if "zh" in self.question_file:
                choice_prompt = ' 选项如下: \n'
            else:
                choice_prompt = ' The choices are listed below: \n'
            for choice in choices:
                choice_prompt += choice + "\n"
            
            if "zh" in self.question_file:
                qs += choice_prompt + "根据图片选择上述多项选择题的最佳答案。仅用字母A、B、C、D或E进行响应。只输出字母——从A到E的单个字符，不要任何解释，也不要标点符号。"
            else:
                qs += choice_prompt + 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter A, B, C, D, or E. Output only the letter — a single character from A to E, with no explanation and no punctuation.'
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        image = load_and_resize_safe(image_file)
        return qs, image

    def __len__(self):
        return len(self.questions)




class MMEDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file):

        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt
        self.eval_type = eval_type

    def __getitem__(self, index):
        line = self.questions[index]
        choices = line['Answer choices']
        image_file = self.image_folder + line["Image"]
        qs = line["Text"]
        qs = DEFAULT_IMAGE_TOKEN + qs

        choice_prompt = ' The choices are listed below: \n'
        for choice in choices:
            choice_prompt += choice + "\n"
        qs += choice_prompt + 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter A, B, C, D, or E. Output strictly only response the letter — a single character from A to E, with no explanation and no punctuation.'
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        image = Image.open(image_file).convert("RGB")
        return qs, image

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    return batch


import torch.nn.init as init


def disable_torch_init():
    init.kaiming_uniform_ = lambda *args, **kwargs: None
    init.kaiming_normal_ = lambda *args, **kwargs: None
    init.xavier_uniform_ = lambda *args, **kwargs: None
    init.xavier_normal_ = lambda *args, **kwargs: None


def create_data_loader(questions, data_name, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file):
    batch_size=1
    num_workers=4
    assert batch_size == 1, "batch_size must be 1"
    if data_name == 'vrs':
        dataset = VRSDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file)
    elif data_name == 'mme':
        dataset = MMEDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file)
    elif data_name == 'valid':
        dataset = ValidDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt, eval_type,question_file)
    else:
        print("dataset not exist!! Please check the Dataset Configuration")
        sys.exit()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return data_loader


def load_pretrained_model(model_path, model_name, device):
    from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 使用AutoModel而不是AutoModelForCausalLM（因为这是自定义模型）
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    # 添加 token（和训练时一样）
    tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
    for token in tokens_to_add:
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    # 确保所有参数都是Float16（有些层可能没被转换）
    model = model.to(dtype=torch.float16, device=device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    model.eval()
    return tokenizer, model, processor, config.max_position_embeddings


def get_data_chunk(questions, world_size, rank):
    """简单的数据分割，避免复杂的同步操作"""
    total_size = len(questions)
    chunk_size = total_size // world_size
    remainder = total_size % world_size

    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

    return questions[start_idx:end_idx]

def extract_answer(text, eval_type=None):
    if eval_type == 'ref':
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        matches = re.search(r'\[([^\[\]]+)\](?!.*\[)', text)
        print(f'text:{text},matches:{matches.group(0)}')


        if matches:
            # 取最后一个匹配内容，并解析为 Python 列表
            try:
                # 加上中括号后转换为列表
                result = matches.group(0)
                return result
            except:
                return None  # 解析失败
        else:
            return text  # 没有匹配
    else:
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text
def eval_model(args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # === 初始化分布式 ===
    local_rank = setup_distributed()
    is_main_process = local_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    disable_torch_init()

    # === 加载模型 ===
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, "9g", device)
    compute_gflops_gpu(model, input_shape=(1, 3, 224, 224), device=device)
    # 只在多GPU时使用DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        print(f"Rank {local_rank}: Using DDP with {world_size} GPUs")
        model_without_ddp = model.module  # 用于访问原始模型
    else:
        print("Single GPU mode, skipping DDP")
        model_without_ddp = model  # 单GPU模式下直接使用模型

    # === 每个进程独立读取和处理数据 ===
    # 避免广播，每个进程自己读取数据
    with open(args.question_file, 'r') as f:
        questions = json.load(f)
    # 获取当前进程的数据分片
    local_questions = get_data_chunk(questions, world_size, local_rank)

    print(f"Rank {local_rank}: Processing {len(local_questions)} / {len(questions)} questions")

    # === prompt 设置 ===
    prompt_text = {
        "ref": "Please identify the location of the described target in the image and express it as a normalized bounding box in the format: [x1, y1, x2, y2], where x1 and y1 represent the top-left corner and x2 and y2 represent the bottom-right corner, all normalized by image width and height; strictly output only the answer, without any additional text or explanation, for example: [0.35, 0.52, 0.52, 0.68]",
        "vqa": "only response the answer, not any explaination",
        "cap": "Describe in several sentences, not over Structured Language."
    }
    prompt = prompt_text.get(args.eval_type, "")

    # === 创建 DataLoader ===
    data_loader = create_data_loader(
        local_questions,
        args.data_name,
        args.image_folder,
        tokenizer,
        image_processor,
        model_without_ddp.config,
        prompt=prompt,
        eval_type=args.eval_type,
        question_file=args.question_file
    )

    # === 每个进程独立评估 ===
    local_results = []
    start_time = time.time()
    
    # 添加推理时间统计
    inference_times = []
    total_inference_time = 0.0

    for idx, (batch,) in enumerate(tqdm(data_loader, desc=f"Rank {local_rank}")):
        try:
            line = local_questions[idx].copy()
            prompt_template = (
                "You are a remote sensing assistant. For each question, think step by step using <think>...</think> and then give your answer in <answer>...</answer>."
            )
            if args.data_name == 'cdchat':
                prompt_text, image_before, image_after = batch
                inputs = image_processor(text=prompt_text, images=[image_before, image_after], return_tensors="pt")
            else:
                prompt_text, image = batch
                msgs = [{"role": "user", "content": prompt_text},
                    # {"role": "system", "content": prompt_template}
                    ]
                prompt_input = image_processor.tokenizer.apply_chat_template(msgs,
                                           tokenize=False,
                                           add_generation_prompt=True)
                # print(prompt_text)
                inputs = image_processor(text=prompt_input, images=image, return_tensors="pt")
            
            # 统一处理：将所有tensor移到GPU并转换为float16（如果是浮点数）
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    # 如果是浮点tensor，转为float16；否则只移动设备
                    if obj.dtype in [torch.float32, torch.float16, torch.float64]:
                        return obj.to(device=device, dtype=torch.float16)
                    else:
                        return obj.to(device=device)
                elif isinstance(obj, list):
                    return [move_to_device(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(move_to_device(item) for item in obj)
                else:
                    return obj
            
            inputs = {k: move_to_device(v) for k, v in inputs.items()}

            # 使用autocast确保整个推理过程都在float16下进行
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 记录推理开始时间
                inference_start_time = time.time()
                
                output_ids = model_without_ddp.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    tgt_sizes=inputs["tgt_sizes"],
                    image_bound=inputs["image_bound"],
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=False,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    num_beams=args.num_beams,
                    use_cache=True,
                )
                
                # 记录推理结束时间
                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time
                inference_times.append(inference_time)
                total_inference_time += inference_time

            output_text = tokenizer.batch_decode(output_ids,skip_special_tokens=False)[0].strip().replace("<|im_end|>","").replace("<|im_start|>","")
            print("output:",output_text)
            line['answer'] = extract_answer(output_text, args.eval_type).strip()
            print(line['answer'])
            local_results.append(line)

            # # ===== 添加显存清理代码 =====
            # # 删除中间变量释放显存
            # del inputs, output_ids, output_text
            # # 清空CUDA缓存
            # if (idx + 1) % 10 == 0:  # 每10个样本清理一次
            #     torch.cuda.empty_cache()
            # # ============================

            # 定期打印进度
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = (len(local_questions) - idx - 1) * avg_time
                
                # 计算推理时间统计
                avg_inference_time = total_inference_time / len(inference_times) if inference_times else 0
                min_inference_time = min(inference_times) if inference_times else 0
                max_inference_time = max(inference_times) if inference_times else 0
                
                print(f"Rank {local_rank}: {idx + 1}/{len(local_questions)} "
                      f"({(idx + 1) / len(local_questions) * 100:.1f}%) "
                      f"Speed: {(idx + 1) / elapsed:.2f} it/s "
                      f"ETA: {remaining:.1f}s")
                print(f"推理时间统计: 平均={avg_inference_time:.4f}s, 最快={min_inference_time:.4f}s, 最慢={max_inference_time:.4f}s")

        except Exception as e:
            print(f"Error processing sample {idx} on rank {local_rank}: {e}")
            line = local_questions[idx].copy()
            line['answer'] = f"ERROR: {str(e)}"
            local_results.append(line)

    # === 打印最终推理时间统计 ===
    if inference_times:
        avg_inference_time = total_inference_time / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        total_elapsed = time.time() - start_time
        
        print(f"\n📊 Rank {local_rank} 最终推理时间统计:")
        print(f"   总样本数: {len(local_questions)}")
        print(f"   成功推理: {len(inference_times)}")
        print(f"   平均推理时间: {avg_inference_time:.4f} 秒")
        print(f"   最快推理时间: {min_inference_time:.4f} 秒")
        print(f"   最慢推理时间: {max_inference_time:.4f} 秒")
        print(f"   总推理时间: {total_inference_time:.2f} 秒")
        print(f"   总评估时间: {total_elapsed:.2f} 秒")
        print(f"   推理时间占比: {(total_inference_time/total_elapsed)*100:.1f}%")
        print(f"   平均吞吐量: {len(inference_times)/total_elapsed:.2f} 样本/秒")
    else:
        print(f"\n⚠️ Rank {local_rank}: 没有成功的推理记录")

    # === 每个进程独立输出结果 ===
    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_merge:
        # 每个进程输出独立文件
        output_file = os.path.join(args.output_dir, f"results_rank_{local_rank}.json")
        with open(output_file, "w") as f:
            for line in local_results:
                f.write(json.dumps(line) + "\n")
        print(f"Rank {local_rank}: Wrote {len(local_results)} results to {output_file}")
    else:
        # 写入临时文件，由主进程合并
        temp_file = os.path.join(args.output_dir, f"temp_rank_{local_rank}.json")
        with open(temp_file, "w") as f:
            for line in local_results:
                f.write(json.dumps(line) + "\n")
        print(f"Rank {local_rank}: Wrote {len(local_results)} results to {temp_file}")

        # 简单的文件检查合并（避免分布式同步）
        if is_main_process:
            print("Main process waiting for all ranks to finish...")
            # 等待所有临时文件出现
            max_wait = 3600000  # 最多等待1小时
            wait_time = 0
            all_files_exist = False

            while wait_time < max_wait and not all_files_exist:
                all_files_exist = True
                for rank in range(world_size):
                    temp_file = os.path.join(args.output_dir, f"temp_rank_{rank}.json")
                    if not os.path.exists(temp_file):
                        all_files_exist = False
                        break

                if not all_files_exist:
                    time.sleep(10)  # 等待10秒
                    wait_time += 10

            if all_files_exist:
                print("All temporary files found, merging results...")
                final_results = []

                for rank in range(world_size):
                    temp_file = os.path.join(args.output_dir, f"temp_rank_{rank}.json")
                    try:
                        with open(temp_file, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    final_results.append(json.loads(line))
                        print(f"Merged results from rank {rank}")
                    except Exception as e:
                        print(f"Error reading results from rank {rank}: {e}")

                # 写入最终结果
                with open(args.answers_file, "w", encoding="utf-8") as f:
                    for line in final_results:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")


                print(f"Merged {len(final_results)} total results to {args.answers_file}")

                # 清理临时文件
                for rank in range(world_size):
                    temp_file = os.path.join(args.output_dir, f"temp_rank_{rank}.json")
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                print("Evaluation completed successfully!")
            else:
                print("Timeout waiting for all processes to complete!")

    # 清理（不强制同步）
    cleanup_distributed()


if __name__ == "__main__":
    print("\n========= 参数确认 (args) =========")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("====================================\n")

    try:
        eval_model(args)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        cleanup_distributed()
        raise
