import re
import time
import os
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:  # pragma: no cover - optional metric dependency
    sentence_bleu = None
    SmoothingFunction = None
# from rouge_score import rouge_scorer
# from pycocoevalcap.cider.cider import Cider
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional GPT scoring dependency
    OpenAI = None


def _build_openai_client():
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("RGRPO_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
    )


client = _build_openai_client()


def compute_caption_score(answer,gt):
    if client is None:
        raise RuntimeError("Set OPENAI_API_KEY or RGRPO_API_KEY before using GPT-based caption scoring.")

    max_retries = 10
    response = None
    
    # 确保输入是字符串类型
    answer = str(answer) if answer is not None else ""
    gt = str(gt) if gt is not None else ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a VQA assessment expert.Given the reference answer: answer,And the model answer: gt. Please rate the accuracy, completeness and fluency of the answer with a decimal between 0 and 1, with two decimal places. Your response must be strictly a two-decimal score, and cannot contain any other content"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "answer:" + answer},
                            {"type": "text", "text": "gt:" + gt},
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0.7
            )
            break  # 成功就退出
        except Exception as e:
            print(f"⚠️ attempt {attempt+1} failed with error: {e}")
            time.sleep(1.5 * (attempt + 1))  # 退避重试

    if response is None:
        raise RuntimeError("Caption scoring failed after all retry attempts.")

    content = response.choices[0].message.content
    print(f" content: {content}")
    return float(content)



def compute_bleu_avg(gen: str, gt: str) -> float:
    """
    if sentence_bleu is None or SmoothingFunction is None:
        raise RuntimeError("Install nltk before using BLEU-based caption rewards.")

    计算 BLEU-1、BLEU-2、BLEU-4 的平均值，返回 [0,1] 之间的分数。
    gen: 生成的文本
    gt: 参考答案（单个字符串）
    """
    # 确保输入是字符串类型
    gen = str(gen) if gen is not None else ""
    gt = str(gt) if gt is not None else ""
    
    # 分词
    pred_tokens = gen.lower().split()
    ref_tokens = gt.lower().split()
    
    # 平滑策略，避免 0 分
    smoothing = SmoothingFunction().method1
    
    # 计算 BLEU-1 (只看 1-gram)
    bleu1 = sentence_bleu(
        [ref_tokens], 
        pred_tokens,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smoothing
    )
    
    # 计算 BLEU-2 (1-gram 和 2-gram)
    bleu2 = sentence_bleu(
        [ref_tokens], 
        pred_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing
    )
    
    # 计算 BLEU-4 (1-gram 到 4-gram)
    bleu4 = sentence_bleu(
        [ref_tokens], 
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    )
    
    # 返回平均值
    avg_bleu = (bleu1 + bleu2 + bleu4) / 3.0
    return avg_bleu
def compute_rouge_l(gen: str, refs: list[str]) -> float:
    """
    计算 ROUGE-L F1 分数，返回 [0,1] 之间的分数。
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # 对每个参考都算一下，最后取平均
    scores = [scorer.score(r, gen)['rougeL'].fmeasure for r in refs]
    return sum(scores) / len(scores)


    # 在模块最外层只实例化一次 Cider，以加快后续多次调用
    _cider_evaluator = Cider()
def compute_cider(gen: str, refs: list[str]) -> float:
    
    """
    计算 CIDEr 分数，返回一个通常在 [0,5] 之间的值，可按需归一化到 [0,1]。
    """
    # pycocoevalcap 接口要求 dict 格式
    # gen_dict: {'0': ['generated sentence']}
    # ref_dict: {'0': ['ref1', 'ref2', ...]}
    gen_dict = {'0': [gen]}
    ref_dict = {'0': refs}
    # 返回 (score_dict, scores_list)；我们取 scores_list[0]
    _, scores = _cider_evaluator.compute_score(ref_dict, gen_dict)
    cider_score = scores[0]
    # 如果你需要将其归一化到 [0,1]，可以除以 5.0
    return cider_score / 5.0




def extract_answer(resp: str) -> str:
    """
    提取 resp 中第一个 <answer>…</answer> 标签内部的文本，
    如果没有匹配则返回空字符串。
    """
    m = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
    return m.group(1).strip() if m else ""


# compute_reward.py


def compute_reward(prompts, responses, ground_truths,
                   think_r=0.5, answer_r=0.5,
                   vqa_r=1.0, caption_r=1.0,
                   grounding_r=1.0, penalty_r=-1.0):
    """
    计算 GRPO 的复合 reward：
      - 结构奖励（think & answer 标签）
      - 任务奖励（VQA/Caption/Grounding）
      - 格式惩罚
    """
    for (prompt, resp, gt) in zip([prompts], [responses], [ground_truths] or [None]*len(responses)):
        r = 0.0
        # 结构奖励
        # print(f"prompt:{prompt}, resp:{resp}, gt:{gt}")
        # if "<think>" in resp and "</think>" in resp:
        #     r += think_r
        # if "<answer>" in resp and "</answer>" in resp:
        #     r += answer_r
        # 任务奖励
        # 确保 gt 是字符串类型
        if gt is not None:
            gt = str(gt).lower()
        else:
            gt = ""
        answer_text = extract_answer(resp).lower()
        
        # ========== VQA任务：多层次奖励（方案1） ==========
        if "[vqa]" in prompt.lower():
            # Level 1: 完全匹配（最高奖励）
            if gt == answer_text:
                r += vqa_r * 3.0
            # Level 2: 部分包含
            elif gt in answer_text or answer_text in gt:
                r += vqa_r * 2.0
            else:
                # Level 3: 使用编辑距离/相似度（连续奖励）
                import difflib
                similarity = difflib.SequenceMatcher(None, gt, answer_text).ratio()
                r += vqa_r * similarity * 1.5  # 0-1.5分的连续奖励
                
                # Level 4: 长度合理性奖励（避免生成过长/过短）
                if len(answer_text) > 0:
                    len_ratio = len(answer_text) / max(len(gt), 1)
                    if 0.5 <= len_ratio <= 2.0:
                        r += vqa_r * 0.3  # 长度合理奖励
        
        # ========== Caption任务：优化奖励幅度 ==========
        elif "[caption]" in prompt.lower():
            # 使用 BLEU-1, BLEU-2, BLEU-4 的平均值
            bleu_score = compute_bleu_avg(answer_text, gt)
            # 调整幅度，使其与VQA接近（0-3分范围）
            r += caption_r * bleu_score * 5.0  # 从*3改为*5，使最高分接近VQA
            
            # 额外奖励：长度合理性
            if len(answer_text) > 10:  # 至少生成了一些内容
                len_ratio = len(answer_text) / max(len(gt), 1)
                if 0.3 <= len_ratio <= 3.0:  # Caption允许更大的长度差异
                    r += caption_r * 0.3
        
        # ========== Refer任务：分阶段奖励（方案2） ==========
        elif "[refer]" in prompt.lower():
            # Level 1: 格式奖励（鼓励生成bbox相关格式）
            if '[' in resp and ']' in resp:
                r += grounding_r * 0.5  # 有方括号就给奖励
            
            # Level 2: 数字提取奖励（鼓励生成数字）
            numbers = re.findall(r'\d+', resp)
            if len(numbers) >= 4:
                r += grounding_r * 0.7  # 至少有4个数字
            
            # Level 3: 完整格式奖励 + IoU
            try:
                coords = re.findall(r"\[(\d+),(\d+),(\d+),(\d+)\]", resp)
                if coords:
                    r += grounding_r * 1.0  # 格式完全正确
                    
                    # Level 4: IoU奖励（最高）
                    if gt is not None:
                        x1, y1, x2, y2 = map(int, coords[0])
                        iou = compute_iou((x1, y1, x2, y2), gt)
                        r += grounding_r * iou * 3.5  # IoU越高，奖励越高（最高可达3.5）
                        
                        # 额外奖励：如果IoU > 0但不完美，也给予鼓励
                        if 0 < iou < 0.5:
                            r += grounding_r * 0.3  # 方向对了，继续努力
            except Exception as e:
                # 格式接近但解析失败，也给一点奖励
                if '[' in resp and ']' in resp and len(numbers) >= 2:
                    r += grounding_r * 0.2  # 至少在尝试
        # 格式惩罚
        # if resp.count("<think>") != resp.count("</think>"):
        #     r += penalty_r
        # if resp.count("<answer>") != resp.count("</answer>"):
        #     r += penalty_r

        
    return r
def compute_format_reward(resp: str,
                          perfect_r: float = 1.0,
                          penalty_unit: float = 1.2) -> float:
    """格式化奖励：恰好一对 think/answer 得 perfect_r，其他按混乱度扣分。"""
    cnt_to = resp.count("<think>")
    cnt_tc = resp.count("</think>")
    cnt_ao = resp.count("<answer>")
    cnt_ac = resp.count("</answer>")

    # 完美情况
    if cnt_to == 1 and cnt_tc == 1 and cnt_ao == 1 and cnt_ac == 1:
        return perfect_r

    # 拆分混乱度
    mismatch_tags   = abs(cnt_to - cnt_tc) + abs(cnt_ao - cnt_ac)
    pairs_think     = min(cnt_to, cnt_tc)
    pairs_ans       = min(cnt_ao, cnt_ac)
    extra_pairs     = max(0, pairs_think - 1) + max(0, pairs_ans - 1)
    missing_pairs   = ((1 - pairs_think) if pairs_think < 1 else 0) \
                    + ((1 - pairs_ans)   if pairs_ans   < 1 else 0)
    confusion       = mismatch_tags + extra_pairs + missing_pairs

    return perfect_r - confusion * penalty_unit


def compute_iou(boxA, boxB):
    # 计算两个框的 IoU
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0
