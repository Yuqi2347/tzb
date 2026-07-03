
import openai
import json
from openai import OpenAI
import os
from tqdm import tqdm
import subprocess
import numpy as np
from clair import clair
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

##### vqa metric
def compute_vqa_metric(result_path):
    import json
    f = open(result_path, 'r')
    results = [json.loads(line) for line in f.readlines()]
    f.close()
    correct = sum([int(result['correct']) for result in results if result['correct'] in ['1', '0']])
    
    print('#'*25 + 'Compute VQA Metric' + '#'*25)
    print(f"Acc: {correct}/{len(results)}:", correct/len(results))
    print('#'*68)
    import inflect

    # Create an engine instance
    convert = inflect.engine()

    data_path = result_path

    correct = 0
    total = 0

    all_types = ['object category', 'object existence', 'object quantity', 'object color', 'object shape', 'object size', 'object position', 'object direction', 'image', 'scene type', 'reasoning', 'rural or urban']

    print('number of question types:', len(all_types))

    all_numbers = [convert.number_to_words(x) for x in range(100)]

    # create a dict with types as key and value to zero
    correct_per_type = {k: 0 for k in all_types}
    total_per_type = {k: 0 for k in all_types}
    invalid_type = 0
    skip_qas = 0
    with open(data_path, 'r') as file:
        for line in file:
            # Convert JSON string to Python dictionary
            item = json.loads(line.strip())
            img_id = item['image_id']

            gt_ans = item['ground_truth'].lower()
            pred_ans = item['predicted'].lower()
            
            q_type = item['type'].lower()
            if q_type == 'image': q_type = 'scene type'
            if q_type == 'rural or urban': q_type = 'scene type'

            if q_type in all_types:
                total_per_type[q_type] += 1
            else:
                print('unknown type:', q_type)
                invalid_type += 1

            if item['correct'] == '1':
                correct += 1
                if q_type in all_types:
                    correct_per_type[q_type] += 1
            
            total += 1

    print('number of questions:', total, 'invalid_type:', invalid_type, 'valid', sum(total_per_type.values()))
    print('Overall acc:', correct/total * 100)
    # divide by the number of questions of that type
    print('#'*68)
    acc_list = []
    for k in all_types:
        if total_per_type[k] == 0:
            continue
        print(f'{k} accuracy: {correct_per_type[k]/total_per_type[k] * 100}, out of {total_per_type[k]}')
        acc = correct_per_type[k]/total_per_type[k] * 100
        acc_list.append(acc)
    print('#'*68)

def check_match_with_gpt(question, ground_truth, predicted):
        # Construct the prompt for GPT-4
        prompt = f"Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {predicted}\nDoes the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-1106",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                    ]
                }
            ],
            max_tokens=100,
        )

        # answer = response.choices[0].text.strip()
        answer =  response.choices[0].message.content
        # print(answer)
        return answer

def get_vqa_result(answer_Path, result_Path):
    qa_list = [json.loads(line) for line in open(answer_Path,'r').readlines()]

    # Iterate over the list and check matches
    results = []
    f = open(result_Path, 'w') 
    for ii, qa in enumerate(tqdm(qa_list)):
        # print(qa["question_id"])
        question = qa['question']
        ground_truth = qa['ground_truth'].lower()
        predicted = qa['answer'].lower()
        if ground_truth in predicted:
            match_result = '1'
        elif ground_truth in ['yes', 'no'] + list(map(str, range(100))):
            match_result = '1' if ground_truth == predicted else '0'
        elif 'correct' not in qa or qa['correct'] not in ['1', '0']:
            match_result='0'
            # match_result = check_match_with_gpt(question, ground_truth, predicted)
            # pass
        else:
            match_result = qa['correct']
        # print(match_result)
        result = {
            'question_id': qa['question_id'],
            'image_id': qa['image_id'],
            "type": qa['type'],
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": match_result,
        }
        results.append(result)

        f.write(json.dumps(result)+'\n')
        f.flush()

    f.close()

def get_cap_result(answer_path,result_path):
    qa_list = [json.loads(line) for line in open(answer_path,'r').readlines()]
    # Iterate over the list and check matches
    results = []
    import shutil
    if os.path.exists(result_path):
        shutil.copy(result_path, result_path + '.bak')
    f = open(result_path, 'w') 
    for ii, qa in enumerate(tqdm(qa_list)):
        question = qa['question']
        ground_truth = qa['ground_truth']
        predicted = qa['answer']
        # clair_score = clair([predicted], [ground_truth], model='gpt-4o')
        # print(clair_score)

        result = {
            'image_id': qa['image_id'],
            'answer':qa['answer'],
            'question_id': qa['question_id'],
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            # "clair": clair_score[0],
            # "clair_reason": clair_score[1],
        }
        results.append(result)

        f.write(json.dumps(result)+'\n')
        f.flush()

    f.close()

def compute_cap_metric(result_path):
    data_path = result_path
    if not os.path.exists('./vrs/pred_cap.txt'):
        # print("*"*50)
        gt_answers= []
        pred_answers = []
        with open(data_path, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                img_id = item['image_id']
                gt_ans = item['ground_truth'].strip().replace('\n', ' ')
                pred_ans = item['answer'].strip().replace('\n', ' ')

                if img_id is None or img_id=='\n' or pred_ans is None or pred_ans=='\n':
                    # print('empty', img_id, pred_ans)
                    continue

                gt_answers.append([img_id, gt_ans])
                pred_answers.append([img_id, pred_ans])

        print('number of captions', len(gt_answers))
        np.savetxt('./vrs/pred_cap.txt',  pred_answers, fmt='%s', delimiter='\t')
        np.savetxt('./vrs/gt_cap.txt',  gt_answers, fmt='%s', delimiter='\t')
        cap_len = [len(ans[1].split()) for ans in pred_answers]
        print('avg len', np.mean(cap_len), np.std(cap_len))
    result = subprocess.run(
    ["python", "./vrs/create_json_references.py", "-i", "./vrs/gt_cap.txt", "-o", "./vrs/gt_cap.json"],
    capture_output=True,
    text=True
    )
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    print("********************************************************************************")
    process = subprocess.Popen(
    ["python", "-u", "./vrs/run_evaluations.py", "-i", "./vrs/pred_cap.txt", "-r", "./vrs/gt_cap.json"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1  # 行缓冲
)

    print("[INFO] Real-time logs from run_evaluations.py:")
    for line in process.stdout:
        print(line, end="")  # 实时输出，不加多余换行
    process.wait()
    # all_scores=[]
    # print("[INFO] calculate the clair score:")
    # with open(data_path, 'r') as file:
    #     for line in file:
    #         pass
    #         # all_scores.append(line["clair"])
    # if all_scores:
    #     average_score = sum(all_scores) / len(all_scores)
    #     print(f"\nAverage CLAIR score over {len(all_scores)} samples: {average_score:.4f}")
    # else:
    #     print("No valid scores computed.")
    temp_files = [
        './vrs/pred_cap.txt',
        './vrs/gt_cap.txt',
        './vrs/gt_cap.json',
        './vrs/pred_cap.txt.json'
    ]
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                
            except Exception as e:
                print(f"[WARN] Failed to remove {f}: {e}")


#########################################################################################                 refer
import re
import json
import numpy as np

def bbox_overlaps_hbb(bboxes1, bboxes2, eps=1e-6):
    """
    计算普通矩形框 IoU
    bboxes1, bboxes2: shape=(N, 4)  [x1, y1, x2, y2]
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32), None, None

    lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])   # 左上角
    rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])   # 右下角
    wh = np.clip(rb - lt, 0, np.inf)                        # 宽高
    inter = wh[..., 0] * wh[..., 1]                         # 交集面积

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2 - inter

    union = np.clip(union, eps, np.inf)
    iou = inter / union
    return iou, inter, union


def parse_answer_list(ans_str, img_id=None):
    """
    解析 GT 或预测框
    - 尖括号格式 {<x1><y1><x2><y2>} → 除以100归一化并裁剪到[0,1]
    - 纯数组 [x1, y1, x2, y2] → 直接裁剪到[0,1]
    """
    try:
        if isinstance(ans_str, str):
            ans_str = ans_str.strip()
            if ans_str.startswith("{<"):  
                # GT 格式
                numbers = re.findall(r"<\s*([-+]?\d*\.?\d+)\s*>", ans_str)
                if len(numbers) != 4:
                    raise ValueError("GT 数量错误")
                coords = [min(1.0, max(0.0, float(v) / 100.0)) for v in numbers]
            elif ans_str.startswith("["):
                # 预测框格式（字符串形式的列表）
                ans_str = ans_str.replace('[', '').replace(']', '')
                coords = [min(1.0, max(0.0, float(v))) for v in ans_str.split(',')]
            else:
                raise ValueError("不支持的字符串格式")
        elif isinstance(ans_str, list):
            # 直接是 list（预测框）
            coords = [min(1.0, max(0.0, float(v))) for v in ans_str]
        else:
            raise ValueError("Unsupported bbox format.")

        return coords[:4]

    except Exception as e:
        print(f"❌ 解析错误 @ {img_id}: {ans_str} -> {e}")
        return None


import argparse
import json 
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=str, default=None)
    parser.add_argument("--result-path", type=str, default=None)
    parser.add_argument("--cap", action="store_true", help="Enable captioning")
    parser.add_argument("--vqa", action="store_true", help="Enable VQA")
    parser.add_argument("--ref", action="store_true", help="Enable refer")
    args = parser.parse_args()
    print("========= 参数确认 (args) =========")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("====================================")
    if args.vqa:
        get_vqa_result(args.answer_path,args.result_path)
        compute_vqa_metric(args.result_path)
    if args.cap:
        get_cap_result(args.answer_path,args.result_path)
        compute_cap_metric(args.result_path)
    if args.ref:
        # ==== 配置 ====
        data_path = args.answer_path
        thres_list = [0.5, 0.7]
        count = np.zeros(len(thres_list))
        cumI = 0
        cumU = 0
        mean_IoU = 0
        total_count = 0
        valid_count = 0

        # ==== 评估 ====
        with open(data_path, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                img_id = item['image_id']

                gt = parse_answer_list(item.get('ground_truth'), img_id)
                pred = parse_answer_list(item.get('answer'), img_id)

                if gt is None or pred is None or len(gt) != 4 or len(pred) != 4:
                    
                    print(f"⚠️ 无效样本跳过: {img_id}")
                    continue

                total_count += 1

                gt_bbox = np.array([gt])
                pred_bbox = np.array([pred])

                try:
                    iou_score, I, U = bbox_overlaps_hbb(gt_bbox, pred_bbox)
                    iou_score = iou_score[0][0]
                    I = I[0][0]
                    U = U[0][0]
                    mean_IoU += iou_score
                    cumI += I
                    cumU += U
                    for ii, thres in enumerate(thres_list):
                        if iou_score >= thres:
                            count[ii] += 1
                    valid_count += 1
                except Exception as e:
                    print(f"🚫 bbox 计算失败 @ {img_id}: {e}")
                    continue

        # ==== 输出 ====
        print('\n========== 评估结果 ==========')
        print(f'📊 总样本数: {total_count}, 有效样本: {valid_count}')
        for ii, thres in enumerate(thres_list):
            print(f'✔️ Acc@IoU ≥ {thres}: {count[ii] / total_count * 100:.2f}%')
        print(f'📐 mean IoU: {mean_IoU / total_count * 100:.2f}%')
        print(f'🔁 cum IoU: {cumI / cumU * 100:.2f}%')
