
import json
from openai import OpenAI
import os
from tqdm import tqdm
import subprocess
import numpy as np
# from clair import clair
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

##### vqa metric
def compute_vqa_metric(result_path):
    import json
    f = open(result_path, 'r')
    results = [json.loads(line) for line in f.readlines()]
    f.close()
    correct = sum([int(result['correct']) for result in results if result['correct'] in ['1', '0']])
    
    print(f"VQA Accuracy: {correct}/{len(results)} = {correct/len(results):.4f}")
    # import inflect

    # # Create an engine instance
    # convert = inflect.engine()

    # data_path = result_path

    # correct = 0
    # total = 0

    # all_types = ['object category', 'object existence', 'object quantity', 'object color', 'object shape', 'object size', 'object position', 'object direction', 'image', 'scene type', 'reasoning', 'rural or urban']

    # print('number of question types:', len(all_types))

    # all_numbers = [convert.number_to_words(x) for x in range(100)]

    # # create a dict with types as key and value to zero
    # correct_per_type = {k: 0 for k in all_types}
    # total_per_type = {k: 0 for k in all_types}
    # invalid_type = 0
    # skip_qas = 0
    # with open(data_path, 'r') as file:
    #     for line in file:
    #         # Convert JSON string to Python dictionary
    #         item = json.loads(line.strip())
    #         img_id = item['image_id']

    #         gt_ans = item['ground_truth'].lower()
    #         pred_ans = item['predicted'].lower()
            
    #         q_type = item['type'].lower()
    #         if q_type == 'image': q_type = 'scene type'
    #         if q_type == 'rural or urban': q_type = 'scene type'

    #         if q_type in all_types:
    #             total_per_type[q_type] += 1
    #         else:
    #             print('unknown type:', q_type)
    #             invalid_type += 1

    #         if item['correct'] == '1':
    #             correct += 1
    #             if q_type in all_types:
    #                 correct_per_type[q_type] += 1
            
    #         total += 1

    # print('number of questions:', total, 'invalid_type:', invalid_type, 'valid', sum(total_per_type.values()))
    # print('Overall acc:', correct/total * 100)
    # # divide by the number of questions of that type
    # print('##############')
    # acc_list = []
    # for k in all_types:
    #     if total_per_type[k] == 0:
    #         continue
    #     print(f'{k} accuracy: {correct_per_type[k]/total_per_type[k] * 100}, out of {total_per_type[k]}')
    #     acc = correct_per_type[k]/total_per_type[k] * 100
    #     acc_list.append(acc)
    # print('########################## ------------------------------ ##############################')

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
    for ii, qa in enumerate(tqdm(qa_list, disable=True)):
        # print(qa["question_id"])
        question = qa['Text']
        ground_truth = qa['Ground truth'].lower()
        predicted = qa['answer'].lower().strip()
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
            'question_id': qa['Question id'],
            'image_id': qa['Image'],
            "type": qa['Task'],
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
    for ii, qa in enumerate(tqdm(qa_list, disable=True)):
        question = qa['Text']
        ground_truth = qa['Ground truth']
        predicted = qa['answer']
        # clair_score = clair([predicted], [ground_truth], model='gpt-4o')
        # print(clair_score)

        result = {
            'image_id': qa['Image'],
            'answer':qa['answer'],
            'question_id': qa['Question id'],
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
    if not os.path.exists('./valid/pred_cap.txt'):
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
                    print('empty', img_id, pred_ans)
                    continue

                gt_answers.append([img_id, gt_ans])
                pred_answers.append([img_id, pred_ans])

        print('number of captions', len(gt_answers))
        np.savetxt('./valid/pred_cap.txt',  pred_answers, fmt='%s', delimiter='\t')
        np.savetxt('./valid/gt_cap.txt',  gt_answers, fmt='%s', delimiter='\t')
        cap_len = [len(ans[1].split()) for ans in pred_answers]
        print('avg len', np.mean(cap_len), np.std(cap_len))
    # Create json references quietly
    _ = subprocess.run(
        ["python", "./valid/create_json_references.py", "-i", "./valid/gt_cap.txt", "-o", "./valid/gt_cap.json"],
        capture_output=True,
        text=True
    )
    # Run evaluation and capture output, then print only final metrics
    eval_run = subprocess.run(
        ["python", "-u", "./valid/run_evaluations.py", "-i", "./valid/pred_cap.txt", "-r", "./valid/gt_cap.json"],
        capture_output=True,
        text=True
    )
    lines = eval_run.stdout.splitlines()
    # Prefer metrics after the summary marker if present
    scores_start_idx = None
    for idx, ln in enumerate(lines):
        if "Scores:" in ln:
            scores_start_idx = idx
    metrics_lines = []
    source_iter = lines[scores_start_idx + 1:] if scores_start_idx is not None else lines
    for line in source_iter:
        line_stripped = line.strip()
        if (
            line_stripped.startswith("Bleu_") or
            line_stripped.startswith("METEOR:") or
            line_stripped.startswith("ROUGE_L:") or
            line_stripped.startswith("CIDEr:")
        ):
            metrics_lines.append(line_stripped)
    # If still empty, fall back to last occurrence per metric key across all lines
    if not metrics_lines:
        last_by_key = {}
        for line in lines:
            ls = line.strip()
            for key in ("Bleu_1:", "Bleu_2:", "Bleu_3:", "Bleu_4:", "METEOR:", "ROUGE_L:", "CIDEr:"):
                if ls.startswith(key):
                    last_by_key[key] = ls
        metrics_lines = [last_by_key[k] for k in ("Bleu_1:", "Bleu_2:", "Bleu_3:", "Bleu_4:", "METEOR:", "ROUGE_L:", "CIDEr:") if k in last_by_key]
    for m in metrics_lines:
        print(m)
    # Compute and print Avg_Bleu = mean(Bleu_1, Bleu_2, Bleu_4)
    bleu_vals = {}
    for ln in metrics_lines:
        ls = ln.strip()
        for key in ("Bleu_1:", "Bleu_2:", "Bleu_4:"):
            if ls.startswith(key):
                try:
                    bleu_vals[key] = float(ls.split(":", 1)[1].strip())
                except Exception:
                    pass
    if all(k in bleu_vals for k in ("Bleu_1:", "Bleu_2:", "Bleu_4:")):
        avg_bleu = (bleu_vals["Bleu_1:"] + bleu_vals["Bleu_2:"] + bleu_vals["Bleu_4:"]) / 3.0
        print(f"Avg_Bleu(Bleu_1, Bleu_2, Bleu_4): {avg_bleu:.3f}")
    temp_files = [
        './valid/pred_cap.txt',
        './valid/gt_cap.txt',
        './valid/gt_cap.json',
        './valid/pred_cap.txt.json'
    ]
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                
            except Exception as e:
                print(f"[WARN] Failed to remove {f}: {e}")




import argparse
import json 
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=str, default=None)
    parser.add_argument("--result-path", type=str, default=None)
    parser.add_argument("--caption", action="store_true", help="Enable captioning")
    parser.add_argument("--mcq", action="store_true", help="Enable VQA")

    args = parser.parse_args()
    # Suppress verbose argument echo; only output evaluation results
    if args.mcq:
        get_vqa_result(args.answer_path,args.result_path)
        compute_vqa_metric(args.result_path)
    if args.caption:
        get_cap_result(args.answer_path,args.result_path)
        compute_cap_metric(args.result_path)
