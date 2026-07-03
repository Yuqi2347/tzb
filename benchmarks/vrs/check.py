import json

def find_diff_intervals(file_a, file_b):
    # 读取 JSONL (file_a)
    data_a = []
    with open(file_a, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_a.append(json.loads(line))

    # 读取普通 JSON (file_b)
    with open(file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    # 获取 file_a 里的所有 image_id
    ids_a = {item["image_id"] for item in data_a}

    # 找出 file_b 里不在 file_a 中的索引
    diff_indices = [i for i, item in enumerate(data_b) if item["image_id"] not in ids_a]

    # 把连续的索引合并成区间
    intervals = []
    if diff_indices:
        start = diff_indices[0]
        prev = diff_indices[0]
        for idx in diff_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                intervals.append((start, prev))
                start = idx
                prev = idx
        intervals.append((start, prev))  # 最后一段

    return intervals

if __name__ == "__main__":
    file_a = "answer_vrs_cap_latest.json"      # JSON Lines
    file_b = "VRSBench_EVAL_cap_v2.json"      # 普通 JSON
    intervals = find_diff_intervals(file_a, file_b)
    print("不同元素的索引区间:", intervals)
