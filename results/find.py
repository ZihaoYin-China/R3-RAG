import json

def find_best_case_studies(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {log_path}")
        return []

    # 自适应解析 JSON 结构
    entries = []
    if isinstance(logs, dict):
        # 情况 1：字典的值是具体的实验记录字典 (如 {"id_1": {...}, "id_2": {...}})
        if all(isinstance(v, dict) for v in logs.values()):
            entries = list(logs.values())
        # 情况 2：真实的列表被包在某个键下面 (如 {"data": [...]})
        else:
            for key, value in logs.items():
                if isinstance(value, list):
                    entries = value
                    break
            if not entries:
                print("Error: 无法在字典中找到包含实验记录的列表。")
                return []
    elif isinstance(logs, list):
        entries = logs
    else:
        print("Error: 无法识别的 JSON 结构。")
        return []

    candidates = []
    # 避开地图和生物相关的关键词
    exclude_keywords = [
        'map', 'state', 'country', 'city', 'ocean', 
        'cell', 'animal', 'plant', 'body', 'biology', 
        'geography', 'tissue', 'organ', 'species'
    ]

    for entry in entries:
        # 确保 entry 确实是个字典
        if not isinstance(entry, dict):
            continue
            
        question = entry.get('question', '').lower()
        image = entry.get('image', None)
        reflection_steps = entry.get('reflection_steps', 0)
        ground_truth = str(entry.get('ground_truth', '')).strip()
        final_pred = str(entry.get('final_prediction', '')).strip()

        is_correct = (final_pred == ground_truth)

        # 1. 必须是多模态题目（有图）
        if not image:
            continue

        # 2. 排除生物和地图题
        if any(kw in question for kw in exclude_keywords):
            continue

        # 3. 筛选触发了反思闭环且最终回答正确的题目
        if reflection_steps > 0 and is_correct:
            candidates.append(entry)

    # 按照反思步数排序，反思步数越多，路由日志越丰富，越适合做 Case Study
    candidates.sort(key=lambda x: x.get('reflection_steps', 0), reverse=True)
    return candidates[:5]

if __name__ == "__main__":
    # 替换为你实际的绝对路径
    log_file = "/Yin_zi_hao/code/HMRAG-Self-Reflection/results/hybrid_local_reflection_v66_test.json"
    top_cases = find_best_case_studies(log_file)
    
    if not top_cases:
        print("没有找到符合条件的候选案例。")
    else:
        for i, case in enumerate(top_cases):
            print(f"--- Candidate {i+1} ---")
            print(f"Question: {case.get('question')}")
            print(f"Reflection Steps: {case.get('reflection_steps')}")
            print(f"Alpha/Fusion Log: {case.get('fusion_log', 'N/A')}")
            print(f"Diagnosis/Route Log: {case.get('reflection_log', 'N/A')}\n")