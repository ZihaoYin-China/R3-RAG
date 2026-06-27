import json
import argparse
import os
import re
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Analyze ScienceQA Results")
    parser.add_argument('--data_root', type=str, required=True, help="Path to ScienceQA data root")
    parser.add_argument('--result_file', type=str, required=True, help="Path to the result .json file")
    args = parser.parse_args()

    # 1. 加载问题元数据
    p_path = os.path.join(args.data_root, "problems.json")
    if not os.path.exists(p_path):
        p_path = os.path.join(args.data_root, "problems_test.json")
    
    if not os.path.exists(p_path):
        print(f"Error: Cannot find problems file in {args.data_root}")
        return

    print(f"Loading problems from: {p_path}")
    problems = json.load(open(p_path, "r", encoding="utf-8"))

    # 2. 加载结果文件
    print(f"Loading results from: {args.result_file}")
    if not os.path.exists(args.result_file):
        print(f"Error: Result file not found: {args.result_file}")
        return
    results = json.load(open(args.result_file, "r", encoding="utf-8"))

    # 3. 初始化统计
    stats = {
        "subject": defaultdict(lambda: {'total': 0, 'correct': 0}),
        "grade":   defaultdict(lambda: {'total': 0, 'correct': 0}),
        "context": defaultdict(lambda: {'total': 0, 'correct': 0}), 
        "overall": {'total': 0, 'correct': 0}
    }

    # 4. 开始统计
    for qid, res in results.items():
        if qid not in problems:
            continue
        
        prob = problems[qid]
        
        # ==== 获取基础标签 (Subject) ====
        subj = prob.get('subject', 'Unknown').capitalize()
        
        # ==== Grade 分组逻辑 (1-6 vs 7-12) ====
        raw_grade = str(prob.get('grade', ''))
        grade_bucket = "Unknown"
        
        # 提取字符串中的数字 (例如 "grade1" -> 1)
        g_match = re.search(r'\d+', raw_grade)
        if g_match:
            g_num = int(g_match.group())
            if 1 <= g_num <= 6:
                grade_bucket = "Grades 1-6"
            elif 7 <= g_num <= 12:
                grade_bucket = "Grades 7-12"
        
        # ==== Context 分类逻辑 (IMG, TXT, NO) ====
        image_val = prob.get('image')
        has_image = (image_val and str(image_val).strip() != "" and str(image_val).lower() != "null")
        
        hint_val = prob.get('hint')
        has_hint = (hint_val and str(hint_val).strip() != "" and str(hint_val).lower() != "null")

        if has_image:
            ctx = "IMG"
        elif has_hint:
            ctx = "TXT"
        else:
            ctx = "NO"
        
        # ==== 判断正误 ====
        if 'is_correct' in res:
            is_corr = res['is_correct']
        else:
            pred = res.get('pred_idx')
            gt = res.get('gt_idx')
            is_corr = (pred == gt) and (pred is not None)

        # ==== 累加统计 ====
        # 注意这里用 grade_bucket 而不是 raw grade
        for key, val in [('subject', subj), ('grade', grade_bucket), ('context', ctx)]:
            stats[key][val]['total'] += 1
            if is_corr:
                stats[key][val]['correct'] += 1
        
        stats['overall']['total'] += 1
        if is_corr:
            stats['overall']['correct'] += 1

    # 5. 打印表格函数
    def print_table(col_name, data_dict):
        rows = []
        for name, d in data_dict.items():
            acc = (d['correct'] / d['total']) * 100 if d['total'] > 0 else 0
            rows.append({
                col_name: name,
                "Total": d['total'],
                "Correct": d['correct'],
                "Accuracy": f"{acc:.2f}%"
            })
        
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"\n[{col_name.upper()} BREAKDOWN] - No Data")
            return

        # ==== 特殊排序逻辑 ====
        if col_name == "Context":
            sorter = ["IMG", "TXT", "NO"]
            df[col_name] = pd.Categorical(df[col_name], categories=sorter, ordered=True)
            df = df.sort_values(col_name)
        elif col_name == "Grade":
            # 强制排序: 先 1-6，后 7-12
            sorter = ["Grades 1-6", "Grades 7-12", "Unknown"]
            # 只保留数据中存在的类别，防止报错
            existing_sorter = [s for s in sorter if s in df[col_name].unique()]
            df[col_name] = pd.Categorical(df[col_name], categories=existing_sorter, ordered=True)
            df = df.sort_values(col_name)
        else:
            df = df.sort_values(by=col_name)

        print(f"\n[{col_name.upper()} BREAKDOWN]")
        print(df.to_string(index=False))

    # 6. 输出结果
    print("\n" + "="*60)
    if stats['overall']['total'] > 0:
        overall_acc = (stats['overall']['correct']/stats['overall']['total'])*100
        print(f"🏆 OVERALL ACCURACY: {stats['overall']['correct']}/{stats['overall']['total']} = {overall_acc:.2f}%")
    else:
        print("No matching problems found.")
    print("="*60)

    print_table("Context", stats['context'])
    print_table("Subject", stats['subject'])
    print_table("Grade", stats['grade'])

if __name__ == "__main__":
    main()