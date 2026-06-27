import os
import json
import argparse
import re
import requests
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pure Text LLM (Vanilla/No-CoT) on ScienceQA")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--llm_model', type=str, default='qwen2.5:32b')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434')
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--label', type=str, default='text_only_vanilla')
    parser.add_argument('--save_every', type=int, default=50)
    return parser.parse_args()

class TextVanillaAgent:
    def __init__(self, args):
        self.host = args.ollama_host
        self.model = args.llm_model
        
        # 自检模型
        print(f"🔍 Checking model {self.model} on {self.host}...")
        try:
            res = requests.post(f"{self.host}/api/show", json={"name": self.model})
            if res.status_code != 200:
                print(f"❌ FATAL ERROR: Model '{self.model}' not found!")
                sys.exit(1)
            print("✅ Model verified! Running Vanilla (Direct Answer) Mode.")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit(1)

    def solve(self, question, options, context, lecture, hint):
        # 构造选项
        opt_lines = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        # 构造纯文本背景
        context_block = ""
        if context: context_block += f"Context: {context}\n"
        if lecture: context_block += f"Background Knowledge: {lecture}\n"
        if hint:    context_block += f"Hint: {hint}\n"
        
        # === Vanilla Prompt (无 CoT，直接问答案) ===
        prompt = (
            f"{context_block}\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            "Instruction: Select the correct answer from the options above.\n"
            "Do not explain. Do not show reasoning. Output ONLY the option letter directly (e.g., '(A)').\n"
            "Answer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1 # 越低越好，只需要答案
        }
        
        try:
            response = requests.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

def extract_answer(text, options):
    text = str(text).strip()
    # 极简提取：直接找 (A) 或者 A
    # 1. 匹配 "(A)"
    m = re.search(r"\(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    
    # 2. 匹配 "The answer is A"
    m = re.search(r"answer is \(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())

    # 3. 兜底
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches: return options.index(matches[-1])
    return None

def main():
    args = parse_args()
    print(f"==== Running Pure Text Vanilla (No CoT) ====")
    print(f"Model: {args.llm_model}")
    
    if args.output_root is None:
        args.output_root = os.path.join(os.path.dirname(args.data_root), "results_text_vanilla")
    os.makedirs(args.output_root, exist_ok=True)
    result_file = os.path.join(args.output_root, f"{args.label}.json")
    
    # Load Data
    p_path = os.path.join(args.data_root, "problems.json")
    if not os.path.exists(p_path):
        p_path = os.path.join(args.data_root, f"problems_{args.test_split}.json")
    problems = json.load(open(p_path, "r", encoding="utf-8"))
    
    pid_path = os.path.join(args.data_root, "pid_splits.json")
    if not os.path.exists(pid_path):
         pid_path = os.path.join(os.path.dirname(args.data_root), "pid_splits.json")

    if os.path.exists(pid_path):
        qids = json.load(open(pid_path, "r", encoding="utf-8")).get(args.test_split, [])
    else:
        qids = list(problems.keys())

    if args.test_number > 0: qids = qids[:args.test_number]

    agent = TextVanillaAgent(args)
    correct = 0
    results = {}
    options = ["A", "B", "C", "D", "E"]
    
    print(f"🚀 Starting Vanilla inference on {len(qids)} questions...")
    
    for i, qid in enumerate(tqdm(qids)):
        prob = problems.get(qid)
        
        # 提取字段 (忽略 image)
        question = prob.get('question')
        choices = prob.get('choices')
        hint = prob.get('hint')
        lecture = prob.get('lecture')
        
        # 调用
        raw_output = agent.solve(question, choices, prob.get('context'), lecture, hint)
        
        gt = prob.get('answer')
        pred = extract_answer(raw_output, options)
        is_corr = (pred == gt)
        if is_corr: correct += 1
        
        results[qid] = {
            "pred_raw": raw_output,
            "pred_idx": pred,
            "gt_idx": gt,
            "is_correct": is_corr,
            "has_image": bool(prob.get('image'))
        }
        
        if (i+1) % args.save_every == 0:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Final Accuracy: {correct}/{len(qids)} = {correct/len(qids):.4f}")
    print(f"Saved to: {result_file}")

if __name__ == "__main__":
    main()