import os
import json
import argparse
import re
import requests
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pure Text LLM with CoT on ScienceQA")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--llm_model', type=str, default='qwen2.5:32b')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434')
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--label', type=str, default='text_only_cot')
    parser.add_argument('--save_every', type=int, default=50)
    return parser.parse_args()

class TextCoTAgent:
    def __init__(self, args):
        self.host = args.ollama_host
        self.model = args.llm_model
        
        # 自检模型是否存在
        print(f"🔍 Checking model {self.model} on {self.host}...")
        try:
            res = requests.post(f"{self.host}/api/show", json={"name": self.model})
            if res.status_code != 200:
                print(f"❌ FATAL ERROR: Model '{self.model}' not found!")
                sys.exit(1)
            print("✅ Model verified!")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit(1)

    def solve(self, question, options, context, lecture, hint):
        """
        核心 CoT 推理函数
        """
        # 构造选项字符串
        opt_lines = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        # 构造背景知识 (ScienceQA 自带的 context, lecture, hint)
        # 注意：这里没有 Image Caption，因为是纯盲测
        context_block = ""
        if context: context_block += f"Context: {context}\n"
        if lecture: context_block += f"Background Knowledge: {lecture}\n"
        if hint:    context_block += f"Hint: {hint}\n"
        
        # === Chain-of-Thought Prompt ===
        prompt = (
            f"{context_block}\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            "Instruction: You are a scientific expert. Answer the multiple-choice question.\n"
            "1. Think step-by-step to analyze the question and options.\n"
            "2. If the question refers to an image, try to infer the answer from the text context or use common sense (since the image is missing).\n"
            "3. Conclude with the final answer option.\n\n"
            "Format your response exactly as:\n"
            "Reasoning: [Your step-by-step logic]\n"
            "Answer: The answer is (X)"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1 # CoT 需要一点确定性
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
    # 优先匹配 "The answer is (A)"
    m = re.search(r"answer is \(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    
    # 其次匹配 "Answer: A"
    m = re.search(r"Answer:\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())

    # 兜底：找最后一个选项字母
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches: return options.index(matches[-1])
    return None

def main():
    args = parse_args()
    print(f"==== Running Pure Text CoT Baseline ====")
    print(f"Model: {args.llm_model}")
    
    if args.output_root is None:
        args.output_root = os.path.join(os.path.dirname(args.data_root), "results_text_cot")
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

    agent = TextCoTAgent(args)
    correct = 0
    results = {}
    options = ["A", "B", "C", "D", "E"]
    
    print(f"🚀 Starting CoT inference on {len(qids)} questions...")
    
    for i, qid in enumerate(tqdm(qids)):
        prob = problems.get(qid)
        
        # 提取 ScienceQA 特有的文本字段
        question = prob.get('question')
        choices = prob.get('choices')
        hint = prob.get('hint')       # 关键：ScienceQA 的 Hint 包含很多信息
        lecture = prob.get('lecture') # 关键：Lecture 是背景知识
        # image = prob.get('image')   # 忽略图片，因为是 Text-Only Baseline
        
        # 调用 LLM
        raw_output = agent.solve(question, choices, context=None, lecture=lecture, hint=hint)
        
        gt = prob.get('answer')
        pred = extract_answer(raw_output, options)
        is_corr = (pred == gt)
        if is_corr: correct += 1
        
        results[qid] = {
            "pred_raw": raw_output,
            "pred_idx": pred,
            "gt_idx": gt,
            "is_correct": is_corr,
            "has_image": bool(prob.get('image')) # 记录一下这题原来有没有图，方便后期分析
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