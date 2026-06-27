import os
import json
import argparse
import re
import base64
import requests
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Dual Agent Vanilla (VLM Caption -> LLM Direct) on ScienceQA")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--test_split', type=str, default='test')
    # 使用您验证过的模型名
    parser.add_argument('--vlm_model', type=str, default='qwen2.5vl:32b')
    parser.add_argument('--llm_model', type=str, default='qwen2.5:32b')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434')
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--label', type=str, default='dual_agent_vanilla')
    parser.add_argument('--save_every', type=int, default=50)
    return parser.parse_args()

def encode_image(image_path):
    if not os.path.exists(image_path): return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

class DualAgentVanilla:
    def __init__(self, args):
        self.host = args.ollama_host
        self.vlm = args.vlm_model
        self.llm = args.llm_model
        
        # === 启动前自检 ===
        print(f"🔍 Checking models on {self.host}...")
        self._check_model_exists(self.vlm)
        self._check_model_exists(self.llm)
        print("✅ Models verified! Running Dual Agent (Vanilla Mode).")

    def _check_model_exists(self, model_name):
        try:
            res = requests.post(f"{self.host}/api/show", json={"name": model_name})
            if res.status_code != 200:
                print(f"\n❌ FATAL ERROR: Model '{model_name}' not found!")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Connection Error: {e}")
            sys.exit(1)

    def get_caption(self, image_path):
        """第一棒：VLM 生成客观描述"""
        if not image_path: return ""
        
        img_b64 = encode_image(image_path)
        if not img_b64: return ""

        prompt = (
            "Describe this image in detail for a science question. "
            "Focus on factual details, text, and data shown. Do not explain the science."
        )
        
        payload = {
            "model": self.vlm,
            "prompt": prompt,
            "stream": False,
            "images": [img_b64],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"VLM Error on {image_path}: {e}")
            return ""

    def solve(self, question, options, caption, context, lecture, hint):
        """第二棒：LLM 直接回答 (无推理)"""
        opt_lines = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        # 构造上下文
        context_block = ""
        if caption: context_block += f"Image Description: {caption}\n"
        if context: context_block += f"Context: {context}\n"
        if lecture: context_block += f"Background Knowledge: {lecture}\n"
        if hint:    context_block += f"Hint: {hint}\n"
        
        # === Vanilla Prompt (禁止 CoT) ===
        prompt = (
            f"{context_block}\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            "Instruction: Based on the information provided, select the correct answer option.\n"
            "Do not explain your reasoning. Do not think step-by-step.\n"
            "Output ONLY the option letter directly (e.g., '(A)').\n"
            "Answer:"
        )

        payload = {
            "model": self.llm,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
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
    m = re.search(r"\(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    m = re.search(r"answer is \(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches: return options.index(matches[-1])
    return None

def main():
    args = parse_args()
    print(f"==== Running Dual Agent Vanilla ====")
    print(f"Vision Model: {args.vlm_model}")
    print(f"Reasoning Model: {args.llm_model}")
    
    if args.output_root is None:
        args.output_root = os.path.join(os.path.dirname(args.data_root), "results_dual")
    os.makedirs(args.output_root, exist_ok=True)
    result_file = os.path.join(args.output_root, f"{args.label}.json")
    
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

    agent = DualAgentVanilla(args)
    correct = 0
    results = {}
    options = ["A", "B", "C", "D", "E"]
    
    print(f"🚀 Starting Dual Agent Vanilla inference on {len(qids)} questions...")
    
    for i, qid in enumerate(tqdm(qids)):
        prob = problems.get(qid)
        
        # 1. 第一棒：看图
        full_image_path = None
        caption = ""
        if prob.get('image'):
            p1 = os.path.join(args.image_root, args.test_split, prob['image'])
            p2 = os.path.join(args.image_root, prob['image'])
            if os.path.exists(p1): full_image_path = p1
            elif os.path.exists(p2): full_image_path = p2
            
            if full_image_path:
                caption = agent.get_caption(full_image_path)
            
        # 2. 第二棒：直接回答 (无 CoT)
        raw_output = agent.solve(
            prob['question'], 
            prob['choices'], 
            caption,
            prob.get('context'),
            prob.get('lecture'),
            prob.get('hint')
        )
        
        gt = prob.get('answer')
        pred = extract_answer(raw_output, options)
        is_corr = (pred == gt)
        if is_corr: correct += 1
        
        results[qid] = {
            "caption": caption,
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