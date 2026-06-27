import os
import json
import argparse
import re
import base64
import requests
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pure VLM (Vanilla/No-CoT) on ScienceQA")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--test_split', type=str, default='test')
    # 使用您验证过的 32B VL 名字
    parser.add_argument('--model_name', type=str, default='qwen2.5vl:32b')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434')
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--label', type=str, default='pure_vlm_vanilla')
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

class PureVLMVanillaAgent:
    def __init__(self, args):
        self.host = args.ollama_host
        self.model = args.model_name
        
        # === 启动前自检 (A100 专用) ===
        print(f"🔍 Checking VLM model {self.model} on {self.host}...")
        try:
            res = requests.post(f"{self.host}/api/show", json={"name": self.model})
            if res.status_code != 200:
                print(f"❌ FATAL ERROR: Model '{self.model}' not found!")
                sys.exit(1)
            print("✅ Model verified! Running End-to-End Vanilla Mode.")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit(1)

    def solve(self, question, options, image_path, context, lecture, hint):
        # 1. 图片编码
        img_b64 = None
        if image_path:
            img_b64 = encode_image(image_path)
            
        # 2. 文本背景
        context_block = ""
        if context: context_block += f"Context: {context}\n"
        if lecture: context_block += f"Background Knowledge: {lecture}\n"
        if hint:    context_block += f"Hint: {hint}\n"
        
        opt_lines = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        # === Vanilla Prompt (直觉模式) ===
        # 只有简单的指令，禁止推理
        prompt = (
            f"{context_block}\n"
            f"Question: {question}\n"
            f"Options:\n{opt_lines}\n\n"
            "Instruction: Analyze the image (if present) and the text. Select the correct answer directly.\n"
            "Do not explain. Do not think step-by-step. Output ONLY the option letter (e.g., '(A)').\n"
            "Answer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1 # 极低温度，追求确定性
        }
        
        if img_b64:
            payload["images"] = [img_b64]
        
        try:
            response = requests.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Inference Error: {e}")
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
    print(f"==== Running Pure VLM Vanilla (Direct Answer) ====")
    print(f"Model: {args.model_name}")
    
    if args.output_root is None:
        args.output_root = os.path.join(os.path.dirname(args.data_root), "results_pure_vl")
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

    agent = PureVLMVanillaAgent(args)
    correct = 0
    results = {}
    options = ["A", "B", "C", "D", "E"]
    
    print(f"🚀 Starting VLM Vanilla inference on {len(qids)} questions...")
    
    for i, qid in enumerate(tqdm(qids)):
        prob = problems.get(qid)
        
        # 准备图片路径
        full_image_path = None
        if prob.get('image'):
            p1 = os.path.join(args.image_root, args.test_split, prob['image'])
            p2 = os.path.join(args.image_root, prob['image'])
            if os.path.exists(p1): full_image_path = p1
            elif os.path.exists(p2): full_image_path = p2
        
        # 调用 VLM
        raw_output = agent.solve(
            prob['question'], 
            prob['choices'], 
            full_image_path, 
            prob.get('context'), 
            prob.get('lecture'), 
            prob.get('hint')
        )
        
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