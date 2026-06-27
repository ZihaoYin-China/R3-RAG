import os
import json
import argparse
import re
import base64
import requests
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Dual Agent (VLM + LLM) without RAG")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--test_split', type=str, default='test')
    # 默认值仅供参考，请在命令行指定准确的 model name
    parser.add_argument('--vlm_model', type=str, default='qwen2.5-vl:72b')
    parser.add_argument('--llm_model', type=str, default='qwen2.5:32b')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434')
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--label', type=str, default='dual_agent_norag')
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

class DualAgent:
    def __init__(self, args):
        self.host = args.ollama_host
        self.vlm = args.vlm_model
        self.llm = args.llm_model
        
        # === 安全检查：防止模型名写错导致跑空 ===
        print(f"🔍 Checking models on {self.host}...")
        self._check_model_exists(self.vlm)
        self._check_model_exists(self.llm)
        print("✅ Models verified! Starting Vanilla Baseline.")

    def _check_model_exists(self, model_name):
        try:
            res = requests.post(f"{self.host}/api/show", json={"name": model_name})
            if res.status_code != 200:
                print(f"\n❌ FATAL ERROR: Model '{model_name}' not found!")
                print("Please check 'ollama list' for exact names.")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Connection Error: {e}")
            sys.exit(1)

    def get_caption(self, image_path):
        """第一棒：VLM 看图说话 (原始朴素 Prompt)"""
        if not image_path: return ""
        
        img_b64 = encode_image(image_path)
        if not img_b64: return ""

        # === 原始 Prompt，无优化 ===
        prompt = "Describe this image in detail for a science question. Focus on objects, text, diagrams, and relationships."
        
        payload = {
            "model": self.vlm,
            "prompt": prompt,
            "stream": False,
            "images": [img_b64],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status() # 遇到 API 错误直接报错，不要吞掉
            res = response.json()
            return res.get("response", "").strip()
        except Exception as e:
            print(f"\n❌ CRITICAL VLM FAILURE on image {image_path}: {e}")
            # 这里如果不抛出异常，就会导致 Baseline 变成纯盲猜。
            # 为了数据有效性，建议打印错误。如果是偶发网络错误可以 pass，但如果是 404 必须注意。
            return ""

    def solve(self, question, options, caption):
        """第二棒：LLM 做题 (原始朴素 Prompt)"""
        opt_lines = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        context_str = f"Image Description: {caption}" if caption else "Image Description: No image."
        
        # === 原始 Prompt，无 COT 引导，无角色扮演 ===
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opt_lines}\n\n"
            "Instruction: Based on the image description (if any) and your internal knowledge, answer the question.\n"
            "First explain your reasoning, then conclude with 'The answer is (X)'.\n"
            "Reasoning and Answer:"
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
            res = response.json()
            return res.get("response", "")
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

def extract_answer(text, options):
    text = str(text).strip()
    m = re.search(r"answer is \(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    m = re.search(r"Answer:\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m: return options.index(m.group(1).upper())
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches: return options.index(matches[-1])
    return None

def main():
    args = parse_args()
    print(f"==== Running Dual Agent (Vanilla Baseline) ====")
    print(f"Vision Model: {args.vlm_model}")
    print(f"Reasoning Model: {args.llm_model}")
    
    if args.output_root is None:
        args.output_root = os.path.join(os.path.dirname(args.data_root), "results_dual")
    os.makedirs(args.output_root, exist_ok=True)
    result_file = os.path.join(args.output_root, f"{args.label}.json")
    
    # 路径容错处理
    p_path = os.path.join(args.data_root, "problems.json")
    if not os.path.exists(p_path):
        alt_path = os.path.join(args.data_root, f"problems_{args.test_split}.json")
        if os.path.exists(alt_path):
            p_path = alt_path
        else:
            print(f"❌ Error: Cannot find problems.json in {args.data_root}")
            return

    problems = json.load(open(p_path, "r", encoding="utf-8"))
    
    pid_path = os.path.join(args.data_root, "pid_splits.json")
    if not os.path.exists(pid_path):
        pid_path = os.path.join(os.path.dirname(args.data_root), "pid_splits.json")

    if os.path.exists(pid_path):
        qids = json.load(open(pid_path, "r", encoding="utf-8")).get(args.test_split, [])
    else:
        qids = list(problems.keys())

    if args.test_number > 0: qids = qids[:args.test_number]

    agent = DualAgent(args)
    correct = 0
    results = {}
    options = ["A", "B", "C", "D", "E"]
    
    print(f"🚀 Starting inference on {len(qids)} questions...")
    
    for i, qid in enumerate(tqdm(qids)):
        prob = problems.get(qid)
        
        image_path = None
        caption = ""
        if prob.get('image'):
            # 路径拼接逻辑
            full_path = os.path.join(args.image_root, args.test_split, prob['image'])
            if not os.path.exists(full_path):
                full_path = os.path.join(args.image_root, prob['image'])
            
            if os.path.exists(full_path):
                caption = agent.get_caption(full_path)
            
        raw_output = agent.solve(prob['question'], prob['choices'], caption)
        
        gt = prob.get('answer')
        pred = extract_answer(raw_output, options)
        is_corr = (pred == gt)
        if is_corr: correct += 1
        
        results[qid] = {
            "caption": caption,
            "pred_raw": raw_output,
            "pred_idx": pred,
            "gt_idx": gt,
            "is_correct": is_corr
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