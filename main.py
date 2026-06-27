import os
import re
import json
import argparse
import random
from tqdm import tqdm

from agents.multi_retrieval_agents import MRetrievalAgent


def parse_args():
    # TODO(reproducibility): add versioned, machine-independent experiment
    # profiles once model endpoints and generated index layouts are portable.
    # Until then, paths and serving options must be supplied for each machine.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--image_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--label', type=str, default='run')
    parser.add_argument('--debug', action='store_true', help='debug mode: only run a few samples')
    parser.add_argument('--caption_file', type=str, default=None)  # 不再强依赖
    parser.add_argument('--save_every', type=int, default=50)

    parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument('--options', nargs='+', default=["A", "B", "C", "D", "E"])

    # user options
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val', 'minival'])
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='CQM-A',
        choices=[
            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
        ],
        help='prompt format template'
    )

    # Retrieval settings
    parser.add_argument('--working_dir', type=str, required=True)

    # ✅ LightRAG KB workdir
    parser.add_argument(
        '--lightrag_workdir',
        type=str,
        default=None,
        help='LightRAG KB workdir (contains rag_storage, vdb_*.json, graph_*.graphml, etc.)'
    )

    parser.add_argument('--llm_model_name', type=str, required=True,
                        help='Language model served by the configured backend.')
    parser.add_argument('--mode', type=str, default='hybrid')
    parser.add_argument('--serper_api_key', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=4)

    # ✅ 新增：重排序参数注册
    parser.add_argument('--rerank_top_k', type=int, default=5, help='Number of documents to keep after reranking')
    parser.add_argument('--rerank_model_name', type=str, required=True,
                        help='Hugging Face model name or path for the reranker.')

    # ---- Ollama host + Summary VLM (final step truly sees images) ----
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434',
                        help='Ollama host, e.g. http://localhost:11434')
    parser.add_argument('--summary_use_vision', action='store_true',
                        help='Enable vision-capable SummaryAgent and send images to Ollama.')
    parser.add_argument('--summary_vlm_model_name', type=str, default='',
                        help='Ollama vision model name, e.g. qwen2.5vl:7b / llama3.2-vision')
    parser.add_argument('--summary_temperature', type=float, default=0.2)
    parser.add_argument('--ollama_timeout', type=float, default=120)

    # ---- Multimodal embedding settings (OpenCLIP for retrieval) ----
    parser.add_argument('--device', type=str, default='',
                        help="cuda/cpu. Empty=auto.")
    parser.add_argument('--clip_model', type=str, required=True,
                        help='OpenCLIP architecture name.')
    parser.add_argument('--clip_pretrained', type=str, required=True,
                        help='OpenCLIP pretrained checkpoint tag or path.')
    parser.add_argument('--clip_batch_size', type=int, default=32)

    # ---- LightRAG stability knobs ----
    parser.add_argument('--lightrag_llm_max_async', type=int, default=8,
                        help='Max concurrent LLM calls inside LightRAG (smaller is safer).')
    parser.add_argument('--lightrag_num_ctx', type=int, default=8192,
                        help='Ollama num_ctx for LightRAG internal llm (smaller is safer).')
    parser.add_argument('--lightrag_embed_model', type=str, required=True,
                        help='Embedding model name served by Ollama for LightRAG.')

    # GPT settings
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_number', type=int, default=-1, help='>0: run first N samples; <=0: run full split')

    # few-shot
    parser.add_argument('--shot_qids', type=str, default='',
                        help='path to qid list OR comma-separated qids; empty to disable')

    parser.add_argument('--engine', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='The maximum number of tokens allowed for the generated answer.')

    return parser.parse_args()


def _read_shot_qids(shot_qids_arg: str):
    """支持：空 / 文件 / 逗号分隔字符串"""
    if shot_qids_arg is None:
        return []
    shot_qids_arg = str(shot_qids_arg).strip()
    if shot_qids_arg == "":
        return []

    # 文件路径：json 或 txt
    if os.path.exists(shot_qids_arg):
        try:
            obj = json.load(open(shot_qids_arg, "r", encoding="utf-8"))
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass

        # txt: 每行一个
        qids = []
        with open(shot_qids_arg, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    qids.append(str(s))
        return qids

    # 逗号/空格分隔
    parts = re.split(r"[,\s]+", shot_qids_arg)
    return [str(x) for x in parts if str(x).strip()]


def ans_to_idx(ans, options):
    """
    把模型输出(如 'B' / 'Answer: B' / '1') 统一转成 0..len(options)-1
    """
    if ans is None:
        return None
    s = str(ans).strip().upper()

    # 匹配字母选项
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        ch = m.group(1)
        if ch in options:
            return options.index(ch)

    # 匹配数字 0-9（兼容某些模型直接输出 index）
    m = re.search(r"\b(\d+)\b", s)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(options):
            return idx

    return None


def load_data(args):
    def find_one(root, names):
        # 先当前目录直找
        for nm in names:
            p = os.path.join(root, nm)
            if os.path.exists(p):
                return p
        # 再递归找
        for dp, _, fns in os.walk(root):
            for nm in names:
                if nm in fns:
                    return os.path.join(dp, nm)
        return None

    root = args.data_root or args.working_dir

    problems_path = find_one(root, [
        "problems.json",
        f"problems_{args.test_split}.json",
        "problems_test.json",
        "problems_val.json",
        "problems_minival.json",
    ])
    if problems_path is None:
        raise FileNotFoundError(f"Cannot find problems*.json under: {root}")

    pid_splits_path = find_one(root, ["pid_splits.json"])

    problems = json.load(open(problems_path, "r", encoding="utf-8"))

    # pid_splits 没有的话：直接用 problems 的 keys 当 split
    if pid_splits_path and os.path.exists(pid_splits_path):
        pid_splits = json.load(open(pid_splits_path, "r", encoding="utf-8"))
        qids = [str(q) for q in pid_splits.get(args.test_split, [])]
        train_qids = [str(q) for q in pid_splits.get("train", [])]
    else:
        qids = [str(k) for k in problems.keys()]
        train_qids = []

    # captions.json 不再强依赖：存在则读，不存在就空
    captions = {}
    if args.caption_file and os.path.exists(args.caption_file):
        try:
            captions = json.load(open(args.caption_file, "r", encoding="utf-8")).get("captions", {}) or {}
        except Exception:
            captions = {}

    for qid in problems:
        problems[qid]["caption"] = captions.get(str(qid), "")

    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"problems_path: {problems_path}")
    print(f"pid_splits_path: {pid_splits_path or 'N/A'}")
    print(f"number of {args.test_split} problems: {len(qids)}\n")

    # few-shot
    shot_qids = _read_shot_qids(args.shot_qids)
    if train_qids and shot_qids:
        for qid in shot_qids:
            assert qid in train_qids, f"shot_qid {qid} not in train split"
    print("training question ids for prompting:", shot_qids, "\n")

    return problems, qids, shot_qids


def main():
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    # --- fill default paths ---
    if args.data_root is None:
        args.data_root = args.working_dir
    if args.output_root is None:
        args.output_root = os.path.join(args.working_dir, 'results')

    # ✅ IMPORTANT: image_root 默认指向 working_dir
    # 你的图片结构是：<working_dir>/<split>/<qid>/image.png
    if args.image_root is None:
        args.image_root = args.working_dir

    # caption_file：默认尝试 working_dir/captions.json；不存在则置空，避免 load_data 崩
    if args.caption_file is None:
        cand = os.path.join(args.working_dir, 'captions.json')
        args.caption_file = cand if os.path.exists(cand) else None

    # ✅ LightRAG workdir：优先用命令行 --lightrag_workdir
    if args.lightrag_workdir is None:
        # 常见默认：<working_dir>/ScienceQA_lightrag_workdir
        cand = os.path.join(args.working_dir, 'ScienceQA_lightrag_workdir')
        args.lightrag_workdir = cand if os.path.exists(cand) else args.working_dir

    print(f"[INFO] working_dir      = {args.working_dir}")
    print(f"[INFO] data_root        = {args.data_root}")
    print(f"[INFO] image_root       = {args.image_root}")
    print(f"[INFO] lightrag_workdir = {args.lightrag_workdir}")
    print("--------------------------------------------------")

    problems, qids, shot_qids = load_data(args)

    os.makedirs(args.output_root, exist_ok=True)
    result_file = os.path.join(args.output_root, f"{args.label}_{args.test_split}.json")

    agent = MRetrievalAgent(args)
    correct = 0
    results = {}
    failed = []

    options = [str(x).upper() for x in args.options]

    for i, qid in enumerate(qids):
        if args.debug and i > 10:
            break
        if args.test_number > 0 and i >= args.test_number:
            break

        qid = str(qid)
        problem = problems[qid]
        gt_idx = int(problem["answer"])  # ScienceQA: 0..4
        gt_letter = options[gt_idx] if 0 <= gt_idx < len(options) else ""

        final_ans, all_messages = agent.predict(problems, shot_qids, qid)

        pred_idx = ans_to_idx(final_ans, options)
        pred_letter = options[pred_idx] if pred_idx is not None else ""

        is_correct = (pred_idx == gt_idx) if pred_idx is not None else False
        if is_correct:
            correct += 1
        else:
            failed.append(qid)

        results[qid] = {
            "pred": pred_letter,
            "gt": gt_letter,
            "gt_idx": gt_idx,
            "is_correct": bool(is_correct),
            "messages": all_messages,
        }

        if (i + 1) % args.save_every == 0:
            with open(result_file, 'w', encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_file} after {i + 1} examples.")

    with open(result_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(qids) if (args.test_number <= 0) else min(len(qids), args.test_number)
    total = min(total, (11 if args.debug else total))  # debug 下最多跑 11 条

    print(f"Results saved to {result_file} after {total} examples.")
    print(f"Number of correct answers: {correct}/{total}")
    print(f"Accuracy: {correct / total:.4f}")
    print(f"Failed question ids: {failed[:50]}{' ...' if len(failed) > 50 else ''}")
    print(f"Number of failed question ids: {len(failed)}")


if __name__ == "__main__":
    main()
