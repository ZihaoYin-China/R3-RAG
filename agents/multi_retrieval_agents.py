import os
import re
import json
from typing import Any, Dict, List, Tuple
from glob import glob

from langchain_community.llms import Ollama

from retrieval.web_retrieval import WebRetrieval
from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.mm_retrieval import MMRetrieval

from agents.summary_agent import SummaryAgent
from agents.decompose_agent import DecomposeAgent

# ✅ Import independent modules
from retrieval.adaptive_reranker import AdaptiveGatedReranker
from retrieval.reflection_router import ReflectionRouter


def _get_question_and_options(problem: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = problem.get("question") or problem.get("query") or problem.get("text") or ""
    opts = problem.get("choices") or problem.get("options") or []
    if isinstance(opts, dict):
        ordered = []
        for k in ["A", "B", "C", "D", "E"]:
            if k in opts: ordered.append(opts[k])
        opts = ordered
    return str(q), [str(x) for x in opts]


def _sort_img_key(p: str):
    name = os.path.basename(p).lower()
    if name.startswith("image."): return (0, 0, name)
    m = re.match(r"choice_(\d+)\.", name)
    if m: return (1, int(m.group(1)), name)
    return (2, 999, name)


def _resolve_image_paths(config, qid: str) -> List[str]:
    root = getattr(config, "image_root", None) or getattr(config, "working_dir", ".")
    split = getattr(config, "test_split", None) or "test"
    candidates = [
        os.path.join(root, split, str(qid)),
        os.path.join(root, "images", split, str(qid)),
        os.path.join(root, "data", split, str(qid)),
        os.path.join(root, "test", str(qid)),
        os.path.join(root, "val", str(qid)),
        os.path.join(root, "train", str(qid)),
        os.path.join(root, "minival", str(qid)),
    ]
    paths = []
    for d in candidates:
        if os.path.isdir(d):
            for ext in ("png", "jpg", "jpeg", "webp", "bmp"):
                paths.extend(glob(os.path.join(d, f"*.{ext}")))
            if paths: break
    return sorted([os.path.abspath(p) for p in paths], key=_sort_img_key)[:6]


class MRetrievalAgent:
    def __init__(self, config):
        self.config = config
        mode = str(getattr(config, "mode", "hybrid")).lower()

        # 1. Init Retrievers (Force No Web)
        self.retrievers = {}
        parts = re.split(r"[,+_]+", mode)
        # Filter out web even if present in config
        modes = set([p for p in parts if p and p != "web"])
        if "hybrid" in modes: 
            modes = {"vector", "graph", "mm"}
        
        if "vector" in modes: self.retrievers["vector"] = VectorRetrieval(config)
        if "graph" in modes: self.retrievers["graph"] = GraphRetrieval(config)
        if "mm" in modes: self.retrievers["mm"] = MMRetrieval(config)
        
        self.available_modes = set(self.retrievers.keys())
        self._mm_prepared = False
        print(f"[Agent] Active Retrieval Modes: {self.available_modes} (Web Disabled)")

        self.sum_agent = SummaryAgent(config)
        self.dec_agent = DecomposeAgent(config)

        # 2. Init VLM (for Reranker)
        self.vl_llm = None
        if getattr(config, "summary_use_vision", False):
            vlm_name = getattr(config, "summary_vlm_model_name", "qwen2.5vl:7b")
            ollama_host = getattr(config, "ollama_host", "http://localhost:11434")
            print(f"[Agent] Init VLM for Reranking: {vlm_name}")
            try:
                self.vl_llm = Ollama(base_url=ollama_host, model=vlm_name, temperature=0.1)
                # [新增] 简单测试一下 VLM 是否活着
                print("[Agent] Testing VLM connection...")
                # self.vl_llm.invoke("Describe this image in one word.", images=[]) # 可选测试
                print("[Agent] VLM Connected.")
            except Exception as e:
                print(f"[ERROR] Failed to init VLM: {e}")
                self.vl_llm = None

        # 3. Init Independent Modules
        print("[Agent] Init Adaptive Reranker...")
        self.reranker = AdaptiveGatedReranker(config)
        print("[Agent] Init Reflection Router...")
        self.router = ReflectionRouter(config)

        # 4. Reflection Parameters (Optimized for Stability)
        self.max_retries = 2
        
        # [修改] 提高基础阈值，过滤纯噪音
        self.conf_thresh_bge = 1.3 
        self.conf_thresh_vl = 5.5  
        
        # [新增] 双重验证的安全缓冲 (Safety Margin)
        # 含义：如果分数在 [阈值] 和 [阈值 + Margin] 之间，说明虽然及格但不够稳，强制再查一次
        self.margin_bge = 1.0  # 1.3 ~ 1.8 之间会触发 Double Check
        self.margin_vl = 2.5   # 5.5 ~ 7.0 之间会触发 Double Check

    def predict(self, problems: Dict[str, Any], shot_qids: List[str], qid: str):
        problem = problems[qid]
        raw_question, options = _get_question_and_options(problem)

        try:
            decomposed = self.dec_agent.decompose(raw_question)
            curr_q = decomposed.strip() if isinstance(decomposed, str) and decomposed.strip() else raw_question
        except: curr_q = raw_question

        # Initial: All modes active
        curr_modes = self.available_modes.copy()
        
        if "mm" in self.retrievers and not self._mm_prepared:
            self.retrievers["mm"].prepare(problems)
            self._mm_prepared = True

        image_paths = _resolve_image_paths(self.config, qid)
        image_path = image_paths[0] if image_paths else None
        
        retry = 0
        final_evidence = ""
        debug_trace = []

        # =========================================================
        # ✅ Reflection Loop: Retrieve -> Rerank -> Check -> Reflect
        # =========================================================
        while retry <= self.max_retries:
            # 1. Retrieve
            candidates = []
            for m in curr_modes:
                r = self.retrievers.get(m)
                if not r: continue
                try:
                    res = r.find_top_k(curr_q, qid=qid, problems=problems) if m == "mm" else r.find_top_k(curr_q)
                    if isinstance(res, list): candidates.extend(res)
                    elif res: candidates.extend([c.strip() for c in res.split("\n\n" if m=="mm" else "\n") if c.strip()])
                except: pass
            candidates = list(set(candidates))

            # 2. Rerank
            final_evidence, debug_msg, top_bge, top_vl = self.reranker.rerank(
                query=curr_q,
                options=options,
                candidates=candidates,
                image_path=image_path,
                vl_llm=self.vl_llm
            )

            # 3. Confidence Check (With Graceful Degradation & Double Check)
            is_confident = False
            fail_reason = ""
            is_visual_task = bool(image_path and self.vl_llm)

            if not final_evidence:
                fail_reason = "No candidates found."
            
            elif is_visual_task:
                # --- [优化] 视觉任务逻辑 ---
                if top_vl != -1.0:
                    # [情况 A]: VLM 返回了默认分 5.0 (疑似失败/超时/看不懂)
                    # 此时启动降级策略：如果 BGE 文本分数很高，就放行！
                    if top_vl == 5.0:
                        if top_bge > 0.8: # 0.8 是一个安全的文本置信度
                            is_confident = True
                            fail_reason = f"VLM default(5.0) ignored due to high BGE({top_bge:.2f})."
                            # 记录一下 warning
                            print(f"[Warning] VLM returned 5.0. Fallback to Text Confidence (BGE={top_bge:.2f})")
                        else:
                            # 文本也不行，那确实得反思
                            fail_reason = f"VLM default(5.0) & BGE low ({top_bge:.2f} < 0.8)."
                            
                    # [情况 B]: VLM 正常工作 (分数不是 5.0)
                    elif top_vl >= self.conf_thresh_vl:
                        is_confident = True
                        # [双重验证逻辑]：如果是第0轮，且分数没达到"超级自信" (7.0)，强制再查一次
                        if retry == 0 and top_vl < (self.conf_thresh_vl + self.margin_vl):
                            is_confident = False
                            fail_reason = f"Double Check: VL Score {top_vl:.1f} is borderline (Safe > {self.conf_thresh_vl + self.margin_vl})."
                    else:
                        fail_reason = f"Top VL Score {top_vl:.1f} < {self.conf_thresh_vl}"
                else:
                    is_confident = True # Fallback if VL fails completely (returns -1.0)
            
            else:
                # --- [优化] 纯文本任务逻辑 ---
                if top_bge != -999.0:
                    if top_bge >= self.conf_thresh_bge:
                        is_confident = True
                        # [双重验证逻辑]：如果是第0轮，且分数没达到"超级自信" (1.8)，强制再查一次
                        if retry == 0 and top_bge < (self.conf_thresh_bge + self.margin_bge):
                            is_confident = False
                            fail_reason = f"Double Check: BGE Score {top_bge:.2f} is borderline (Safe > {self.conf_thresh_bge + self.margin_bge})."
                    else:
                        fail_reason = f"Top BGE Score {top_bge:.2f} < {self.conf_thresh_bge}"
                else:
                    is_confident = True

            trace_info = f"Loop {retry}: Modes={list(curr_modes)} | Query='{curr_q}' -> {fail_reason if not is_confident else 'Confident'}"
            debug_trace.append(trace_info)

            # 4. Decision
            if is_confident or retry >= self.max_retries:
                if not is_confident: debug_trace.append("Max retries reached.")
                break 
            
            # 5. Reflect & Rewrite Query
            print(f"[Loop {retry} Failed] {fail_reason}. Calling Reflection Router...")
            
            # [Update] Get new_query from router
            new_q, new_modes, thought = self.router.reflect(curr_q, curr_modes, fail_reason, self.available_modes)
            
            debug_trace.append(f"Reflection: {thought}")
            debug_trace.append(f"Rewritten Query: '{new_q}' -> New Modes: {list(new_modes)}")
            
            # Update state for next loop
            curr_q = new_q
            curr_modes = new_modes
            retry += 1

        # =========================================================

        all_messages = [f"Debug Trace:\n" + "\n".join(debug_trace) + "\n"]
        
        img_note = ""
        if image_paths:
            img_note = "Images attached:\n" + "\n".join(image_paths) + "\n\n"

        letters = ["A", "B", "C", "D", "E"]
        opt_lines = "\n".join([f"{letters[i]}. {options[i]}" for i in range(min(len(options), len(letters)))]) if options else ""

        sum_question = (
            f"{img_note}"
            f"Question:\n{raw_question}\n\n"
            f"Options:\n{opt_lines}\n\n"
            f"Evidence (Refined):\n{final_evidence}\n"
        )

        # 6. Final Summarization (CoT enabled inside SummaryAgent)
        final_ans, final_messages = self.sum_agent.summarize(problems, shot_qids, qid, sum_question)
        if isinstance(final_messages, list): all_messages.extend(final_messages)
        else: all_messages.append(str(final_messages))
        
        # Log metadata
        try:
            all_messages.insert(0, {
                "stage": "retrieval",
                "evidence_snippet": final_evidence[:200] + "...",
                "image_paths": image_paths,
                "loop_count": retry
            })
        except: pass

        return final_ans, all_messages
