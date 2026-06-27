#import os
#import re
#import base64
#import numpy as np
#from typing import List, Tuple
#from langchain_core.messages import HumanMessage
#
#try:
#    from FlagEmbedding import FlagReranker
#except ImportError:
#    FlagReranker = None
#    print("[WARN] FlagEmbedding not installed. Text reranking will be disabled.")
#
#class AdaptiveGatedReranker:
#    def __init__(self, config):
#        self.config = config
#        self.rerank_top_k = int(getattr(config, "rerank_top_k", 5))
#        
#        self.use_rerank = (FlagReranker is not None)
#        if self.use_rerank:
#            model_name = getattr(config, "rerank_model_name", "BAAI/bge-reranker-v2-m3")
#            try:
#                print(f"[Reranker] Loading BGE model: {model_name}...")
#                self.bge_reranker = FlagReranker(model_name, use_fp16=True)
#            except Exception as e:
#                print(f"[Reranker] Failed to load BGE: {e}")
#                self.use_rerank = False
#
#    def rerank(self, query: str, options: List[str], candidates: List[str], 
#               image_path: str = None, vl_llm = None) -> Tuple[str, str, float, float]:
#        """
#        Returns: (final_text, debug_info, top_bge_score, top_vl_score)
#        """
#        # 默认分数 (用于表示无效/未计算)
#        top_bge = -999.0
#        top_vl = -1.0
#
#        if not candidates:
#            return "", "No candidates found.", top_bge, top_vl
#            
#        if not self.use_rerank:
#            debug_info = "No reranking applied (disabled)."
#            final_text = "\n\n".join(candidates[:self.rerank_top_k])
#            return final_text, debug_info, 0.0, -1.0
#
#        # --- Stage 1: BGE Text Rerank ---
#        opt_str = " ".join(options)
#        query_pair = f"{query} {opt_str}"
#        pairs = [[query_pair, doc] for doc in candidates]
#        
#        scores = self.bge_reranker.compute_score(pairs)
#        if isinstance(scores, float): scores = [scores]
#        
#        ranked_stage1 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
#        
#        # ✅ 捕获 BGE 最高分
#        if ranked_stage1:
#            top_bge = ranked_stage1[0][1]
#
#        stage1_limit = 15
#        stage2_candidates = ranked_stage1[:stage1_limit]
#        
#        # --- Stage 2: Adaptive VL Rerank ---
#        has_image = bool(image_path and vl_llm)
#        final_ranked_list = []
#        debug_msg = ""
#        
#        if has_image:
#            final_ranked_list, top_vl_raw = self._vl_adaptive_scoring(
#                query, options, image_path, stage2_candidates, vl_llm
#            )
#            # ✅ 捕获 VL 最高分
#            top_vl = top_vl_raw
#            debug_msg = f"Cascade Rerank: BGE({len(candidates)}) -> VL({len(stage2_candidates)})"
#        else:
#            final_ranked_list = stage2_candidates
#            debug_msg = f"Single-Stage Rerank: BGE({len(candidates)})"
#
#        # --- Final Output ---
#        final_top_docs = final_ranked_list[:self.rerank_top_k]
#        lines = [f"[Reranked-{i+1} s={item[1]:.2f}] {item[0]}" for i, item in enumerate(final_top_docs)]
#        final_text = "\n\n".join(lines)
#        
#        return final_text, f"{debug_msg} -> Top-{self.rerank_top_k}", top_bge, top_vl
#
#    def _vl_adaptive_scoring(self, query, options, image_path, candidates, vl_llm):
#        try:
#            with open(image_path, "rb") as f:
#                b64_img = base64.b64encode(f.read()).decode('utf-8')
#        except: return candidates, -1.0
#
#        visual_keywords = ["image", "figure", "graph", "diagram", "map", "chart", "picture", "shown", "look at"]
#        is_visual_heavy = any(k in query.lower() for k in visual_keywords)
#        
#        temp_results = []
#        vl_raw_scores = []
#
#        for doc, bge_score in candidates:
#            prompt = (
#                f"Question: {query}\nOptions: {options}\nEvidence: {doc[:600]} ...\n\n"
#                "Task: Rate relevance (0-10) based on question and IMAGE consistency.\n"
#                "Output ONLY a number."
#            )
#            msg = HumanMessage(content=[
#                {"type": "text", "text": prompt},
#                {"type": "image_url", "image_url": f"data:image/png;base64,{b64_img}"}
#            ])
#            
#            vl_score = 5.0
#            try:
#                resp = vl_llm.invoke([msg])
#                match = re.search(r"(\d+(\.\d+)?)", resp.content.strip())
#                if match: vl_score = float(match.group(1))
#            except: pass
#            
#            temp_results.append({"doc": doc, "bge_raw": bge_score, "vl_raw": vl_score})
#            vl_raw_scores.append(vl_score)
#
#        if not temp_results: return [], -1.0
#
#        score_std = np.std(vl_raw_scores) if len(vl_raw_scores) > 1 else 0.0
#        base_lambda = 0.15 
#        if is_visual_heavy: base_lambda += 0.15
#        if score_std > 2.0: base_lambda += 0.10
#        elif score_std < 0.5: base_lambda -= 0.10
#        final_lambda = max(0.05, min(0.45, base_lambda))
#        
#        bge_vals = [x["bge_raw"] for x in temp_results]
#        min_b, max_b = min(bge_vals), max(bge_vals)
#        rng = max_b - min_b
#        
#        final_list = []
#        for item in temp_results:
#            norm_bge = (item["bge_raw"] - min_b) / rng if rng > 1e-6 else 1.0
#            norm_vl = item["vl_raw"] / 10.0
#            final_score = ((1 - final_lambda) * norm_bge) + (final_lambda * norm_vl)
#            final_list.append((item["doc"], final_score))
#            
#        final_list.sort(key=lambda x: x[1], reverse=True)
#        
#        # 返回排序后的列表 和 这一轮最高的原始 VL 分数
#        top_vl_raw = max(vl_raw_scores) if vl_raw_scores else -1.0
#        return final_list, top_vl_raw
import re
import base64
import os
import numpy as np
from typing import List, Tuple
from PIL import Image  # 🚀 新增：用于图像压缩
from io import BytesIO # 🚀 新增：用于内存中转图像

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None
    print("[WARN] FlagEmbedding not installed. Text reranking will be disabled.")

class AdaptiveGatedReranker:
    def __init__(self, config):
        self.config = config
        self.rerank_top_k = int(getattr(config, "rerank_top_k", 5))
        
        self.use_rerank = (FlagReranker is not None)
        if self.use_rerank:
            model_name = getattr(config, "rerank_model_name", "BAAI/bge-reranker-v2-m3")
            try:
                print(f"[Reranker] Loading BGE model: {model_name}...")
                self.bge_reranker = FlagReranker(model_name, use_fp16=True)
            except Exception as e:
                print(f"[Reranker] Failed to load BGE: {e}")
                self.use_rerank = False

    def rerank(self, query: str, options: List[str], candidates: List[str], 
               image_path: str = None, vl_llm = None) -> Tuple[str, str, float, float]:
        """
        Returns: (final_text, debug_info, top_bge_score, top_vl_score)
        """
        # 默认分数 (用于表示无效/未计算)
        top_bge = -999.0
        top_vl = -1.0

        if not candidates:
            return "", "No candidates found.", top_bge, top_vl
            
        if not self.use_rerank:
            debug_info = "No reranking applied (disabled)."
            final_text = "\n\n".join(candidates[:self.rerank_top_k])
            return final_text, debug_info, 0.0, -1.0

        # --- Stage 1: BGE Text Rerank ---
        opt_str = " ".join(options)
        query_pair = f"{query} {opt_str}"
        pairs = [[query_pair, doc] for doc in candidates]
        
        scores = self.bge_reranker.compute_score(pairs)
        if isinstance(scores, float): scores = [scores]
        
        ranked_stage1 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # 捕获 BGE 最高分
        if ranked_stage1:
            top_bge = ranked_stage1[0][1]

        stage1_limit = 3
        stage2_candidates = ranked_stage1[:stage1_limit]
        
        # --- Stage 2: Adaptive VL Rerank ---
        has_image = bool(image_path and vl_llm)
        final_ranked_list = []
        debug_msg = ""
        
        if has_image:
            final_ranked_list, top_vl_raw = self._vl_adaptive_scoring(
                query, options, image_path, stage2_candidates, vl_llm
            )
            # 捕获 VL 最高分
            top_vl = top_vl_raw
            debug_msg = f"Cascade Rerank: BGE({len(candidates)}) -> VL({len(stage2_candidates)})"
        else:
            final_ranked_list = stage2_candidates
            debug_msg = f"Single-Stage Rerank: BGE({len(candidates)})"

        # --- Final Output ---
        final_top_docs = final_ranked_list[:self.rerank_top_k]
        lines = [f"[Reranked-{i+1} s={item[1]:.2f}] {item[0]}" for i, item in enumerate(final_top_docs)]
        final_text = "\n\n".join(lines)
        
        return final_text, f"{debug_msg} -> Top-{self.rerank_top_k}", top_bge, top_vl

    def _vl_adaptive_scoring(self, query, options, image_path, candidates, vl_llm):
        # =================== 【核心优化：物理限速器】 ===================
        try:
            with Image.open(image_path) as img:
                # 强行将图片最长边压缩至 512 像素，大幅削减 Token 数量！
                img.thumbnail((512, 512))
                
                # 转换颜色通道，防止带有透明通道的 PNG 报错
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # 存入内存并转为 Base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: 
            print(f"[Warning] Image load/compression failed: {e}")
            return candidates, -1.0
        # ================================================================

        visual_keywords = ["image", "figure", "graph", "diagram", "map", "chart", "picture", "shown", "look at"]
        is_visual_heavy = any(k in query.lower() for k in visual_keywords)
        
        temp_results = []
        vl_raw_scores = []

        print(f"[Reranker] Starting VLM scoring for {len(candidates)} docs...")

        for doc, bge_score in candidates:
            # =================== 【优化后的专家级 Prompt】 ===================
            prompt = (
                "You are a strict Evaluator Engine.\n"
                "Your Goal: Score the relevance between the TEXT EVIDENCE and the IMAGE relative to the QUESTION.\n\n"
                f"--- INPUT DATA ---\n"
                f"Question: {query}\n"
                f"Options: {options}\n"
                f"Text Evidence: {doc[:600]} ...\n"
                f"--- END DATA ---\n\n"
                "Scoring Criteria:\n"
                "- 0.0: Evidence contradicts image or is completely irrelevant.\n"
                "- 5.0: Evidence is generic text but not contradictory.\n"
                "- 10.0: Evidence perfectly matches the specific visual details in the image.\n\n"
                "### OUTPUT FORMAT INSTRUCTION ###\n"
                "You must respond with a specific numeric format only. Do not output any explanation.\n"
                "Example Response: 0.0\n"
                "Example Response: 8.5\n\n"
                "Your Score:"
            )
            # ==============================================================
            
            vl_score = 5.0 # Default fallback
            try:
                # 【核心修复】摒弃复杂的 HumanMessage，直接使用原生 invoke 传递 kwargs
                resp = vl_llm.invoke(prompt, images=[b64_img])
                
                # 兼容 str 和 AIMessage
                if isinstance(resp, str):
                    content = resp.strip()
                else:
                    content = getattr(resp, "content", "").strip()
                
                print(f"[Debug] VLM Raw Response: {content}")
                
                match = re.search(r"(\d+(\.\d+)?)", content)
                if match: 
                    vl_score = float(match.group(1))
                else:
                    print(f"[Warning] Regex failed to find number in: {content}")
                    
            except Exception as e:
                # 打印详细报错信息
                print(f"[Error] VLM Invoke Failed: {e}")
            
            temp_results.append({"doc": doc, "bge_raw": bge_score, "vl_raw": vl_score})
            vl_raw_scores.append(vl_score)

        if not temp_results: return [], -1.0

        score_std = np.std(vl_raw_scores) if len(vl_raw_scores) > 1 else 0.0
        base_lambda = 0.15 
        if is_visual_heavy: base_lambda += 0.15
        if score_std > 2.0: base_lambda += 0.10
        elif score_std < 0.5: base_lambda -= 0.10
        final_lambda = max(0.05, min(0.45, base_lambda))
        
        bge_vals = [x["bge_raw"] for x in temp_results]
        min_b, max_b = min(bge_vals), max(bge_vals)
        rng = max_b - min_b
        
        final_list = []
        for item in temp_results:
            norm_bge = (item["bge_raw"] - min_b) / rng if rng > 1e-6 else 1.0
            norm_vl = item["vl_raw"] / 10.0
            final_score = ((1 - final_lambda) * norm_bge) + (final_lambda * norm_vl)
            final_list.append((item["doc"], final_score))
            
        final_list.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的列表 和 这一轮最高的原始 VL 分数
        top_vl_raw = max(vl_raw_scores) if vl_raw_scores else -1.0
        return final_list, top_vl_raw