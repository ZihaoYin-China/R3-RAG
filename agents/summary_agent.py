import re
import base64
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

# 推荐使用 ChatOllama 以获得更好的多模态支持，或者继续用 Ollama
from langchain_community.llms import Ollama

def _extract_choice_letter(text: str) -> str:
    """
    从模型输出中提取选项字母。
    针对 CoT (Reasoning -> Answer) 进行了优化。
    """
    t = (text or "").strip()
    
    # 1. 最优先：抓取明确的 "Answer: C" 格式
    m = re.search(r"(?:Answer|答案|Option)\s*[:：]\s*([A-E])", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
        
    # 2. 次优先：抓取最后出现的单独字母 (防止推理过程中提到其他选项)
    matches = re.findall(r"\b([A-E])\b", t.upper())
    if matches:
        return matches[-1]
        
    # 3. 兜底
    return t[:100]

class SummaryAgent:
    def __init__(self, config):
        self.config = config
        self.model_name = getattr(config, "llm_model_name", "qwen2.5:7b")
        self.base_url = getattr(config, "ollama_host", "http://localhost:11434")
        self.use_vision = bool(getattr(config, "summary_use_vision", False))
        
        # 视觉模型名称
        self.vlm_model_name = getattr(config, "summary_vlm_model_name", "") or self.model_name
        
        # 确定使用的模型
        tgt_model = self.vlm_model_name if self.use_vision else self.model_name
        
        # 初始化基础 LLM (默认低温，用于单次推理)
        self.llm = Ollama(base_url=self.base_url, model=tgt_model, temperature=0.1)
        
        # [新增] 投票配置
        # 如果觉得 3 次太慢，可以改成 1 (关闭投票)
        # 建议设置为 3，性价比最高
        self.sc_samples = 3  

    def _encode_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return ""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[SummaryAgent] Error encoding image {image_path}: {e}")
            return ""

    def summarize(self, problems: Dict[str, Any], shot_qids: List[str], qid: str, sum_question: str) -> Tuple[str, List[str]]:
        
        # 1. 解析图片
        image_b64s = []
        if self.use_vision:
            paths = re.findall(r"(?:Images attached \(paths\):|/)[^\s\n]+\.(?:png|jpg|jpeg|webp|bmp)", sum_question)
            valid_paths = list(set([p for p in paths if os.path.exists(p)]))
            for p in valid_paths:
                b64 = self._encode_image(p)
                if b64: image_b64s.append(b64)

        # 2. 构造 CoT Prompt
        prompt = (
            "You are a scientific reasoning assistant. Answer the multiple-choice question strictly based on the logic and facts provided.\n\n"
            "CRITICAL INSTRUCTION:\n"
            "The retrieved evidence may contain 'Similar Questions' to show the solving method.\n"
            "**DO NOT COPY the answer from a similar question directly**.\n"
            "Instead, **learn the solving method** from the evidence and apply it to the current question.\n\n"
            "RESPONSE FORMAT:\n"
            "Reasoning: [Think step-by-step. Compare the current question with the evidence.]\n"
            "Answer: [Return ONLY the correct option letter, e.g., A, B, C, D, or E]\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{sum_question}\n"
            "--- END CONTEXT ---\n"
        )

        # 3. [核心] 自洽性投票 (Self-Consistency)
        # 只有在有证据的情况下才值得投票，纯盲猜投票意义不大
        answers = []
        raw_outputs = []
        
        # 动态调整温度：如果只跑1次，用低温(0.1)；如果跑多次，用高温(0.7)以增加多样性
        run_temperature = 0.7 if self.sc_samples > 1 else 0.1
        
        # 临时绑定参数 (Ollama 支持 bind)
        llm_engine = self.llm.bind(temperature=run_temperature)

        print(f"[Summary] Running Self-Consistency Voting (samples={self.sc_samples}, temp={run_temperature})...")

        for i in range(self.sc_samples):
            try:
                if self.use_vision and image_b64s:
                    out = llm_engine.invoke(prompt, images=image_b64s)
                else:
                    out = llm_engine.invoke(prompt)
                
                ans = _extract_choice_letter(str(out))
                answers.append(ans)
                raw_outputs.append(f"--- Sample {i+1} (Ans: {ans}) ---\n{out[:200]}...")
                
            except Exception as e:
                print(f"[Summary] Error in sample {i}: {e}")
                answers.append("C") # 兜底

        # 4. 统计票数
        vote_counts = Counter(answers)
        # 获取票数最多的答案
        final_ans, count = vote_counts.most_common(1)[0]
        
        # 5. 构造返回日志
        confidence_msg = f"Vote Result: {dict(vote_counts)} -> Winner: {final_ans}"
        if count < self.sc_samples:
             confidence_msg += " (⚠️ Disagreement detected - SC helped!)"
        else:
             confidence_msg += " (Unanimous)"

        msgs = [
            f"Summary Model: {self.llm.model}",
            f"Self-Consistency: {confidence_msg}",
            f"Prompt Snippet: ...{prompt[-300:]}", 
            f"Raw Outputs Snippet:\n" + "\n".join(raw_outputs),
            f"Final Decision: {final_ans}"
        ]
        
        return final_ans, msgs