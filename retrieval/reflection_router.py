import json
import re
from typing import Set, Tuple, List
from langchain_community.llms import Ollama

class ReflectionRouter:
    """
    【独立模块】本地反思路由 (Local Reflection Router)
    职责：当检索失败时，分析原因，并决定：
    1. 如何改写查询 (Query Rewriting) 以更精准匹配本地知识库。
    2. 使用哪些本地检索源 (Vector/Graph/MM) 进行重试。
    """
    def __init__(self, config):
        self.config = config
        # 初始化反思用的大模型 (建议用 14b 以保证逻辑能力)
        self.model_name = getattr(config, "llm_model_name", "qwen3:32b")
        self.ollama_host = getattr(config, "ollama_host", "http://localhost:11434")
        
        print(f"[ReflectionRouter] Initializing Brain: {self.model_name}...")
        try:
            self.llm = Ollama(
                base_url=self.ollama_host, 
                model=self.model_name, 
                temperature=0.1 # 低温以保证 JSON 格式稳定
            )
        except Exception as e:
            print(f"[ERROR] Reflection Router init failed: {e}")
            self.llm = None

    def reflect(self, question: str, current_modes: Set[str], fail_reason: str, available_modes: Set[str]) -> Tuple[str, Set[str], str]:
        """
        执行反思逻辑。
        Returns: (new_query, new_modes, thought_process)
        """
        if not self.llm:
            return question, current_modes, "LLM not initialized"

        # ------------------------------------------------------------------
        # [升级] Prompt：引入 DIAGNOSE -> REWRITE -> ROUTE 三步思维链
        # ------------------------------------------------------------------
        prompt = (
            f"You are a Search Strategy Optimizer for a Closed-Domain Local Database.\n"
            f"The previous search attempt FAILED.\n"
            f"Failure Reason: \"{fail_reason}\".\n"
            f"Original Query: \"{question}\"\n"
            f"Previous Modes: {list(current_modes)}\n\n"
            "Your Task: Improve the search strategy using ONLY local resources (No Internet).\n"
            "Available Resources:\n"
            "- 'vector': Unstructured text chunks (Best for descriptions, context, fuzzy match).\n"
            "- 'graph': Knowledge Graph (Best for definitions, specific entity relations).\n"
            "- 'mm': Image captions & similarity (Best for visual questions).\n\n"
            "--------------------------------------------------\n"
            "STEP 1: DIAGNOSE\n"
            "Why did it fail? (e.g., 'Keywords too vague', 'Too much noise', 'Needs specific entity for Graph', 'Visual dependence ignored')\n\n"
            "STEP 2: REWRITE QUERY (Critical)\n"
            "Rewrite the query to maximize retrieval success:\n"
            "   - If switching to 'graph': Extract CORE ENTITIES only (e.g., 'Mitochondria function'). Graph hates long sentences.\n"
            "   - If keeping 'vector': Simplify or Expand with SYNONYMS (e.g., 'Mitochondria OR power house').\n"
            "   - If visual: Focus on visual attributes.\n\n"
            "STEP 3: ROUTE\n"
            "Select the best new modes based on the rewritten query.\n"
            "--------------------------------------------------\n\n"
            "Output JSON ONLY:\n"
            "{\n"
            "  \"thought\": \"Brief diagnosis and reasoning...\",\n"
            "  \"new_query\": \"The rewritten optimized query\",\n"
            "  \"new_modes\": [\"vector\", \"graph\"]\n"
            "}"
        )

        try:
            resp = self.llm.invoke(prompt)
            
            # JSON 清洗
            s = resp.strip()
            # 移除可能存在的 markdown 代码块标记
            if "```json" in s: 
                s = s.split("```json")[1].split("```")[0]
            elif "```" in s: 
                s = s.split("```")[1].split("```")[0]
            
            # 有时候模型还是会返回非标准 JSON，做简单的清理
            s = s.strip()
            
            data = json.loads(s)
            
            # --- 模式解析与过滤 ---
            raw_modes = data.get("new_modes", [])
            new_modes = set()
            
            if "hybrid" in raw_modes or "all" in raw_modes:
                new_modes = available_modes.copy()
            else:
                for m in raw_modes:
                    # 映射容错
                    m = m.lower().strip()
                    if m == 'text': m = 'vector'
                    if m == 'kg': m = 'graph'
                    if m == 'vision': m = 'mm'
                    
                    # 关键：只允许本地存在的模式，严防 LLM 幻觉出 'web'
                    if m in available_modes:
                        new_modes.add(m)
            
            # 兜底：如果为空，默认全开
            if not new_modes:
                new_modes = available_modes.copy()
            
            # 获取新 Query，如果模型没返回，则沿用旧的
            new_query = data.get("new_query", question).strip()
            if not new_query: 
                new_query = question
                
            thought = data.get("thought", "Strategy adjusted")
            
            return new_query, new_modes, thought
            
        except Exception as e:
            print(f"[Reflection Error] {e}")
            # 出错时，换个最稳妥的策略：全开，且不改写 Query
            return question, available_modes.copy(), f"Reflection failed ({str(e)}), fallback to all"