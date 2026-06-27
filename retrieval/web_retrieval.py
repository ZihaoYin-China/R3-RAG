import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple


def _opensearch_web_search(
    host: str,
    api_key: str,
    query: str,
    top_k: int = 10,
    workspace_name: str = "default",
    service_id: str = "ops-web-search-001",
    content_type: str = "snippet",  # "snippet" | "summary"
    query_rewrite: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"{host.rstrip('/')}/v3/openapi/workspaces/{workspace_name}/web-search/{service_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": int(top_k),
        "query_rewrite": bool(query_rewrite),
        "content_type": content_type,
    }
    if history:
        payload["history"] = history

    s = requests.Session()
    s.trust_env = False
    r = s.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


class WebRetrieval:
    """
    输出格式保持与你原来的 Serper 版本一致：
      List[str], 每条是 "Title: ...\nSnippet: ...\nURL: ..."
    """

    # 你也可以通过环境变量注入关键词（不改 main.py）：
    #   WEB_COARSE_KW="keyword1,keyword2"
    #   WEB_FINE_KW="detail1,detail2"
    #   WEB_COARSE_MODE="any|all"
    #   WEB_FINE_MODE="any|all"
    #   WEB_FINE_MIN_OVERLAP="2"
    #   WEB_FINE_MIN_KW_HITS="1"
    #   WEB_FINE_MIN_LEN="40"

    _STOPWORDS_EN = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
        "is", "are", "was", "were", "be", "as", "at", "it", "this", "that", "from",
    }
    _STOPWORDS_ZH = {"什么", "怎么", "如何", "为什么", "是不是", "以及", "一个", "一些", "我们", "你们", "他们", "它们"}

    def __init__(self, config):
        self.config = config
        self.results: List[str] = []

    # ---------- helper: parse keywords ----------
    @staticmethod
    def _parse_keywords(x: Any) -> List[str]:
        if not x:
            return []
        if isinstance(x, (list, tuple)):
            kws = [str(i).strip() for i in x if str(i).strip()]
            return kws
        if isinstance(x, str):
            # 支持逗号/分号/换行/空格分隔
            parts = re.split(r"[,\n;，；\s]+", x)
            kws = [p.strip() for p in parts if p.strip()]
            return kws
        return [str(x).strip()] if str(x).strip() else []

    @staticmethod
    def _norm_text(s: str) -> str:
        return (s or "").strip().lower()

    def _get_coarse_keywords(self) -> List[str]:
        return self._parse_keywords(
            getattr(self.config, "web_coarse_keywords", None)
            or os.environ.get("WEB_COARSE_KW")
        )

    def _get_fine_keywords(self) -> List[str]:
        return self._parse_keywords(
            getattr(self.config, "web_fine_keywords", None)
            or os.environ.get("WEB_FINE_KW")
        )

    def _get_modes_and_thresholds(self) -> Tuple[str, str, int, int, int]:
        coarse_mode = (
            getattr(self.config, "web_coarse_mode", None)
            or os.environ.get("WEB_COARSE_MODE")
            or "any"
        ).lower()  # any|all
        fine_mode = (
            getattr(self.config, "web_fine_mode", None)
            or os.environ.get("WEB_FINE_MODE")
            or "any"
        ).lower()  # any|all

        fine_min_overlap = int(
            getattr(self.config, "web_fine_min_overlap", None)
            or os.environ.get("WEB_FINE_MIN_OVERLAP")
            or 1
        )
        fine_min_kw_hits = int(
            getattr(self.config, "web_fine_min_kw_hits", None)
            or os.environ.get("WEB_FINE_MIN_KW_HITS")
            or 0
        )
        fine_min_len = int(
            getattr(self.config, "web_fine_min_len", None)
            or os.environ.get("WEB_FINE_MIN_LEN")
            or 0
        )
        return coarse_mode, fine_mode, fine_min_overlap, fine_min_kw_hits, fine_min_len

    # ---------- helper: query tokenization for fine-grained ----------
    def _query_terms(self, query: str) -> List[str]:
        q = self._norm_text(query)
        # 英文/数字 token + 中文连续片段(>=2)
        terms = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", q)
        filtered = []
        for t in terms:
            if t in self._STOPWORDS_EN or t in self._STOPWORDS_ZH:
                continue
            if len(t) <= 1:
                continue
            filtered.append(t)
        # 去重但保持顺序
        seen = set()
        out = []
        for t in filtered:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # ---------- coarse filter ----------
    def _coarse_pass(self, item: Dict[str, Any], coarse_kws: List[str], mode: str) -> bool:
        if not coarse_kws:
            return True  # 没配关键词则不粗筛
        title = item.get("title") or item.get("tilte") or ""
        snippet = item.get("snippet") or item.get("content") or ""
        text = self._norm_text(f"{title}\n{snippet}")

        hits = [(kw.lower() in text) for kw in coarse_kws]
        if mode == "all":
            return all(hits)
        return any(hits)

    # ---------- fine filter + scoring ----------
    def _fine_score_and_pass(
        self,
        item: Dict[str, Any],
        query_terms: List[str],
        fine_kws: List[str],
        fine_mode: str,
        min_overlap: int,
        min_kw_hits: int,
        min_len: int,
    ) -> Tuple[bool, float]:
        title = item.get("title") or item.get("tilte") or ""
        snippet = item.get("snippet") or item.get("content") or ""
        text = self._norm_text(f"{title}\n{snippet}")

        if min_len > 0 and len(snippet or "") < min_len:
            return False, 0.0

        # 1) 细节：query 词项覆盖（越多越好）
        overlap = 0
        for t in query_terms:
            if t in text:
                overlap += 1

        # 2) 细节关键词命中
        kw_hits = 0
        if fine_kws:
            hit_bools = [(kw.lower() in text) for kw in fine_kws]
            kw_hits = sum(1 for b in hit_bools if b)
            if fine_mode == "all" and not all(hit_bools):
                return False, 0.0
            if fine_mode == "any" and kw_hits == 0:
                # 如果你希望 fine_kws 是“强约束”，就保留这行；
                # 如果想让 fine_kws 只是加分项，把这行注释掉即可。
                pass

        if overlap < min_overlap:
            return False, 0.0
        if kw_hits < min_kw_hits:
            return False, 0.0

        # 打分：词项覆盖为主，细节关键词加权，snippet 长度做轻微偏好
        score = float(overlap) + 2.0 * float(kw_hits) + 0.001 * float(len(snippet or ""))
        return True, score

    def find_top_k(self, query: str) -> List[str]:
        k = int(getattr(self.config, "top_k", 4))

        host = (
            getattr(self.config, "opensearch_host", None)
            or os.environ.get("OS_HOST")
            or os.environ.get("OPENSEARCH_HOST")
        )
        api_key = (
            getattr(self.config, "opensearch_api_key", None)
            or os.environ.get("OS_KEY")
            or os.environ.get("OPENSEARCH_API_KEY")
            or getattr(self.config, "serper_api_key", None)
        )
        if not host or not api_key:
            self.results = []
            return self.results

        workspace_name = (
            getattr(self.config, "opensearch_workspace_name", None)
            or os.environ.get("OS_WS")
            or "default"
        )
        service_id = (
            getattr(self.config, "opensearch_service_id", None)
            or os.environ.get("OS_SVC")
            or "ops-web-search-001"
        )
        content_type = getattr(self.config, "opensearch_content_type", "snippet")
        query_rewrite = bool(getattr(self.config, "opensearch_query_rewrite", True))
        history = getattr(self.config, "opensearch_history", None)

        try:
            resp = _opensearch_web_search(
                host=host,
                api_key=api_key,
                query=query,
                top_k=max(k, 10),  # 多取一点，方便筛选后还能剩 k 条
                workspace_name=workspace_name,
                service_id=service_id,
                content_type=content_type,
                query_rewrite=query_rewrite,
                history=history,
                timeout=30,
            )
            result = resp.get("result") or {}
            search_results = result.get("search_result") or []
        except Exception as e:
            print(f"[WARN] OpenSearch web-search failed, fallback to empty results: {e}")
            search_results = []

        # ====== 粗筛 + 细筛 ======
        coarse_kws = self._get_coarse_keywords()
        fine_kws = self._get_fine_keywords()
        coarse_mode, fine_mode, fine_min_overlap, fine_min_kw_hits, fine_min_len = self._get_modes_and_thresholds()

        # 1) coarse: keyword filter
        coarse_candidates = [it for it in search_results if self._coarse_pass(it, coarse_kws, coarse_mode)]
        # 如果粗筛后一个不剩，回退到原始结果（避免“全空”）
        candidates = coarse_candidates if coarse_candidates else search_results

        # 2) fine: detail filter + scoring
        q_terms = self._query_terms(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for it in candidates:
            ok, score = self._fine_score_and_pass(
                it,
                query_terms=q_terms,
                fine_kws=fine_kws,
                fine_mode=fine_mode,
                min_overlap=fine_min_overlap,
                min_kw_hits=fine_min_kw_hits,
                min_len=fine_min_len,
            )
            if ok:
                scored.append((score, it))

        # 如果细筛后空了，再次回退：仅做重排不过滤（避免空）
        if not scored:
            for it in candidates:
                title = it.get("title") or it.get("tilte") or ""
                snippet = it.get("snippet") or it.get("content") or ""
                text = self._norm_text(f"{title}\n{snippet}")
                overlap = sum(1 for t in q_terms if t in text)
                kw_hits = sum(1 for kw in fine_kws if kw.lower() in text) if fine_kws else 0
                score = float(overlap) + 2.0 * float(kw_hits) + 0.001 * float(len(snippet or ""))
                scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        final_items = [it for _, it in scored[:k]]

        passages: List[str] = []
        for item in final_items:
            title = item.get("title") or item.get("tilte") or ""
            link = item.get("link") or ""
            snippet = item.get("snippet") or item.get("content") or ""
            passages.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        self.results = passages
        return passages
