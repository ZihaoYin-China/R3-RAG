# /Yin_zi_hao/code/HMRAG-main5/retrieval/vector_retrieval.py
import os
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from nano_vectordb import NanoVectorDB

from retrieval.base_retrieval import BaseRetrieval


def _find_one(root: str, candidates: List[str], contains: Optional[str] = None) -> Optional[str]:
    """Find file under root (non-recursive first, then recursive)."""
    # 1) direct
    for nm in candidates:
        p = os.path.join(root, nm)
        if os.path.exists(p):
            return p

    # 2) contains scan (non-recursive)
    if contains and os.path.isdir(root):
        for fn in os.listdir(root):
            if contains in fn and fn.endswith(".json"):
                p = os.path.join(root, fn)
                if os.path.exists(p):
                    return p

    # 3) recursive
    for dp, _, fns in os.walk(root):
        for nm in candidates:
            if nm in fns:
                return os.path.join(dp, nm)
        if contains:
            for fn in fns:
                if contains in fn and fn.endswith(".json"):
                    return os.path.join(dp, fn)
    return None


def _detect_embedding_dim(vdb_json_path: str, default_dim: int = 768) -> int:
    """Try detect embedding_dim from nano_vectordb json header."""
    try:
        with open(vdb_json_path, "r", encoding="utf-8") as f:
            head = f.read(2_000_000)
        m = re.search(r'"embedding_dim"\s*:\s*(\d+)', head)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return int(default_dim)


def _extract_text_from_chunk_obj(obj: Any) -> str:
    """kv_store_text_chunks.json 里 value 可能是 str / dict / list 等，尽量提取正文。"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ["content", "text", "chunk", "data", "value", "page_content"]:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)
    if isinstance(obj, list):
        parts = [x for x in obj if isinstance(x, str)]
        if parts:
            return "\n".join(parts)
        return str(obj)
    return str(obj)


def _ollama_embed_sync(host: str, model: str, text: str, timeout: float = 60.0) -> List[float]:
    """
    同步获取 Ollama embedding。
    兼容两种 API：
      - POST /api/embed       {"model":..., "input": "..."}  -> {"embeddings":[[...]]}
      - POST /api/embeddings  {"model":..., "prompt":"..."}  -> {"embedding":[...]}
    """
    host = host.rstrip("/")

    # 1) /api/embed（新）
    try:
        url = f"{host}/api/embed"
        r = requests.post(url, json={"model": model, "input": text}, timeout=timeout)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
    except Exception:
        pass

    # 2) /api/embeddings（旧）
    url = f"{host}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embeddings response missing 'embedding': keys={list(data.keys())}")
    return data["embedding"]


class VectorRetrieval(BaseRetrieval):
    """
    直接用 nano_vectordb 的 vdb_chunks.json 做向量检索：
      - NanoVectorDB load vdb_chunks.json
      - query -> 返回 top_k chunk dict（含 __id__/content/__metrics__）
      - 再用 kv_store_text_chunks.json 做更稳的文本回填
    """

    def __init__(self, config):
        self.config = config

        kb_workdir = getattr(config, "lightrag_workdir", None) or getattr(config, "working_dir", None) or "."
        self.kb_workdir = kb_workdir
        print(f"[VectorRetrieval] using kb_workdir = {self.kb_workdir}")

        self.vdb_chunks_path = _find_one(
            self.kb_workdir,
            candidates=["vdb_chunks.json"],
            contains="vdb_chunks",
        )
        if not self.vdb_chunks_path:
            raise FileNotFoundError(f"[VectorRetrieval] cannot find vdb_chunks*.json under {self.kb_workdir}")
        print(f"[VectorRetrieval] vdb_chunks_path = {self.vdb_chunks_path}")

        self.text_chunks_path = _find_one(
            self.kb_workdir,
            candidates=["kv_store_text_chunks.json", "text_chunks.json"],
            contains="text_chunks",
        )
        if not self.text_chunks_path:
            raise FileNotFoundError(f"[VectorRetrieval] cannot find kv_store_text_chunks*.json under {self.kb_workdir}")
        print(f"[VectorRetrieval] text_chunks_path = {self.text_chunks_path}")

        # detect dim + init NanoVectorDB
        dim = _detect_embedding_dim(self.vdb_chunks_path, default_dim=768)
        print(f"[VectorRetrieval] detected embedding_dim = {dim}")
        self.vdb = NanoVectorDB(embedding_dim=dim, metric="cosine", storage_file=self.vdb_chunks_path)
        self.embedding_dim = dim

        # load text chunks kv
        with open(self.text_chunks_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        self.text_chunks: Dict[str, Any] = tc if isinstance(tc, dict) else {str(i): v for i, v in enumerate(tc)}
        print(f"[VectorRetrieval] text_chunks loaded = {len(self.text_chunks)}")

        # embedding config
        self.ollama_host = getattr(config, "ollama_host", "http://localhost:11434")
        self.embed_model = getattr(config, "lightrag_embed_model", "nomic-embed-text")
        self.embed_timeout = float(getattr(config, "ollama_timeout", 120))

        # cache
        self._emb_cache: Dict[str, np.ndarray] = {}

    def find_top_k(self, query: str) -> str:
        if not query:
            return ""

        debug = bool(getattr(self.config, "debug", False))
        top_k = int(getattr(self.config, "top_k", 4))
        top_k = max(1, top_k)

        if debug:
            q_show = query.replace("\n", "\\n")[:160]
            print(f"[VectorRetrieval] query = {q_show}")

        # embed query
        if query in self._emb_cache:
            qvec = self._emb_cache[query]
        else:
            try:
                emb = _ollama_embed_sync(
                    host=self.ollama_host,
                    model=self.embed_model,
                    text=query,
                    timeout=self.embed_timeout,
                )
            except Exception as e:
                print(f"[VectorRetrieval] embed failed: {repr(e)}")
                return ""
            qvec = np.asarray(emb, dtype=np.float32)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            self._emb_cache[query] = qvec

        # query NanoVectorDB
        try:
            results = self.vdb.query(qvec.tolist(), top_k=top_k)
        except TypeError:
            # 兼容老签名
            results = self.vdb.query(qvec.tolist(), top_k)

        if not isinstance(results, list) or not results:
            return ""

        lines: List[str] = []
        for rank, it in enumerate(results, start=1):
            if not isinstance(it, dict):
                continue
            cid = str(it.get("__id__", ""))
            metric = it.get("__metrics__", None)

            # 优先用 kv_store_text_chunks 回填（更稳），没有再用 vdb 自带 content
            chunk_obj = self.text_chunks.get(cid, None)
            text = _extract_text_from_chunk_obj(chunk_obj).strip() if chunk_obj is not None else ""
            if not text:
                text = str(it.get("content", "")).strip()

            if not text:
                continue

            text = text.replace("\n", " ").strip()
            if len(text) > 400:
                text = text[:400] + " ..."

            if metric is None:
                lines.append(f"[VEC-{rank}] (chunk={cid}) {text}")
            else:
                lines.append(f"[VEC-{rank}] (chunk={cid}, metric={float(metric):.6f}) {text}")

        return "\n".join(lines)
