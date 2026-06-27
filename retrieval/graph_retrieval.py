# /Yin_zi_hao/code/HMRAG-main5/retrieval/graph_retrieval.py
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import requests
from nano_vectordb import NanoVectorDB

from retrieval.base_retrieval import BaseRetrieval


def _find_one(root: str, candidates: List[str], contains: Optional[str] = None) -> Optional[str]:
    """Find file under root (non-recursive first, then recursive)."""
    for nm in candidates:
        p = os.path.join(root, nm)
        if os.path.exists(p):
            return p

    if contains and os.path.isdir(root):
        for fn in os.listdir(root):
            if contains in fn and fn.endswith(".json"):
                p = os.path.join(root, fn)
                if os.path.exists(p):
                    return p

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
    host = host.rstrip("/")

    # /api/embed
    try:
        url = f"{host}/api/embed"
        r = requests.post(url, json={"model": model, "input": text}, timeout=timeout)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
    except Exception:
        pass

    # /api/embeddings
    url = f"{host}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embeddings response missing 'embedding': keys={list(data.keys())}")
    return data["embedding"]


def _relation_endpoints(rel_obj: Any, entity_id_set: Set[str]) -> List[str]:
    """
    尽量从 relation chunk 对象里抽取“关系两端实体 id”
    - 先看常见 metadata 字段
    - 再用正则在文本/json dump 里找类似 entity-xxxx/ent-xxxx/node-xxxx 的 token，再过滤到 entity_id_set
    """
    eps: List[str] = []

    if isinstance(rel_obj, dict):
        # 常见字段（不同 LightRAG 版本可能不一样）
        key_pairs = [
            ("source_id", "target_id"),
            ("src_id", "dst_id"),
            ("head_id", "tail_id"),
            ("from_id", "to_id"),
            ("source", "target"),  # 有时这里就是 id
        ]
        for a, b in key_pairs:
            va, vb = rel_obj.get(a), rel_obj.get(b)
            if isinstance(va, str) and va in entity_id_set:
                eps.append(va)
            if isinstance(vb, str) and vb in entity_id_set:
                eps.append(vb)

        # metadata 包一层的情况
        meta = rel_obj.get("metadata", None)
        if isinstance(meta, dict):
            for a, b in key_pairs:
                va, vb = meta.get(a), meta.get(b)
                if isinstance(va, str) and va in entity_id_set:
                    eps.append(va)
                if isinstance(vb, str) and vb in entity_id_set:
                    eps.append(vb)

    # 正则兜底：在字符串里找 “entity-<hex32> / ent-<hex32> / node-<hex32>”
    # 再用 entity_id_set 过滤，避免误匹配
    s = _extract_text_from_chunk_obj(rel_obj)
    if not s:
        try:
            s = json.dumps(rel_obj, ensure_ascii=False)
        except Exception:
            s = str(rel_obj)

    # 允许 id 前缀多样
    pat = re.compile(r'\b(?:entity|ent|node)[-_][0-9a-f]{32}\b', re.IGNORECASE)
    for m in pat.findall(s):
        # 统一大小写匹配：entity_id_set 里通常是原样；这里直接先尝试原串/小写
        if m in entity_id_set:
            eps.append(m)
        else:
            ml = m.lower()
            if ml in entity_id_set:
                eps.append(ml)

    # 去重保持顺序
    seen = set()
    out = []
    for x in eps:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class GraphRetrieval(BaseRetrieval):
    """
    仍然基于 NanoVectorDB（不走 LightRAG.query），但实现“1-hop local + 1 global”：
      - local: seed entities + 与 seed 直接相连的一跳 relations（从 rel pool 里过滤出来）
      - global: 额外补 1 条全局最相关 relation（不要求连接 seed）
    """

    def __init__(self, config):
        self.config = config

        kb_workdir = getattr(config, "lightrag_workdir", None) or getattr(config, "working_dir", None) or "."
        self.kb_workdir = kb_workdir
        print(f"[GraphRetrieval] using kb_workdir = {self.kb_workdir}")

        self.vdb_entities_path = _find_one(
            self.kb_workdir, candidates=["vdb_entities.json"], contains="vdb_entities"
        )
        if not self.vdb_entities_path:
            raise FileNotFoundError(f"[GraphRetrieval] cannot find vdb_entities*.json under {self.kb_workdir}")
        print(f"[GraphRetrieval] vdb_entities_path = {self.vdb_entities_path}")

        self.vdb_relations_path = _find_one(
            self.kb_workdir,
            candidates=["vdb_relationships.json", "vdb_relations.json"],
            contains="vdb_relationship",
        )
        if not self.vdb_relations_path:
            raise FileNotFoundError(f"[GraphRetrieval] cannot find vdb_relationships*.json under {self.kb_workdir}")
        print(f"[GraphRetrieval] vdb_relations_path = {self.vdb_relations_path}")

        self.entity_chunks_path = _find_one(
            self.kb_workdir, candidates=["kv_store_entity_chunks.json"], contains="entity_chunks"
        )
        if not self.entity_chunks_path:
            raise FileNotFoundError(f"[GraphRetrieval] cannot find kv_store_entity_chunks*.json under {self.kb_workdir}")
        print(f"[GraphRetrieval] entity_chunks_path = {self.entity_chunks_path}")

        self.relation_chunks_path = _find_one(
            self.kb_workdir, candidates=["kv_store_relation_chunks.json"], contains="relation_chunks"
        )
        if not self.relation_chunks_path:
            raise FileNotFoundError(f"[GraphRetrieval] cannot find kv_store_relation_chunks*.json under {self.kb_workdir}")
        print(f"[GraphRetrieval] relation_chunks_path = {self.relation_chunks_path}")

        # init NanoVectorDBs
        ent_dim = _detect_embedding_dim(self.vdb_entities_path, default_dim=768)
        rel_dim = _detect_embedding_dim(self.vdb_relations_path, default_dim=ent_dim)
        print(f"[GraphRetrieval] detected embedding_dim: ent={ent_dim}, rel={rel_dim}")

        self.ent_db = NanoVectorDB(embedding_dim=ent_dim, metric="cosine", storage_file=self.vdb_entities_path)
        self.rel_db = NanoVectorDB(embedding_dim=rel_dim, metric="cosine", storage_file=self.vdb_relations_path)

        # load chunks kv
        with open(self.entity_chunks_path, "r", encoding="utf-8") as f:
            ec = json.load(f)
        self.entity_chunks: Dict[str, Any] = ec if isinstance(ec, dict) else {str(i): v for i, v in enumerate(ec)}

        with open(self.relation_chunks_path, "r", encoding="utf-8") as f:
            rc = json.load(f)
        self.relation_chunks: Dict[str, Any] = rc if isinstance(rc, dict) else {str(i): v for i, v in enumerate(rc)}

        print(f"[GraphRetrieval] entity_chunks loaded = {len(self.entity_chunks)}")
        print(f"[GraphRetrieval] relation_chunks loaded = {len(self.relation_chunks)}")

        # build endpoints index for 1-hop filtering
        self.entity_id_set: Set[str] = set(self.entity_chunks.keys())
        self.rel_endpoints: Dict[str, List[str]] = {}
        bad = 0
        for rel_key, meta in self.relation_chunks.items():
            if not isinstance(rel_key, str):
                bad += 1
                continue
            # ScienceQA KB relation key format: "src<SEP>tgt"
            if "<SEP>" in rel_key:
                src, tgt = rel_key.split("<SEP>", 1)
                src, tgt = src.strip(), tgt.strip()
            else:
                bad += 1
                continue
            if not src or not tgt:
                bad += 1
                continue
            rid = meta.get("_id") if isinstance(meta, dict) else None
            rid = rid or rel_key
            self.rel_endpoints[rid] = [src, tgt]
        print(f"[GraphRetrieval] rel_endpoints built = {len(self.rel_endpoints)}/{len(self.relation_chunks)} (bad={bad})")
        # embedding config
        self.ollama_host = getattr(config, "ollama_host", "http://localhost:11434")
        self.embed_model = getattr(config, "lightrag_embed_model", "nomic-embed-text")
        self.embed_timeout = float(getattr(config, "ollama_timeout", 120))

        self._emb_cache: Dict[str, np.ndarray] = {}

    def find_top_k(self, query: str) -> str:
        if not query:
            return ""

        debug = bool(getattr(self.config, "debug", False))

        top_k = int(getattr(self.config, "graph_top_k", getattr(self.config, "top_k", 4)))
        top_k = max(2, top_k)  # 至少留 1 条 global

        # 约定：1 条 global，其余当 local（seed ent + 1-hop rel）
        global_k = int(getattr(self.config, "graph_global_k", 1))
        global_k = max(1, min(global_k, top_k - 1))
        local_budget = top_k - global_k

        # local budget 切成：seed entities + local relations
        seed_ent_k = int(getattr(self.config, "graph_seed_ent_k", max(1, local_budget // 2)))
        local_rel_k = int(getattr(self.config, "graph_local_rel_k", max(1, local_budget - seed_ent_k)))

        if debug:
            q_show = query.replace("\n", "\\n")[:160]
            print(
                f"[GraphRetrieval] query = {q_show} "
                f"(top_k={top_k}, seed_ent_k={seed_ent_k}, local_rel_k={local_rel_k}, global_k={global_k})"
            )

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
                print(f"[GraphRetrieval] embed failed: {repr(e)}")
                return ""
            qvec = np.asarray(emb, dtype=np.float32)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            self._emb_cache[query] = qvec

        # 1) seed entities（局部起点）
        try:
            ent_res = self.ent_db.query(qvec.tolist(), top_k=seed_ent_k)
        except TypeError:
            ent_res = self.ent_db.query(qvec.tolist(), seed_ent_k)

        seed_ids: List[str] = []
        if isinstance(ent_res, list):
            for it in ent_res:
                if isinstance(it, dict) and "__id__" in it:
                    seed_ids.append(str(it["__id__"]))
        seed_set = set(seed_ids)

        # 2) relations pool（用于筛 local 1-hop + 选 global）
        # pool 要大一点：否则可能筛不出与 seed 相连的关系
        rel_pool_k = int(getattr(self.config, "graph_rel_pool_k", max(50, local_rel_k * 10)))
        try:
            rel_pool = self.rel_db.query(qvec.tolist(), top_k=rel_pool_k)
        except TypeError:
            rel_pool = self.rel_db.query(qvec.tolist(), rel_pool_k)

        # 2.1 local 1-hop relations：必须与 seed 有端点交集
        local_rels: List[Dict[str, Any]] = []
        used_rel_ids: Set[str] = set()

        if isinstance(rel_pool, list):
            for it in rel_pool:
                if not isinstance(it, dict):
                    continue
                rid = str(it.get("__id__", ""))
                if not rid or rid in used_rel_ids:
                    continue
                eps = self.rel_endpoints.get(rid, [])
                if eps and (set(eps) & seed_set):
                    local_rels.append(it)
                    used_rel_ids.add(rid)
                if len(local_rels) >= local_rel_k:
                    break

        # 2.2 global relations：从 pool 里再挑 global_k 条（不要求连接 seed）
        global_rels: List[Dict[str, Any]] = []
        if isinstance(rel_pool, list):
            for it in rel_pool:
                if not isinstance(it, dict):
                    continue
                rid = str(it.get("__id__", ""))
                if not rid or rid in used_rel_ids:
                    continue
                global_rels.append(it)
                used_rel_ids.add(rid)
                if len(global_rels) >= global_k:
                    break

        lines: List[str] = []

        # ---- format: local seed entities ----
        if isinstance(ent_res, list):
            for rank, it in enumerate(ent_res, start=1):
                if not isinstance(it, dict):
                    continue
                eid = str(it.get("__id__", ""))
                metric = it.get("__metrics__", None)

                chunk_obj = self.entity_chunks.get(eid, None)
                text = _extract_text_from_chunk_obj(chunk_obj).strip() if chunk_obj is not None else ""
                if not text:
                    text = str(it.get("content", "")).strip()
                if not text:
                    continue

                text = text.replace("\n", " ").strip()
                if len(text) > 350:
                    text = text[:350] + " ..."

                if metric is None:
                    lines.append(f"[G-LOCAL-ENT-{rank}] (id={eid}) {text}")
                else:
                    lines.append(f"[G-LOCAL-ENT-{rank}] (id={eid}, metric={float(metric):.6f}) {text}")

        # ---- format: local 1-hop relations + neighbor entities ----
        # neighbor entity 可选补充（更像“一跳”）
        add_neighbors = bool(getattr(self.config, "graph_add_neighbors", True))
        neighbor_cap = int(getattr(self.config, "graph_neighbor_cap", 3))
        neighbor_added: Set[str] = set()

        for rank, it in enumerate(local_rels, start=1):
            rid = str(it.get("__id__", ""))
            metric = it.get("__metrics__", None)

            robj = self.relation_chunks.get(rid, None)
            text = _extract_text_from_chunk_obj(robj).strip() if robj is not None else ""
            if not text:
                text = str(it.get("content", "")).strip()
            if not text:
                continue

            text = text.replace("\n", " ").strip()
            if len(text) > 400:
                text = text[:400] + " ..."

            if metric is None:
                lines.append(f"[G-LOCAL-REL-{rank}] (id={rid}) {text}")
            else:
                lines.append(f"[G-LOCAL-REL-{rank}] (id={rid}, metric={float(metric):.6f}) {text}")

            # neighbor entities（取关系端点里非 seed 的那个）
            if add_neighbors and len(neighbor_added) < neighbor_cap:
                eps = self.rel_endpoints.get(rid, [])
                for e in eps:
                    if e in seed_set:
                        continue
                    if e in neighbor_added:
                        continue
                    eobj = self.entity_chunks.get(e, None)
                    et = _extract_text_from_chunk_obj(eobj).strip() if eobj is not None else ""
                    if not et:
                        continue
                    et = et.replace("\n", " ").strip()
                    if len(et) > 260:
                        et = et[:260] + " ..."
                    neighbor_added.add(e)
                    lines.append(f"[G-1H-NEI] (id={e}) {et}")
                    if len(neighbor_added) >= neighbor_cap:
                        break

        # ---- format: global relation(s) ----
        for rank, it in enumerate(global_rels, start=1):
            rid = str(it.get("__id__", ""))
            metric = it.get("__metrics__", None)

            robj = self.relation_chunks.get(rid, None)
            text = _extract_text_from_chunk_obj(robj).strip() if robj is not None else ""
            if not text:
                text = str(it.get("content", "")).strip()
            if not text:
                continue

            text = text.replace("\n", " ").strip()
            if len(text) > 400:
                text = text[:400] + " ..."

            if metric is None:
                lines.append(f"[G-GLOBAL-{rank}] (id={rid}) {text}")
            else:
                lines.append(f"[G-GLOBAL-{rank}] (id={rid}, metric={float(metric):.6f}) {text}")

        return "\n".join(lines)
