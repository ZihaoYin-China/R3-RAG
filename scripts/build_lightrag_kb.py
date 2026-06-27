#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import asyncio
import inspect
import shutil
import glob
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

try:
    from lightrag.utils import setup_logger
    setup_logger("lightrag", level="INFO")
except Exception:
    pass
# ---------------------------
# Helpers
# ---------------------------

def _safe_lightrag_init(**kwargs):
    """Init LightRAG with only supported kwargs (compat across versions)."""
    sig = inspect.signature(LightRAG.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return LightRAG(**filtered)


def _find_one(root: str, names: List[str]) -> Optional[str]:
    for nm in names:
        p = os.path.join(root, nm)
        if os.path.exists(p):
            return p
    for dp, _, fns in os.walk(root):
        for nm in names:
            if nm in fns:
                return os.path.join(dp, nm)
    return None
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _get_question_and_options(problem: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = problem.get("question") or problem.get("query") or problem.get("text") or ""
    opts = problem.get("choices") or problem.get("options") or []
    if isinstance(opts, dict):
        ordered = []
        for k in ["A", "B", "C", "D", "E"]:
            if k in opts:
                ordered.append(opts[k])
        opts = ordered
    return str(q), [str(x) for x in opts]


def _pack_doc(qid: str, prob: Dict[str, Any]) -> str:
    """
    ⚠️ 为避免数据泄漏：不要写入 answer / lecture / solution。
    只写 Context(hint+caption) + Question + Options。
    """
    q, options = _get_question_and_options(prob)
    hint = str(prob.get("hint") or "").strip()
    cap = str(prob.get("caption") or "").strip()
    context = "\n".join([x for x in [hint, cap] if x]).strip() or "N/A"

    letters = ["A", "B", "C", "D", "E"]
    opt_lines = "\n".join(
        [f"{letters[i]}. {options[i]}" for i in range(min(len(options), len(letters)))]
    )

    doc = (
        f"[QID {qid}]\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{q}\n\n"
        f"Options:\n{opt_lines}\n"
    )
    return doc.strip()


def _maybe_is_coro(obj) -> bool:
    return asyncio.iscoroutine(obj) or asyncio.isfuture(obj)


async def _maybe_await(obj):
    if _maybe_is_coro(obj):
        return await obj
    return obj


async def _init_pipeline_status_compat():
    """
    兼容：PipelineNotInitializedError: pipeline_status not found
    """
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await _maybe_await(initialize_pipeline_status())
    except Exception:
        return


def _clear_workdir(working_dir: str):
    """
    清空已有 KB（graph/vdb/kv/rag_storage 等），保证从零开始重建。
    """
    os.makedirs(working_dir, exist_ok=True)
    patterns = [
        "rag_storage",
        "graph_chunk_entity_relation.graphml",
        "vdb_*.json",
        "kv_*.json",
        "kv_store_*.json",
        "*.graphml",
    ]
    # 删除文件
    for pat in patterns:
        for fp in glob.glob(os.path.join(working_dir, pat)):
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    # 删除目录
    rp = os.path.join(working_dir, "rag_storage")
    if os.path.isdir(rp):
        shutil.rmtree(rp, ignore_errors=True)


async def _call_insert_maybe_async(fn, doc: str, qid_fp: str):
    """
    ✅ 适配你当前 LightRAG 版本的签名：
      insert/ainsert(..., ids=..., file_paths=..., track_id=...)
    统一调用：不依赖 iscoroutinefunction，调用后判断是否 coroutine 再 await。
    """
    # 0) 最推荐：同时传 ids + file_paths（你这版签名明确支持）
    try:
        res = fn(doc, ids=qid_fp, file_paths=qid_fp)
        if _maybe_is_coro(res):
            return await res
        return res
    except TypeError:
        pass

    # 1) 只传 file_paths（复数）
    try:
        res = fn(doc, file_paths=qid_fp)
        if _maybe_is_coro(res):
            return await res
        return res
    except TypeError:
        pass

    # 2) 只传 ids
    try:
        res = fn(doc, ids=qid_fp)
        if _maybe_is_coro(res):
            return await res
        return res
    except TypeError:
        pass

    # 3) second positional（极少数版本用 insert(doc, file_path)）
    try:
        res = fn(doc, qid_fp)
        if _maybe_is_coro(res):
            return await res
        return res
    except TypeError:
        pass

    # 4) fallback（会回到 unknown_source）
    res = fn(doc)
    if _maybe_is_coro(res):
        return await res
    return res


async def _ainsert_one(rag: LightRAG, doc: str, retries: int, sleep_s: float, qid: str, fail_policy: str) -> bool:
    """
    修复 unknown_source：qid 这里传入的是 qid_fp = f"{split}:{qid}"
    """
    insert_async = getattr(rag, "ainsert", None)
    insert_sync = getattr(rag, "insert", None)
    if insert_async is None and insert_sync is None:
        raise RuntimeError("LightRAG has neither ainsert nor insert; version mismatch?")

    for t in range(retries + 1):
        try:
            if insert_async is not None:
                await _call_insert_maybe_async(insert_async, doc, qid)
            else:
                await _call_insert_maybe_async(insert_sync, doc, qid)
            return True
        except Exception as e:
            msg = str(e)
            print(f"[WARN] insert failed file_path={qid} try {t+1}/{retries+1}: {msg[:200]}")
            if t < retries:
                await asyncio.sleep(sleep_s * (t + 1))
                continue
            if fail_policy == "skip":
                return False
            raise


async def main_async(args):
    os.makedirs(args.working_dir, exist_ok=True)

    if args.clear:
        _clear_workdir(args.working_dir)
        print(f"[OK] Cleared working_dir: {args.working_dir}")

    problems_path = _find_one(
        args.data_root,
        [
            "problems.json",
            f"problems_{args.split}.json",
            f"problems_{args.split.lower()}.json",
            "problems_train.json",
            "problems_val.json",
            "problems_test.json",
            "problems_minival.json",
        ],
    )
    if not problems_path:
        raise FileNotFoundError(f"Cannot find problems*.json under: {args.data_root}")

    pid_splits_path = _find_one(args.data_root, ["pid_splits.json"])
    problems = _load_json(problems_path)

    # captions 可选
    if args.caption_file and os.path.exists(args.caption_file):
        try:
            cap_obj = _load_json(args.caption_file)
            captions = cap_obj.get("captions", cap_obj) if isinstance(cap_obj, dict) else {}
            if isinstance(captions, dict):
                for qid in problems:
                    problems[qid]["caption"] = captions.get(str(qid), "")
        except Exception as e:
            print(f"[WARN] Failed to load caption_file: {e}")

    # 选择 qids
    if args.split == "all":
        qids = [str(k) for k in problems.keys()]
    else:
        if pid_splits_path and os.path.exists(pid_splits_path):
            pid = _load_json(pid_splits_path)
            qids = [str(x) for x in (pid.get(args.split, []) or [])]
            if not qids:
                qids = [str(k) for k in problems.keys()]
        else:
            qids = [str(k) for k in problems.keys()]

    if args.limit > 0:
        qids = qids[: args.limit]

    rag = _safe_lightrag_init(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=args.llm_model_name,
        llm_model_kwargs={
            "host": args.ollama_host,
            "options": {"num_ctx": args.num_ctx},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=args.embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=lambda texts: ollama_embed(texts, embed_model=args.embed_model, host=args.ollama_host),
        ),
        llm_model_max_async=args.llm_max_async,
        embedding_func_max_async=args.embed_max_async,
        max_parallel_insert=args.max_parallel_insert,
    )

    init_st = getattr(rag, "initialize_storages", None)
    if init_st is None:
        raise RuntimeError("LightRAG has no initialize_storages(); version mismatch?")
    await _maybe_await(init_st())
    await _init_pipeline_status_compat()

    failed: List[str] = []
    inserted = 0

    batch: List[Tuple[str, str]] = []

    def _flush_batch_now() -> List[Tuple[str, str]]:
        nonlocal batch
        b = batch
        batch = []
        return b

    for qid in tqdm(qids, desc=f"Inserting split={args.split}"):
        prob = problems.get(str(qid))
        if not isinstance(prob, dict):
            continue
        doc = _pack_doc(str(qid), prob)

        if args.batch > 1:
            batch.append((qid, doc))
            if len(batch) < args.batch:
                continue
            to_ins = _flush_batch_now()
        else:
            to_ins = [(qid, doc)]

        for _qid, _doc in to_ins:
            # ✅ 关键：把来源设成 split:qid
            qid_fp = f"{args.split}:{_qid}"
            ok = await _ainsert_one(
                rag=rag,
                doc=_doc,
                retries=args.retries,
                sleep_s=args.retry_sleep,
                qid=qid_fp,
                fail_policy=args.fail_policy,
            )
            if ok:
                inserted += 1
            else:
                failed.append(_qid)

        if args.throttle_ms > 0:
            await asyncio.sleep(args.throttle_ms / 1000.0)

    # flush leftover
    if batch:
        to_ins = _flush_batch_now()
        for _qid, _doc in to_ins:
            qid_fp = f"{args.split}:{_qid}"
            ok = await _ainsert_one(
                rag=rag,
                doc=_doc,
                retries=args.retries,
                sleep_s=args.retry_sleep,
                qid=qid_fp,
                fail_policy=args.fail_policy,
            )
            if ok:
                inserted += 1
            else:
                failed.append(_qid)

    fin = getattr(rag, "finalize_storages", None)
    if fin is not None:
        await _maybe_await(fin())

    print("\n[DONE]")
    print(f"  problems_path: {problems_path}")
    print(f"  pid_splits_path: {pid_splits_path or 'N/A'}")
    print(f"  inserted: {inserted}")
    print(f"  failed: {len(failed)}")
    if failed:
        print(f"  failed_qids(head): {failed[:20]}")
    print(f"  rag_storage: {os.path.join(args.working_dir, 'rag_storage')}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--caption_file", default="")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "minival", "all"])
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--clear", action="store_true")

    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--llm_model_name", default="qwen2.5:32b")
    ap.add_argument("--num_ctx", type=int, default=8192)

    ap.add_argument("--embed_model", default="nomic-embed-text")
    ap.add_argument("--embedding_dim", type=int, default=768)
    ap.add_argument("--max_embed_tokens", type=int, default=8192)

    ap.add_argument("--llm_max_async", type=int, default=2)
    ap.add_argument("--embed_max_async", type=int, default=4)
    ap.add_argument("--max_parallel_insert", type=int, default=1)

    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry_sleep", type=float, default=2.0)
    ap.add_argument("--fail_policy", choices=["skip", "raise"], default="skip")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--throttle_ms", type=int, default=0)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
