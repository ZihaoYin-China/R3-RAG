import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from retrieval.base_retrieval import BaseRetrieval
import open_clip


def _sort_img_key(p: str):
    name = os.path.basename(p).lower()
    if name.startswith("image."):
        return (0, 0, name)
    m = re.match(r"choice_(\d+)\.", name)
    if m:
        return (1, int(m.group(1)), name)
    return (2, 999, name)


def _collect_images(dir_path: str) -> List[str]:
    paths = []
    for ext in ("png", "jpg", "jpeg", "webp", "bmp"):
        paths.extend(glob(os.path.join(dir_path, f"*.{ext}")))
    paths = [os.path.abspath(p) for p in paths]
    paths.sort(key=_sort_img_key)
    return paths


def _resolve_image_paths(root: str, qid: str, split_hint: Optional[str] = None) -> List[str]:
    candidates = []

    splits = []
    if split_hint:
        splits.append(split_hint)
    splits += ["test", "val", "train", "minival"]

    for sp in splits:
        candidates.append(os.path.join(root, sp, str(qid)))
        candidates.append(os.path.join(root, "images", sp, str(qid)))
        candidates.append(os.path.join(root, "ScienceQA", sp, str(qid)))
        candidates.append(os.path.join(root, "ScienceQA", "images", sp, str(qid)))
        candidates.append(os.path.join(root, "data", sp, str(qid)))

    candidates.append(os.path.join(root, "data", "scienceqa", "images", str(qid)))
    candidates.append(os.path.join(root, "data", "scienceqa", str(qid)))

    for d in candidates:
        if os.path.isdir(d):
            imgs = _collect_images(d)
            if imgs:
                return imgs
    return []


def _get_text(problem: Dict[str, Any]) -> str:
    q = str(problem.get("question", "")).strip()
    cap = str(problem.get("caption", "")).strip()
    hint = str(problem.get("hint", "") or "").strip()
    txt = f"Caption: {cap}\nQuestion: {q}"
    if hint:
        txt += f"\nHint: {hint}"
    return txt.strip()


def _hash_text_embed(text: str, dim: int = 1024) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    for t in toks:
        idx = (hash(t) % dim + dim) % dim
        vec[idx] += 1.0
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


class MMRetrieval(BaseRetrieval):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.device = (getattr(config, "device", "") or "").strip()
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = getattr(config, "clip_model", "ViT-B-32")
        self.clip_pretrained = getattr(config, "clip_pretrained", "laion2b_s34b_b79k")
        self.batch_size = int(getattr(config, "clip_batch_size", 32))
        self.alpha = float(getattr(config, "mm_alpha", 0.5))

        self.root = getattr(config, "image_root", None) or getattr(config, "working_dir", ".")
        self.split_hint = getattr(config, "test_split", None) or "test"

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.clip_model}-{self.clip_pretrained}-a{self.alpha}")
        self.index_path = os.path.join(getattr(config, "working_dir", "."), f"mm_index_{safe}.npz")

        self.force_rebuild = bool(getattr(config, "mm_rebuild", False))
        self.rebuild_if_empty = bool(getattr(config, "mm_rebuild_if_empty", True))
        self.fallback_dim = int(getattr(config, "mm_fallback_dim", 1024))

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._qids: List[str] = []
        self._vecs: Optional[np.ndarray] = None
        self._prepared = False
        self._clip_ok = True
        self._clip_err = ""

    def _lazy_load_model(self):
        if self._model is not None:
            return
        if not self._clip_ok:
            return

        try:
            pretrained_arg = self.clip_pretrained
            if isinstance(pretrained_arg, str) and os.path.exists(pretrained_arg):
                pretrained_arg = os.path.abspath(pretrained_arg)

            kwargs = {}
            cache_dir = getattr(self.config, "clip_cache_dir", None)
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            try:
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self.clip_model, pretrained=pretrained_arg, **kwargs
                )
            except TypeError:
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self.clip_model, pretrained=pretrained_arg
                )

            self._tokenizer = open_clip.get_tokenizer(self.clip_model)
            self._model.to(self.device)
            self._model.eval()

        except Exception as e:
            self._clip_ok = False
            self._clip_err = f"{type(e).__name__}: {e}"
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            print(f"[MMRetrieval] OpenCLIP load failed, switching to fallback text-hash. Error: {self._clip_err}")

    def _load_cache(self) -> bool:
        if self.force_rebuild:
            return False
        if not os.path.exists(self.index_path):
            return False

        try:
            data = np.load(self.index_path, allow_pickle=True)
            qids = list(data["qids"].tolist())
            vecs = data["vecs"].astype(np.float32)
            kind = str(data.get("kind", "clip")).lower()

            if self.rebuild_if_empty and (len(qids) == 0 or vecs.shape[0] == 0):
                return False

            if (not self._clip_ok) and kind == "clip":
                return False

            self._qids = qids
            self._vecs = vecs
            self._prepared = True
            print(f"[MMRetrieval] Loaded cache from {self.index_path} (kind={kind}, count={len(qids)})")
            return True
        except Exception:
            return False

    def prepare(self, problems: Dict[str, Any], train_qids: Optional[List[str]] = None):
        if self._prepared:
            return
        if self._load_cache():
            return

        self._lazy_load_model()

        if train_qids:
            iter_qids = [str(x) for x in train_qids if str(x) in problems]
        else:
            iter_qids = [str(k) for k in problems.keys()]
            
        print(f"[MMRetrieval] Building index for {len(iter_qids)} items... (CLIP_OK={self._clip_ok})")

        if self._clip_ok and self._model is not None:
            items: List[Tuple[str, Optional[str], str]] = []
            for qid in iter_qids:
                prob = problems.get(qid, {})
                imgs = _resolve_image_paths(self.root, qid, split_hint=None)
                img_path = imgs[0] if imgs else None
                items.append((qid, img_path, _get_text(prob)))

            qids_out: List[str] = []
            vecs_out: List[np.ndarray] = []

            if not items:
                self._qids = []
                self._vecs = np.zeros((0, 512), dtype=np.float32)
                self._prepared = True
                np.savez_compressed(
                    self.index_path,
                    qids=np.array(self._qids, dtype=object),
                    vecs=self._vecs,
                    kind=np.array("clip")
                )
                return

            with torch.no_grad():
                for st in range(0, len(items), self.batch_size):
                    batch = items[st: st + self.batch_size]
                    b_qids = [x[0] for x in batch]
                    b_imgs = [x[1] for x in batch]
                    b_txts = [x[2] for x in batch]

                    tok = self._tokenizer(b_txts).to(self.device)
                    txt_feat = self._model.encode_text(tok)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                    img_feat = None
                    has_img = [p is not None and os.path.exists(p) for p in b_imgs]
                    
                    if any(has_img):
                        img_tensors = []
                        valid_pos = []
                        for i, p in enumerate(b_imgs):
                            if p and os.path.exists(p):
                                try:
                                    im = Image.open(p).convert("RGB")
                                    img_tensors.append(self._preprocess(im))
                                    valid_pos.append(i)
                                except Exception:
                                    pass
                        if img_tensors:
                            img_t = torch.stack(img_tensors, dim=0).to(self.device)
                            _img_feat = self._model.encode_image(img_t)
                            _img_feat = _img_feat / _img_feat.norm(dim=-1, keepdim=True)
                            
                            img_feat = [None] * len(b_imgs)
                            for j, pos in enumerate(valid_pos):
                                img_feat[pos] = _img_feat[j:j+1]

                    fused_list = []
                    for i in range(len(b_qids)):
                        tf = txt_feat[i:i+1]
                        if img_feat is not None and img_feat[i] is not None:
                            ff = self.alpha * img_feat[i] + (1.0 - self.alpha) * tf
                            ff = ff / ff.norm(dim=-1, keepdim=True)
                        else:
                            ff = tf
                        fused_list.append(ff)

                    fused = torch.cat(fused_list, dim=0)
                    qids_out.extend(b_qids)
                    vecs_out.append(fused.detach().cpu().numpy().astype(np.float32))

            self._qids = qids_out
            self._vecs = np.concatenate(vecs_out, axis=0)

            np.savez_compressed(
                self.index_path,
                qids=np.array(self._qids, dtype=object),
                vecs=self._vecs,
                kind=np.array("clip")
            )
            self._prepared = True
            print(f"[MMRetrieval] Index built & saved to {self.index_path} (size={len(self._qids)})")
            return

        print("[MMRetrieval] Running in FALLBACK mode (Text-Hash) ...")
        items_fb: List[Tuple[str, str]] = []
        for qid in iter_qids:
            prob = problems.get(qid, {})
            txt = _get_text(prob)
            imgs = _resolve_image_paths(self.root, qid, split_hint=None)
            if imgs:
                txt = txt + "\nHAS_IMAGE: 1"
            items_fb.append((qid, txt))

        self._qids = [x[0] for x in items_fb]
        if not self._qids:
             self._vecs = np.zeros((0, self.fallback_dim), dtype=np.float32)
        else:
             self._vecs = np.stack([_hash_text_embed(x[1], dim=self.fallback_dim) for x in items_fb], axis=0).astype(np.float32)

        np.savez_compressed(
            self.index_path,
            qids=np.array(self._qids, dtype=object),
            vecs=self._vecs,
            kind=np.array("fallback"),
            clip_error=np.array(self._clip_err, dtype=object),
        )
        self._prepared = True
        print(f"[MMRetrieval] Fallback index built & saved (size={len(self._qids)})")

    # ================= 修改重点在此方法 =================
    def find_top_k(self, query: str, qid: Optional[str] = None, problems: Optional[Dict[str, Any]] = None):
        if problems is None:
            return ""

        if not self._prepared:
            self.prepare(problems)

        if self._vecs is None or len(self._qids) == 0:
            if not self._clip_ok:
                return f"[MM] OpenCLIP unavailable, and index is empty. Error: {self._clip_err}"
            return ""

        kind = "clip"
        try:
            if os.path.exists(self.index_path):
                data = np.load(self.index_path, allow_pickle=True)
                kind = str(data.get("kind", "clip")).lower()
        except Exception:
            pass

        if kind == "clip":
            self._lazy_load_model()
            if (not self._clip_ok) or (self._model is None):
                return f"[MM] Index is CLIP, but OpenCLIP model unavailable. Error: {self._clip_err}"

            # --- 【创新：意图引导的动态 Alpha 策略】 ---
            # 定义视觉敏感触发词
            visual_triggers = [
                'look at', 'shown in', 'the image', 'diagram', 'graph', 'map', 
                'identify', 'color', 'shape', 'picture', 'figure', 'illustration'
            ]
            # 定义纯理论/非视觉词
            theoretical_triggers = [
                'definition', 'principle', 'law', 'theory', 'statement', 
                'formula', 'calculated', 'reasoning'
            ]
            
            q_lower = query.lower()
            current_alpha = self.alpha # 基础权重 (0.5)

            # 权重动态调节逻辑
            if any(vt in q_lower for vt in visual_triggers):
                # 强视觉意图：图片更重要，提高 alpha 至 0.8
                current_alpha = min(0.9, self.alpha * 1.6)
            elif any(tt in q_lower for tt in theoretical_triggers):
                # 纯理论意图：文本更重要，降低图片权重 alpha 至 0.2
                current_alpha = max(0.1, self.alpha * 0.4)
            # --- 【创新结束】 ---

            img_path = None
            if qid is not None:
                imgs = _resolve_image_paths(self.root, str(qid), split_hint=self.split_hint)
                if imgs:
                    img_path = imgs[0]

            with torch.no_grad():
                tok = self._tokenizer([query]).to(self.device)
                txt_feat = self._model.encode_text(tok)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                if img_path and os.path.exists(img_path):
                    try:
                        im = Image.open(img_path).convert("RGB")
                        img_t = self._preprocess(im).unsqueeze(0).to(self.device)
                        img_feat = self._model.encode_image(img_t)
                        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                        
                        # 使用动态计算的 current_alpha 进行特征融合
                        fused = current_alpha * img_feat + (1.0 - current_alpha) * txt_feat
                    except Exception:
                        fused = txt_feat
                else:
                    fused = txt_feat

                fused = fused / fused.norm(dim=-1, keepdim=True)
                qv = fused.detach().cpu().numpy().astype(np.float32)

        else:
            qv = _hash_text_embed(query, dim=self._vecs.shape[1]).reshape(1, -1).astype(np.float32)

        sims = (self._vecs @ qv.T).reshape(-1)
        top_k = int(getattr(self.config, "top_k", 4))
        idx = np.argsort(-sims)[: max(top_k + 8, top_k)]

        evidences = []
        for i in idx:
            rid = self._qids[int(i)]
            if qid is not None and str(rid) == str(qid):
                continue
            pr = problems.get(str(rid), {})

            prefix = "[MM Similar QID"
            if kind != "clip":
                prefix = "[MM-Fallback Similar QID"

            evidences.append(
                f"{prefix} {rid}] score={float(sims[int(i)]):.4f}\n{_get_text(pr)}"
            )
            if len(evidences) >= top_k:
                break

        res_str = "\n\n".join(evidences)
        if kind != "clip":
            res_str += "\n\n[MM Note] Using offline text-hash fallback."
        
        return res_str