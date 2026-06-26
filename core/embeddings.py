"""Local sentence embeddings — the engine behind TARS's *associative* recall.

The episodic/profile layers surface memories by salience+recency (see memory.py). That
makes him remember what's generally important, but not what's *relevant to the thing you
just said*. Human memory is associative: a word drags up a connected memory even if you
hadn't thought about it in months. That's what this gives him — every turn, the user's
utterance is embedded and the most semantically similar memories are pulled up on demand.

Deliberately dependency-free beyond what TARS already ships: a small MiniLM model run as
plain ONNX via onnxruntime (already here for Kokoro), tokenized with `tokenizers`, fetched
once with `huggingface_hub`. No torch, no fastembed, no sentence-transformers — those pull
a CPU `onnxruntime` that collides with the project's onnxruntime-gpu (see requirements.txt).

Runs on CPU on purpose: embeddings are tiny and the GPU is the scarce resource (Whisper +
maybe llama). It's all best-effort — if the model can't be fetched/loaded (offline first
run, etc.) the embedder reports unavailable and TARS simply falls back to salience recall.
"""
import threading

import numpy as np

# Xenova's ONNX export of all-MiniLM-L6-v2: ships a ready tokenizer.json and an int8
# model, 384-dim output. Quantized is ~23MB and plenty accurate for nearest-neighbour
# recall, and faster on CPU than the fp32 build.
_REPO = "Xenova/all-MiniLM-L6-v2"
_MODEL_FILE = "onnx/model_quantized.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_MAX_TOKENS = 256
EMBED_DIM = 384


class Embedder:
    """Lazy singleton around the ONNX MiniLM. First use downloads + loads the model;
    failures flip `available` to False so callers can degrade gracefully."""

    _instance: "Embedder | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._load_lock = threading.Lock()
        self._tried = False
        self.available = False

    @classmethod
    def get(cls) -> "Embedder":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> bool:
        if self.available:
            return True
        with self._load_lock:
            if self.available:
                return True
            if self._tried:  # already failed once — don't hammer HF every turn
                return False
            self._tried = True
            try:
                import os

                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                cache_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "embeddings",
                )
                os.makedirs(cache_dir, exist_ok=True)
                model_path = hf_hub_download(
                    _REPO, _MODEL_FILE, local_dir=cache_dir)
                tok_path = hf_hub_download(
                    _REPO, _TOKENIZER_FILE, local_dir=cache_dir)

                tok = Tokenizer.from_file(tok_path)
                tok.enable_truncation(max_length=_MAX_TOKENS)
                tok.enable_padding()
                self._tokenizer = tok

                # CPU on purpose: keep the GPU free for Whisper/llama. Quiet logs.
                so = ort.SessionOptions()
                so.log_severity_level = 3
                self._session = ort.InferenceSession(
                    model_path, sess_options=so, providers=["CPUExecutionProvider"])
                self._input_names = {i.name for i in self._session.get_inputs()}
                self.available = True
                print("[EMBED] sentence embedder ready (MiniLM int8, CPU)")
            except Exception as e:
                print(f"[EMBED] embedder unavailable, semantic recall off: "
                      f"{type(e).__name__}: {e}")
                self.available = False
        return self.available

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Embed a batch into L2-normalized 384-dim vectors (so cosine = dot product).
        Returns None if the model isn't available or the input is empty."""
        if not texts or not self._ensure_loaded():
            return None
        encs = self._tokenizer.encode_batch([t or "" for t in texts])
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        try:
            out = self._session.run(None, feed)[0]  # [n, seq, dim]
        except Exception as e:
            print(f"[EMBED] inference failed, disabling recall: {type(e).__name__}: {e}")
            self.available = False
            return None
        # Mean-pool over real tokens, then L2-normalize.
        mask = attention_mask[..., None].astype(np.float32)
        summed = (out * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        emb = summed / counts
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray | None:
        out = self.encode([text])
        return None if out is None else out[0]

    def warmup(self) -> None:
        """Load the model + run one tiny inference now, so the first real turn doesn't
        pay the cold start. Safe to call in a background thread at startup."""
        self.encode_one("warmup")
