import json
import os
import re
from typing import List

import torch
from torch.utils.data import Dataset

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# Re-exported so existing `from rwkv_model import clone_state` imports keep working.
from state_utils import clone_state, move_state_to_cpu, move_state_to_device  # noqa: F401

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question only using the provided context. "
    "You must always reason in <think>...</think> format first, "
    "then provide the final answer after </think>."
)


def resolve_rwkv_model_path(model_path: str, model_filename: str = "") -> str:
    if os.path.isfile(model_path):
        if model_path.endswith(".pth"):
            return model_path[:-4]
        return model_path

    if os.path.isfile(model_path + ".pth"):
        return model_path

    if model_path.endswith(".pth"):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if "/" not in model_path:
        raise FileNotFoundError(
            f"Invalid model path '{model_path}'. "
            "Use local .pth path or Hugging Face repo id like 'BlinkDL/rwkv7-g1'."
        )

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for auto download. "
            "Install it with: pip install huggingface_hub"
        ) from e

    repo_files = list_repo_files(repo_id=model_path)
    pth_files = [f for f in repo_files if f.endswith(".pth")]
    if not pth_files:
        raise FileNotFoundError(f"No .pth file found in Hugging Face repo: {model_path}")

    target_file = model_filename or sorted(pth_files)[-1]
    if target_file not in pth_files:
        raise FileNotFoundError(
            f"'{target_file}' not found in repo {model_path}. Available: {pth_files}"
        )

    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=model_path,
        filename=target_file,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    if not local_path.endswith(".pth"):
        raise RuntimeError(f"Downloaded file is not a .pth model: {local_path}")
    return local_path[:-4]


def strip_pth_suffix(model_filename: str) -> str:
    name = os.path.basename(model_filename)
    return name[:-4] if name.endswith(".pth") else name


def default_state_dir(model_filename: str, dataset_root: str = "./dataset") -> str:
    return os.path.join(dataset_root, strip_pth_suffix(model_filename))


def default_result_dir(
    model_filename: str,
    experiment_name: str,
    result_root: str = "./result",
) -> str:
    return os.path.join(result_root, experiment_name, strip_pth_suffix(model_filename))


class ValidationStateDataset(Dataset):
    def __init__(self, state_dir: str, limit: int = -1):
        self.state_dir = state_dir
        index_path = os.path.join(state_dir, "index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"State index not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise TypeError("State index file must be a JSON list.")

        if limit is not None and limit > 0:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rel_file = row.get("file")
        if not rel_file:
            raise KeyError("Index row must contain 'file'.")
        path = rel_file if os.path.isabs(rel_file) else os.path.join(self.state_dir, rel_file)
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"State record must be a dict: {path}")
        return payload


def load_validation_state_dataset(state_dir: str, limit: int = -1) -> ValidationStateDataset:
    return ValidationStateDataset(state_dir=state_dir, limit=limit)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def squad_em(prediction: str, answers: list[str]) -> int:
    if not answers:
        return 0
    norm_pred = normalize_answer(prediction)
    return int(any(normalize_answer(a) in norm_pred for a in answers))


def init_jsonl_file(path: str) -> None:
    with open(path, "w", encoding="utf-8"):
        pass


def append_jsonl_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


class RWKVModel:
    def __init__(
        self,
        model_path: str,
        model_filename: str = "",
        strategy: str = "cuda fp16",
        tokenizer: str = "rwkv_vocab_v20230424",
    ):
        resolved_model_path = resolve_rwkv_model_path(model_path, model_filename)
        self.model = RWKV(model=resolved_model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, tokenizer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen_args = PIPELINE_ARGS(
            temperature=0.0,
            top_p=0.0,
            top_k=1,
            alpha_frequency=0.0,
            alpha_presence=0.0,
            alpha_decay=1.0,
            token_ban=[],
            token_stop=[],
            chunk_len=256,
        )

    @staticmethod
    def build_prefill_prompt(system_prompt: str, context: str) -> str:
        return f"System: {system_prompt}\n\nContext: {context}\n\n"

    @staticmethod
    def build_question_prompt(question: str) -> str:
        return f"Question: {question}\nAnswer:"

    @staticmethod
    def strip_think_block(text: str) -> str:
        text = re.sub(r"(?is)<think>.*?</think>", "", text)
        return text.strip()

    @torch.no_grad()
    def prefill_state(self, context: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        tokens = self.pipeline.encode(self.build_prefill_prompt(system_prompt, context))
        _, state = self.model.forward(tokens, None)
        return move_state_to_cpu(state)

    @torch.no_grad()
    def generate(self, question: str, init_state, max_new_tokens: int = 256) -> str:
        prompt = self.build_question_prompt(question)
        state = move_state_to_device(clone_state(init_state), self.device)
        tokens = self.pipeline.encode(prompt)
        out, state = self.model.forward(tokens, state)

        generated: List[int] = []
        seen_think_open = False
        seen_think_close = False

        for _ in range(max_new_tokens):
            token = self.pipeline.sample_logits(
                out,
                temperature=self.gen_args.temperature,
                top_p=self.gen_args.top_p,
                top_k=self.gen_args.top_k,
            )
            candidate = self.pipeline.decode(generated + [token])
            if "�" in candidate:
                break
            generated.append(token)
            candidate_lower = candidate.lower()

            if not seen_think_open and "<think>" in candidate_lower:
                seen_think_open = True
            if not seen_think_open and len(generated) >= 32:
                break
            if "</think>" in candidate_lower:
                seen_think_close = True
                out, state = self.model.forward([token], state)
                break

            out, state = self.model.forward([token], state)

        if seen_think_open and not seen_think_close:
            forced_close_tokens = self.pipeline.encode("</think>")
            for forced_token in forced_close_tokens:
                candidate = self.pipeline.decode(generated + [forced_token])
                if "�" in candidate:
                    break
                generated.append(forced_token)
                out, state = self.model.forward([forced_token], state)
            seen_think_close = "</think>" in self.pipeline.decode(generated).lower()

        if seen_think_close:
            for _ in range(32):
                token = self.pipeline.sample_logits(
                    out,
                    temperature=self.gen_args.temperature,
                    top_p=self.gen_args.top_p,
                    top_k=self.gen_args.top_k,
                )
                candidate = self.pipeline.decode(generated + [token])
                if "�" in candidate:
                    break
                generated.append(token)
                out, state = self.model.forward([token], state)

        decoded = self.pipeline.decode(generated).strip()
        return self.strip_think_block(decoded)
