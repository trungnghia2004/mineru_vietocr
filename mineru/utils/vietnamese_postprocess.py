import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import torch
from loguru import logger

from mineru.utils.config_reader import get_device


_VI_CHAR_CLASS = r"A-Za-z0-9\u00C0-\u1EF9\u0110\u0111"
_SINGLE_CHAR_TOKEN_RE = re.compile(rf"^[{_VI_CHAR_CLASS}]$")
_ALPHA_BOUNDARY_DIGIT_RE = re.compile(rf"(?<=[A-Za-z\u00C0-\u1EF9\u0110\u0111])[0158](?=[A-Za-z\u00C0-\u1EF9\u0110\u0111])")
_DIGIT_BOUNDARY_ALPHA_RE = re.compile(r"(?<=\d)[OoilISB|](?=\d)")
_PUNCT_NO_SPACE_PATTERNS = (
    (re.compile(r"\s+([,.;!?%)\]])"), r"\1"),
    (re.compile(r"([(\[])\s+"), r"\1"),
    (re.compile(r"(?<=[A-Za-z0-9])\s*/\s*(?=[A-Za-z0-9])"), "/"),
    (re.compile(r"(?<=[A-Za-z0-9])\s*\.\s*(?=[A-Za-z0-9])"), "."),
    (re.compile(r"(?<=[A-Za-z0-9])\s*_\s*(?=[A-Za-z0-9])"), "_"),
    (re.compile(r"(?<=[A-Za-z0-9])\s*-\s*(?=[A-Za-z0-9])"), "-"),
)
_WARNED_UNAVAILABLE = False
_SCORER = None


def _is_enabled() -> bool:
    value = os.getenv("MINERU_VIETNAMESE_BERT_POSTPROCESS", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _resolve_phobert_dir() -> Optional[Path]:
    env_dir = os.getenv("MINERU_PHOBERT_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    repo_dir = Path(__file__).resolve().parents[2] / "models_cache" / "phobert-base-v2"
    candidates.append(repo_dir)

    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots_dir = hf_home / "hub" / "models--vinai--phobert-base-v2" / "snapshots"
    if snapshots_dir.exists():
        candidates.extend(sorted(snapshots_dir.iterdir(), reverse=True))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if (candidate / "config.json").exists() and (
            (candidate / "pytorch_model.bin").exists()
            or (candidate / "model.safetensors").exists()
        ):
            return candidate
    return None


class _PhoBertScorer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.max_tokens = int(os.getenv("MINERU_VIETNAMESE_BERT_MAX_TOKENS", "96"))
        self.enabled = False
        self._load()

    def _load(self) -> None:
        phobert_dir = _resolve_phobert_dir()
        if phobert_dir is None:
            return

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            requested_device = os.getenv("MINERU_VIETNAMESE_BERT_DEVICE", "").strip()
            if requested_device:
                device_name = requested_device
            else:
                device_name = get_device()
                if device_name.startswith(("cuda", "mps", "npu")):
                    device_name = "cpu"

            self.device = torch.device(device_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                phobert_dir, local_files_only=True
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                phobert_dir, local_files_only=True
            ).to(self.device)
            self.model.eval()
            self.enabled = True
            logger.info(f"Loaded PhoBERT postprocessor from {phobert_dir}")
        except Exception as exc:
            logger.warning(f"PhoBERT postprocessor unavailable: {exc}")

    @lru_cache(maxsize=2048)
    def score_text(self, text: str) -> float:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return float("-inf")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        seq_len = int(input_ids.shape[1])
        if seq_len <= 2:
            return float("-inf")

        total_log_prob = 0.0
        token_count = 0
        with torch.inference_mode():
            for index in range(1, seq_len - 1):
                token_id = int(input_ids[0, index].item())
                if token_id in self.tokenizer.all_special_ids:
                    continue
                masked_ids = input_ids.clone()
                masked_ids[0, index] = self.tokenizer.mask_token_id
                logits = self.model(
                    input_ids=masked_ids, attention_mask=attention_mask
                ).logits[0, index]
                total_log_prob += torch.log_softmax(logits, dim=-1)[token_id].item()
                token_count += 1

        if token_count == 0:
            return float("-inf")
        return total_log_prob / token_count


def _get_scorer() -> Optional[_PhoBertScorer]:
    global _SCORER, _WARNED_UNAVAILABLE
    if _SCORER is None:
        _SCORER = _PhoBertScorer()
    if not _SCORER.enabled and not _WARNED_UNAVAILABLE:
        logger.warning(
            "Vietnamese BERT post-processing is enabled but PhoBERT weights were not found locally."
        )
        _WARNED_UNAVAILABLE = True
    return _SCORER if _SCORER.enabled else None


def _normalize_spacing(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    for pattern, replacement in _PUNCT_NO_SPACE_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


def _has_single_char_run(text: str) -> bool:
    run_length = 0
    for token in text.split():
        if _SINGLE_CHAR_TOKEN_RE.fullmatch(token):
            run_length += 1
            if run_length >= 4:
                return True
        else:
            run_length = 0
    return False


def _join_single_char_runs(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    merged_tokens = []
    run = []

    def flush_run() -> None:
        if run:
            merged_tokens.append("".join(run))
            run.clear()

    for token in tokens:
        if _SINGLE_CHAR_TOKEN_RE.fullmatch(token):
            run.append(token)
        else:
            flush_run()
            merged_tokens.append(token)
    flush_run()
    return " ".join(merged_tokens)


def _replace_common_ocr_confusions(text: str) -> str:
    chars = list(text)
    for index, char in enumerate(chars):
        prev_char = chars[index - 1] if index > 0 else ""
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        prev_is_alpha = prev_char.isalpha()
        next_is_alpha = next_char.isalpha()
        prev_is_digit = prev_char.isdigit()
        next_is_digit = next_char.isdigit()

        if char == "0" and (prev_is_alpha or next_is_alpha) and not (prev_is_digit or next_is_digit):
            chars[index] = "O"
        elif char == "1" and (prev_is_alpha or next_is_alpha) and not (prev_is_digit or next_is_digit):
            chars[index] = "I"
        elif char == "5" and (prev_is_alpha or next_is_alpha) and not (prev_is_digit or next_is_digit):
            chars[index] = "S"
        elif char == "8" and (prev_is_alpha or next_is_alpha) and not (prev_is_digit or next_is_digit):
            chars[index] = "B"
        elif char in {"O", "o"} and (prev_is_digit or next_is_digit) and not (prev_is_alpha or next_is_alpha):
            chars[index] = "0"
        elif char in {"I", "l", "|"} and (prev_is_digit or next_is_digit) and not (prev_is_alpha or next_is_alpha):
            chars[index] = "1"
        elif char == "S" and (prev_is_digit or next_is_digit) and not (prev_is_alpha or next_is_alpha):
            chars[index] = "5"
        elif char == "B" and (prev_is_digit or next_is_digit) and not (prev_is_alpha or next_is_alpha):
            chars[index] = "8"
    return "".join(chars)


def _looks_suspicious(text: str, score: Optional[float], table_cell: bool) -> bool:
    if len(text) < 3:
        return False
    if _has_single_char_run(text):
        return True
    if _ALPHA_BOUNDARY_DIGIT_RE.search(text):
        return True
    if _DIGIT_BOUNDARY_ALPHA_RE.search(text):
        return True
    if re.search(r"\s+[/:._-]\s+|\s+[/:._-]|[/:._-]\s+", text):
        return True
    if score is None:
        return table_cell
    return score < (0.95 if table_cell else 0.90)


def _generate_candidates(text: str) -> Iterable[str]:
    normalized = _normalize_spacing(text)
    joined = _join_single_char_runs(normalized)
    swapped = _replace_common_ocr_confusions(normalized)
    joined_swapped = _replace_common_ocr_confusions(joined)
    for candidate in (text.strip(), normalized, joined, swapped, joined_swapped):
        candidate = candidate.strip()
        if candidate:
            yield candidate


def postprocess_ocr_text(
    text: str,
    lang: Optional[str] = None,
    score: Optional[float] = None,
    table_cell: bool = False,
) -> str:
    if not text:
        return text

    cleaned = _normalize_spacing(text)
    if lang != "vi":
        return cleaned
    if not _is_enabled():
        return cleaned
    if len(cleaned) > 120:
        return cleaned
    if not _looks_suspicious(cleaned, score, table_cell):
        return cleaned

    candidates = list(dict.fromkeys(_generate_candidates(cleaned)))
    if len(candidates) == 1:
        return candidates[0]

    scorer = _get_scorer()
    if scorer is None:
        return candidates[1] if len(candidates) > 1 else cleaned

    baseline_score = scorer.score_text(candidates[0])
    best_text = candidates[0]
    best_score = baseline_score

    for candidate in candidates[1:]:
        candidate_score = scorer.score_text(candidate)
        if candidate_score > best_score:
            best_text = candidate
            best_score = candidate_score

    min_gain = float(os.getenv("MINERU_VIETNAMESE_BERT_MIN_GAIN", "0.05"))
    if best_text != candidates[0] and (best_score - baseline_score) >= min_gain:
        logger.debug(
            f"PhoBERT postprocessed OCR text: '{candidates[0]}' -> '{best_text}'"
        )
        return best_text

    return cleaned
