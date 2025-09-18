import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_words_from_language(language_dir: Path) -> List[str]:
    words = set()
    # standard train/val/test files
    for fname in ["train.txt", "val.txt", "test.txt"]:
        f = language_dir / fname
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        words.add(parts[0])
                        words.add(parts[1])
    # optional relation files
    for fname in [
        "all_synonyms_synonyms.txt",
        "all_antonyms_antonyms.txt",
        "wordnet_antonyms_antonyms.txt",
        "conceptnet_synonyms_synonyms.txt",
        "conceptnet_antonyms_antonyms.txt",
    ]:
        f = language_dir / fname
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        words.add(parts[0])
                        words.add(parts[1])
    return sorted(w for w in words if w)


def extract_word_features(
    model_name: str,
    words: List[str],
    device: torch.device,
    trained_checkpoint: Path = None,
    layer_pool: str = "last_hidden_cls",
) -> Tuple[np.ndarray, List[str]]:
    """
    layer_pool options:
      - last_hidden_cls: use last_hidden_state[:,0,:]
      - last4_avg_cls: average last 4 hidden states CLS
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ft_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if trained_checkpoint and trained_checkpoint.exists():
        logger.info(f"Loading fine-tuned checkpoint from {trained_checkpoint}")
        try:
            sd = torch.load(trained_checkpoint, map_location="cpu")
            model_sd = ft_model.state_dict()
            compat = {}
            skipped = []
            for k, v in sd.items():
                if k in model_sd and hasattr(model_sd[k], 'shape') and hasattr(v, 'shape') and tuple(v.shape) == tuple(model_sd[k].shape):
                    compat[k] = v
                elif k in model_sd:
                    skipped.append(k)
            if not compat:
                logger.warning("Checkpoint is incompatible with the provided model_name; proceeding with base pretrained weights.")
            else:
                if any('embeddings.word_embeddings.weight' in k for k in skipped):
                    logger.warning("Tokenizer/base mismatch detected: embedding table shapes differ. Use a model_name that matches the checkpoint's base (e.g., the same multilingual/german variant used for fine-tuning). Using base model embeddings and loading other compatible layers.")
                ft_model.load_state_dict(compat, strict=False)
                logger.info(f"Loaded {len(compat)}/{len(sd)} keys from checkpoint; skipped {len(skipped)} mismatched keys.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint; continuing with base weights. Error: {e}")
    base = None
    for attr in ("base_model", "bert", "roberta", "distilbert", "transformer", "model"):
        if hasattr(ft_model, attr):
            base = getattr(ft_model, attr)
            break
    if base is None:
        logger.warning("Could not find base encoder; using full model as encoder")
        base = ft_model
    base.to(device)
    base.eval()

    embs = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            toks = tokenizer(batch_words, return_tensors="pt", padding=True, truncation=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            outputs = base(**toks, output_hidden_states=(layer_pool == "last4_avg_cls"))
            if layer_pool == "last4_avg_cls":
                hs = outputs.hidden_states  # tuple of layers
                last4 = torch.stack(hs[-4:], dim=0)  # (4, B, T, H)
                cls = last4[:, :, 0, :].mean(dim=0)  # (B, H)
            else:
                last = getattr(outputs, "last_hidden_state", None)
                if last is None:
                    last = outputs[0]
                cls = last[:, 0, :]  # (B, H)
            embs.append(cls.cpu())
    embs = torch.cat(embs, dim=0).numpy()
    return embs, words


def main():
    ap = argparse.ArgumentParser(description="Precompute word features from a BERT model")
    ap.add_argument("--language", required=True)
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--output_dir", default="assets/analysis/embeddings")
    ap.add_argument("--model_name", required=False, default=None)
    ap.add_argument("--trained_checkpoint", default=None)
    ap.add_argument("--layer_pool", default="last_hidden_cls", choices=["last_hidden_cls", "last4_avg_cls"])
    args = ap.parse_args()

    lang = args.language
    droot = Path(args.data_root) / lang
    oroot = Path(args.output_dir)
    oroot.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.trained_checkpoint) if args.trained_checkpoint else None

    # Try to infer model_name from config if not provided
    model_name = args.model_name
    if model_name is None:
        # Fallback simple default
        model_name = "bert-base-multilingual-cased"
        logger.info(f"No model_name provided; defaulting to {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    words = load_words_from_language(droot)
    logger.info(f"Collected {len(words)} unique words for {lang}")

    embs, ordered_words = extract_word_features(model_name, words, device, ckpt, args.layer_pool)
    out_path = oroot / f"{lang}_word_features.npz"
    np.savez_compressed(out_path, embeddings=embs, words=np.array(ordered_words, dtype=object))
    logger.info(f"Saved features to {out_path} with shape {embs.shape}")


if __name__ == "__main__":
    main()
