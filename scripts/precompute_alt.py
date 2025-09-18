# save as scripts/precompute_word_features_alt.py
import argparse, logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# reuse same word collector as your existing script
def load_words_from_language(language_dir: Path) -> List[str]:
    words = set()
    for fname in ["train.txt","val.txt","test.txt"]:
        f = language_dir / fname
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        words.add(parts[0]); words.add(parts[1])
    # optional relation files as in original script if you want more words
    return sorted(w for w in words if w)

def extract_tokenmean_features(model_name: str, words: List[str], device: torch.device,
                               trained_checkpoint: Optional[Path]=None,
                               last4: bool=False, batch_size:int=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # same checkpoint loading strategy as your script (optional): load compatible keys
    if trained_checkpoint and trained_checkpoint.exists():
        try:
            sd = torch.load(trained_checkpoint, map_location="cpu")
            compat = {k:v for k,v in sd.items() if k in model.state_dict() and tuple(v.shape)==tuple(model.state_dict()[k].shape)}
            if compat:
                model.load_state_dict(compat, strict=False)
        except Exception as e:
            logger.warning(f"checkpoint load failed: {e}")
    # find base encoder like in your script
    base = None
    for attr in ("base_model","bert","roberta","distilbert","transformer","model"):
        if hasattr(model, attr):
            base = getattr(model, attr); break
    if base is None:
        base = model
    base.to(device); base.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            toks = {k:v.to(device) for k,v in toks.items()}
            # request hidden states if last4 needed
            outputs = base(**toks, output_hidden_states=last4)
            if last4:
                hs = outputs.hidden_states  # tuple (L+1, B, T, H)
                last4_stack = torch.stack(hs[-4:], dim=0).mean(dim=0)  # (B,T,H) average last4 layers
                token_embs = last4_stack
            else:
                token_embs = outputs.last_hidden_state  # (B,T,H)
            # attention mask to ignore padding
            att = toks.get("attention_mask", None)  # (B,T)
            if att is None:
                mean_pool = token_embs.mean(dim=1)
            else:
                att = att.unsqueeze(-1)  # (B,T,1)
                summed = (token_embs * att).sum(dim=1)  # (B,H)
                lens = att.sum(dim=1).clamp(min=1.0)
                mean_pool = summed / lens
            all_embs.append(mean_pool.cpu())
    embs = torch.cat(all_embs, dim=0).numpy()
    return embs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--output_dir", default="models/trained/bert/embeddings")
    ap.add_argument("--model_name", default="bert-base-multilingual-cased")
    ap.add_argument("--trained_checkpoint", default=None)
    ap.add_argument("--last4", action="store_true", help="average last4 layers then token-mean")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    droot = Path(args.data_root) / args.language
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    words = load_words_from_language(droot)
    embs = extract_tokenmean_features(args.model_name, words, device, Path(args.trained_checkpoint) if args.trained_checkpoint else None, last4=args.last4, batch_size=args.batch_size)
    out_path = out_dir / f"{args.language}_word_features_tokenmean.npz"
    np.savez_compressed(out_path, embeddings=embs, words=np.array(words, dtype=object))
    logger.info(f"Saved token-mean features to {out_path} shape={embs.shape}")

if __name__ == "__main__":
    main()