#!/usr/bin/env python3
"""
Materialize HuggingFace snapshot cache into models/bert/<lang>/model and tokenizer.
This is a helper to run after the downloader has used snapshot_download which stores files
under hf_cache/.../snapshots/<hash>/. Some environments keep the snapshot in a cache
and don't copy it into the model folder; this script copies the relevant files so
`AutoModel.from_pretrained(<lang>/model)` works.
"""
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models' / 'bert'

CANDIDATES = [
    'pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'flax_model.msgpack',
    'config.json', 'tokenizer_config.json', 'vocab.txt', 'tokenizer.json',
    'sentencepiece.bpe.model', 'added_tokens.json', 'special_tokens_map.json', 'README.md'
]


def find_latest_snapshot(hf_cache_dir: Path) -> Path | None:
    # HF cache layout often: models--<owner>--<repo>/snapshots/<hash>/
    if not hf_cache_dir.exists():
        return None
    # try to find a directory under hf_cache_dir that contains 'snapshots'
    for models_root in hf_cache_dir.iterdir():
        snaps = models_root / 'snapshots'
        if snaps.exists() and snaps.is_dir():
            snapshots = sorted([p for p in snaps.iterdir() if p.is_dir()])
            if snapshots:
                return snapshots[-1]
    # fallback: look for any snapshots dir anywhere under hf_cache_dir
    for p in hf_cache_dir.rglob('snapshots'):
        if p.is_dir():
            snapshots = sorted([q for q in p.iterdir() if q.is_dir()])
            if snapshots:
                return snapshots[-1]
    return None


def copy_snapshot_to_model(snapshot_dir: Path, target_model_dir: Path, target_tokenizer_dir: Path):
    logger.info(f'Copying files from snapshot {snapshot_dir} to {target_model_dir}')
    target_model_dir.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        for f in snapshot_dir.rglob(name):
            try:
                shutil.copy2(f, target_model_dir / name)
                logger.info(f'Copied {f} -> {target_model_dir / name}')
            except Exception as e:
                logger.warning(f'Failed to copy {f}: {e}')
    # copy tokenizer dir if present
    for td in snapshot_dir.rglob('tokenizer'):
        if td.is_dir():
            try:
                if not target_tokenizer_dir.exists():
                    shutil.copytree(td, target_tokenizer_dir)
                    logger.info(f'Copied tokenizer dir {td} -> {target_tokenizer_dir}')
            except Exception as e:
                logger.warning(f'Failed to copy tokenizer dir {td}: {e}')


if __name__ == '__main__':
    if not MODELS_DIR.exists():
        logger.error(f'Models dir not found: {MODELS_DIR}')
        raise SystemExit(1)

    for lang_dir in sorted(MODELS_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        hf_cache = lang_dir / 'hf_cache'
        model_dir = lang_dir / 'model'
        tokenizer_dir = lang_dir / 'tokenizer'

        # Skip if model dir already has weight files
        has_weights = any((model_dir / fname).exists() for fname in ('pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'flax_model.msgpack'))
        if has_weights:
            logger.info(f'{lang_dir.name}: model directory already has weight files, skipping')
            continue

        snapshot = find_latest_snapshot(hf_cache)
        if snapshot is None:
            logger.info(f'{lang_dir.name}: no snapshot found under {hf_cache}, skipping')
            continue

        copy_snapshot_to_model(snapshot, model_dir, tokenizer_dir)

    logger.info('Done materializing snapshots.')
