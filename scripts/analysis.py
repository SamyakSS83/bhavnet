#!/usr/bin/env python3
"""Comprehensive analysis script for multilingual antonym detection.

Produces:
- Per-language evaluation for BERT and Dual-Encoder
- Comparison plots (BERT vs Dual-Encoder, baselines)
- Convergence plots (requires saved histories)
- t-SNE visualizations of embeddings
- Detailed per-sample error analysis CSVs

Outputs are saved under assets/analysis/<language>/
"""
import argparse
import yaml
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import subprocess
import itertools
from umap import UMAP
import yaml as pyyaml
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class Analyzer:
    def __init__(self, config, data_root='datasets', assets_root='assets'):
        self.config = config
        self.data_root = Path(data_root)
        self.assets_root = Path(assets_root)
        self.analysis_root = self.assets_root / 'analysis'
        self.analysis_root.mkdir(parents=True, exist_ok=True)

    def evaluate_bert(self, language, model_path, tokenizer_name):
        logger.info(f"Evaluating BERT for {language}")
        df = self.get_pairs_df(language, split='test')
        # Try to load tokenizer/model, but tolerate environment import/runtime errors (timm/wandb issues etc.)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModel.from_pretrained(tokenizer_name)
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            inputs = tokenizer(list(df.word1 + ' <sep> ' + df.word2), padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attn = inputs['attention_mask'].to(device)
            with torch.no_grad():
                out = model(input_ids, attention_mask=attn)
                last_hidden = getattr(out, 'last_hidden_state', out[0])
                cls = last_hidden[:,0,:].cpu().numpy()
        except Exception as e:
            logger.warning(f"BERT model load/encode failed for {language}: {e}")
            cls = None

        # Simple linear probe classifier from saved checkpoint isn't available here; instead we compute embeddings and save them
        save_dir = self.analysis_root / language
        save_dir.mkdir(parents=True, exist_ok=True)
        if cls is not None:
            np.save(save_dir / 'bert_test_cls_embeddings.npy', cls)
        else:
            logger.info(f"Skipping embedding save for {language} (no CLS embeddings)")
        df.to_csv(save_dir / 'bert_test_pairs.csv', index=False)
        logger.info(f"Saved BERT embeddings and test pairs for {language} to {save_dir}")
        return cls, df

    def evaluate_dual_encoder(self, language, model_path, bert_model_name):
        # Load model - requires the trained dual encoder class; here we only load embeddings by running BERT base
        logger.info(f"Collecting Dual-Encoder embeddings for {language}")
        df = self.get_pairs_df(language, split='test')
        cls = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            model = AutoModel.from_pretrained(bert_model_name)
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            inputs = tokenizer(list(df.word1 + ' <sep> ' + df.word2), padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attn = inputs['attention_mask'].to(device)
            with torch.no_grad():
                out = model(input_ids, attention_mask=attn)
                last_hidden = getattr(out, 'last_hidden_state', out[0])
                cls = last_hidden[:,0,:].cpu().numpy()
        except Exception as e:
            logger.warning(f"Dual-encoder embedding collection failed for {language}: {e}")
            cls = None

        save_dir = self.analysis_root / language
        save_dir.mkdir(parents=True, exist_ok=True)
        if cls is not None:
            np.save(save_dir / 'dual_encoder_test_cls_embeddings.npy', cls)
        else:
            logger.info(f"Skipping dual-encoder embedding save for {language} (no CLS embeddings)")
        df.to_csv(save_dir / 'dual_encoder_test_pairs.csv', index=False)
        logger.info(f"Saved Dual-Encoder embeddings (via BERT) and test pairs for {language} to {save_dir}")
        return cls, df

    def baseline_probe(self, language, bert_model_name, probe_model_name='logreg'):
        """Train a simple probe classifier (logistic regression) on BERT CLS embeddings using train/val split.
        Saves predictions and metrics to assets/analysis/<language>/baseline_*."""
        logger.info(f"Running baseline probe for {language} using {bert_model_name}")
        # Load train and test pairs
        # Use helper which supports multiple dataset layouts (datasets/<lang>/ or combined dataset/ files)
        try:
            train_df = self.get_pairs_df(language, split='train')
            test_df = self.get_pairs_df(language, split='test')
        except FileNotFoundError:
            logger.warning(f"Train/test files missing for {language}")
            return None

        # Encode with BERT
        # Try to load model/tokenizer; if it fails, skip the probe but still save pair files
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            model = AutoModel.from_pretrained(bert_model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
        except Exception as e:
            logger.warning(f"Baseline probe model load failed for {language} ({bert_model_name}): {e}")
            # Save pair files for analysis and return
            save_dir = self.analysis_root / language
            save_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(save_dir / 'baseline_train_pairs.csv', index=False)
            test_df.to_csv(save_dir / 'baseline_test_pairs.csv', index=False)
            return None

        def encode(df):
            inputs = tokenizer(list(df.word1 + ' <sep> ' + df.word2), padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                out = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
                last_hidden = getattr(out, 'last_hidden_state', out[0])
                cls = last_hidden[:,0,:].cpu().numpy()
            return cls

        X_train = encode(train_df)
        X_test = encode(test_df)
        y_train = train_df['label'].values
        y_test = test_df['label'].values

        # Fit logistic regression probe (regularized)
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        probs = probe.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=['Not Antonym','Antonym'])

        save_dir = self.analysis_root / language
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save predictions
        rows = []
        for i in range(len(preds)):
            rows.append({'word1': test_df.loc[i,'word1'], 'word2': test_df.loc[i,'word2'], 'label': int(y_test[i]), 'pred': int(preds[i]), 'score': float(np.max(probs[i]))})
        pd.DataFrame(rows).to_csv(save_dir / f'{language}_baseline_probe_predictions.csv', index=False)
        with open(save_dir / f'{language}_baseline_probe_report.txt', 'w') as f:
            f.write(f'Accuracy: {acc}\n')
            f.write(report)
        logger.info(f"Saved baseline probe results for {language} to {save_dir}")
        return {'accuracy': acc, 'report': report}

    def layerwise_probe(self, language, bert_model_name, layers=(2,6,11)):
        """Run a logistic-probe on different transformer layers' CLS embeddings.
        layers: tuple of layer indices (0-based, last layer typically -1 or model.config.num_hidden_layers-1)
        Saves per-layer reports and a comparison plot.
        """
        logger.info(f"Running layer-wise probe for {language} on layers {layers}")
        try:
            train_df = self.get_pairs_df(language, split='train')
            test_df = self.get_pairs_df(language, split='test')
        except FileNotFoundError:
            logger.warning("Missing train/test for layerwise probe")
            return


        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = AutoModel.from_pretrained(bert_model_name, output_hidden_states=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        def encode_layer(df, layer_idx):
            inputs = tokenizer(list(df.word1 + ' <sep> ' + df.word2), padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                out = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
                hs = out.hidden_states
                # hs is tuple: (embeddings, layer1, ..., layerN)
                layer = hs[layer_idx]
                cls = layer[:,0,:].cpu().numpy()
            return cls

        results = {}
        for li in layers:
            try:
                X_train = encode_layer(train_df, li)
                X_test = encode_layer(test_df, li)
                y_train = train_df['label'].values
                y_test = test_df['label'].values
                probe = LogisticRegression(max_iter=1000)
                probe.fit(X_train, y_train)
                preds = probe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                report = classification_report(y_test, preds, target_names=['Not Antonym','Antonym'])
                results[li] = {'acc': acc, 'report': report}
                save_dir = self.analysis_root / language
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / f'layer_{li}_probe_report.txt', 'w') as f:
                    f.write(f'Layer {li} Accuracy: {acc}\n')
                    f.write(report)
            except Exception as e:
                logger.warning(f"Layerwise probe failed for layer {li}: {e}")

        # Plot layer comparison
        try:
            xs = [li for li in results.keys()]
            ys = [results[li]['acc'] for li in xs]
            plt.figure(figsize=(6,4))
            plt.plot(xs, ys, marker='o')
            plt.xlabel('Layer index')
            plt.ylabel('Accuracy')
            plt.title(f'{language} - Layer-wise Probe Accuracy')
            plt.grid(True)
            plt.savefig(self.analysis_root / language / 'layerwise_probe.png')
            plt.close()
            logger.info(f"Saved layer-wise probe plot for {language}")
        except Exception as e:
            logger.warning(f"Failed to plot layerwise results: {e}")

        return results

    def umap_visualization(self, language, embedding_key='bert_test_cls_embeddings.npy'):
        """Run UMAP on saved embeddings and save plot."""
        save_dir = self.analysis_root / language
        emb_path = save_dir / embedding_key
        if not emb_path.exists():
            logger.warning(f"Embeddings not found for UMAP: {emb_path}")
            return
        embs = np.load(emb_path)
        labels = None
        labels_path = save_dir / 'bert_test_pairs.csv'
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            labels = df['label'].values

        try:
            reducer = UMAP(n_components=2, random_state=42)
            emb2 = reducer.fit_transform(embs)
            plt.figure(figsize=(8,6))
            c = labels if labels is not None else None
            plt.scatter(emb2[:,0], emb2[:,1], c=c, cmap='coolwarm', s=5)
            plt.title(f'{language} - UMAP')
            plt.savefig(save_dir / 'umap.png')
            plt.close()
            logger.info(f"Saved UMAP for {language} to {save_dir}")
        except Exception as e:
            logger.warning(f"UMAP failed: {e}")

    def transfer_matrix(self, languages, model='bert'):
        """Compute a cross-lingual transfer matrix: train probe on language A, evaluate on language B.
        Saves a heatmap under assets/analysis/transfer_matrix_<model>.png"""
        logger.info(f"Computing transfer matrix for model={model}")
        acc_matrix = np.zeros((len(languages), len(languages)))
        for i, src in enumerate(languages):
            # Train probe on src
            src_train_path = self.data_root / src / 'train.txt'
            if not src_train_path.exists():
                continue
            src_df = self._read_pairs(src_train_path)
            # encode src
            bert_name = self.config['languages'][src]['bert_model']
            tokenizer = AutoTokenizer.from_pretrained(bert_name)
            model_b = AutoModel.from_pretrained(bert_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_b.to(device)
            model_b.eval()

            def encode_df(df):
                inputs = tokenizer(list(df.word1 + ' <sep> ' + df.word2), padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    out = model_b(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
                    last_hidden = getattr(out, 'last_hidden_state', out[0])
                    cls = last_hidden[:,0,:].cpu().numpy()
                return cls

            X_src = encode_df(src_df)
            y_src = src_df['label'].values
            probe = LogisticRegression(max_iter=1000).fit(X_src, y_src)

            for j, tgt in enumerate(languages):
                tgt_test_path = self.data_root / tgt / 'test.txt'
                if not tgt_test_path.exists():
                    acc_matrix[i, j] = np.nan
                    continue
                tgt_df = self._read_pairs(tgt_test_path)
                X_tgt = encode_df(tgt_df)
                y_tgt = tgt_df['label'].values
                preds = probe.predict(X_tgt)
                acc = accuracy_score(y_tgt, preds)
                acc_matrix[i, j] = acc

        # Save a heatmap
        plt.figure(figsize=(8,6))
        sns.heatmap(acc_matrix, xticklabels=languages, yticklabels=languages, annot=True, fmt='.2f', cmap='viridis')
        plt.xlabel('Target language')
        plt.ylabel('Source language')
        plt.title(f'Cross-lingual transfer matrix ({model})')
        out_png = self.analysis_root / f'transfer_matrix_{model}.png'
        plt.savefig(out_png, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved transfer matrix to {out_png}")


    def run_ablation(self, language, base_config_path, param_grid: dict, output_root=None):
        """Run a small ablation grid by launching train runs with different hyperparameters.
        This writes outputs under assets/ablation/<language>/<run_id>/ and collects histories.
        NOTE: This function launches subprocesses calling train_models.py; ensure paths are correct."""
        output_root = Path(output_root) if output_root else (self.assets_root / 'ablation')
        output_root = output_root / language
        output_root.mkdir(parents=True, exist_ok=True)

        # Generate grid and create per-run configs
        keys = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        runs = []
        # Load base config
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_cfg = pyyaml.safe_load(f)

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            run_dir = output_root / f'run_{i+1}'
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create a per-run config by copying base and overriding training values
            cfg_copy = dict(base_cfg)
            # Apply overrides in the training section if applicable
            if 'training' not in cfg_copy:
                cfg_copy['training'] = {}
            for k, v in params.items():
                # write top-level training overrides where sensible
                cfg_copy['training'][k] = v

            per_run_cfg_path = run_dir / 'config.yaml'
            with open(per_run_cfg_path, 'w', encoding='utf-8') as pcw:
                pyyaml.safe_dump(cfg_copy, pcw)

            # Construct command: call train_models.py with this config
            cmd = [
                'python3', 'train_models.py',
                '--language', language,
                '--model-type', 'dual',
                '--config', str(per_run_cfg_path),
                '--output_dir', str(run_dir)
            ]

            logger.info(f"Launching ablation run {i+1}/{len(combos)}: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd)
            runs.append({'proc': proc, 'cmd': cmd, 'run_dir': str(run_dir), 'params': params})

        # Wait for processes to complete and collect final accuracy from history CSVs
        summary = []
        for r in runs:
            r['proc'].wait()
            logger.info(f"Ablation run finished: {r['cmd']}")
            # Try to read history CSV under run_dir/tables or run_dir
            run_dir = Path(r['run_dir'])
            # look for history CSV
            history_csv = None
            for candidate in (run_dir / 'tables', run_dir, run_dir / 'assets' / 'tables'):
                p = candidate
                if p.exists():
                    for f in p.glob('*history*.csv'):
                        history_csv = f
                        break
                if history_csv:
                    break

            final_acc = None
            if history_csv and history_csv.exists():
                try:
                    hdf = pd.read_csv(history_csv)
                    # Attempt to read last recorded accuracy column
                    if 'test_acc' in hdf.columns:
                        final_acc = float(hdf['test_acc'].dropna().iloc[-1])
                    elif 'val_acc' in hdf.columns:
                        final_acc = float(hdf['val_acc'].dropna().iloc[-1])
                except Exception:
                    final_acc = None

            summary.append({'run_dir': str(run_dir), 'params': r['params'], 'final_acc': final_acc})

        # Aggregate summary into CSV and plot bar chart
        summary_df = pd.DataFrame([{'run': i+1, **s['params'], 'final_acc': s['final_acc']} for i, s in enumerate(summary)])
        summary_csv = output_root / 'ablation_summary.csv'
        summary_df.to_csv(summary_csv, index=False)

        try:
            plt.figure(figsize=(8,4))
            # Create a label per run describing params
            labels = [str(s['params']) for s in summary]
            accs = [s['final_acc'] if s['final_acc'] is not None else 0 for s in summary]
            plt.bar(range(len(accs)), accs)
            plt.xticks(range(len(accs)), labels, rotation=45, ha='right')
            plt.ylabel('Final accuracy')
            plt.title(f'Ablation summary for {language}')
            plt.tight_layout()
            out_png = output_root / 'ablation_summary.png'
            plt.savefig(out_png)
            plt.close()
            logger.info(f"Saved ablation summary to {summary_csv} and {out_png}")
        except Exception as e:
            logger.warning(f"Failed to plot ablation summary: {e}")

        return summary

    def compare_across_languages(self, languages, bert_results, dual_results):
        # bert_results and dual_results are dict language->accuracy
        df = pd.DataFrame([{ 'language': lang, 'bert_acc': bert_results.get(lang, np.nan), 'dual_acc': dual_results.get(lang, np.nan)} for lang in languages])
        save_path = self.analysis_root / 'models_comparison.csv'
        df.to_csv(save_path, index=False)
        # Scatter plot
        plt.figure(figsize=(6,6))
        plt.scatter(df['bert_acc'], df['dual_acc'])
        plt.xlabel('BERT Accuracy')
        plt.ylabel('Dual Encoder Accuracy')
        plt.plot([0,1],[0,1], '--', color='gray')
        plt.title('BERT vs Dual Encoder Accuracy')
        plt.savefig(self.analysis_root / 'bert_vs_dual_scatter.png')
        plt.close()
        logger.info(f"Saved comparison CSV and scatter to {self.analysis_root}")

    def detailed_error_analysis(self, language, model_type='bert'):
        """Produce per-sample error analysis using saved predictions if available.
        Expects files under assets/analysis/<language> containing predictions or embeddings.
        """
        save_dir = self.analysis_root / language
        pred_csv = save_dir / f'{model_type}_predictions.csv'
        if not pred_csv.exists():
            logger.warning(f"Predictions file not found for {language}/{model_type}: {pred_csv}")
            return
        df = pd.read_csv(pred_csv)
        # compute confusion and per-sample info
        cm = confusion_matrix(df['label'], df['pred'])
        report = classification_report(df['label'], df['pred'], target_names=['Not Antonym','Antonym'], output_dict=False)
        # Save
        with open(save_dir / f'{model_type}_error_analysis.txt', 'w') as f:
            f.write('Confusion Matrix:\n')
            f.write(str(cm) + '\n\n')
            f.write('Classification Report:\n')
            f.write(report)
        # Save false positives/negatives
        fp = df[(df['label'] == 0) & (df['pred'] == 1)]
        fn = df[(df['label'] == 1) & (df['pred'] == 0)]
        fp.to_csv(save_dir / f'{model_type}_false_positives.csv', index=False)
        fn.to_csv(save_dir / f'{model_type}_false_negatives.csv', index=False)
        logger.info(f"Saved detailed error analysis for {language}/{model_type} to {save_dir}")

    def _read_pairs(self, file_path):
        data = {'word1': [], 'word2': [], 'label': []}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 3:
                    data['word1'].append(p[0])
                    data['word2'].append(p[1])
                    data['label'].append(int(p[2]))
        return pd.DataFrame(data)

    def get_pairs_df(self, language: str, split: str = 'test') -> pd.DataFrame:
        """Resolve and load pair files for a language and split.
        Tries these locations in order:
         1) data_root / <language> / <split>.txt
         2) project_root / 'dataset' / *.<split> (combines adjective/noun/verb files)
        Raises FileNotFoundError if no files found.
        """
        # 1) datasets/<language>/<split>.txt
        p1 = self.data_root / language / f"{split}.txt"
        if p1.exists():
            logger.info(f"Found {split} for {language} at {p1}")
            return self._read_pairs(p1)

        # 2) top-level 'dataset' folder (legacy English layout)
        project_root = Path(__file__).resolve().parents[1]
        legacy_dir = project_root / 'dataset'
        if legacy_dir.exists() and legacy_dir.is_dir():
            # collect files ending with .{split} (e.g., adjective-pairs.test)
            candidates = sorted([p for p in legacy_dir.iterdir() if p.is_file() and p.name.endswith(f'.{split}') or p.name.endswith(f'-{split}') or p.name.endswith(f'.{split}.txt')])
            # Some files are named like adjective-pairs.test or adjective-pairs.test.txt
            if not candidates:
                # try explicit patterns
                for stem in ['adjective-pairs', 'noun-pairs', 'verb-pairs', 'combined_antonyms', 'test']:
                    for ext in [f'.{split}', f'.{split}.txt', f'-pairs.{split}']:
                        p = legacy_dir / (stem + ext)
                        if p.exists():
                            candidates.append(p)
            if candidates:
                dfs = []
                for c in candidates:
                    try:
                        dfs.append(self._read_pairs(c))
                        logger.info(f"Loaded legacy split file {c} for {language}")
                    except Exception as e:
                        logger.warning(f"Failed to read legacy split {c}: {e}")
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    return combined

        # 3) try datasets/<language> with alternative filenames (e.g., test.txt without language folder)
        alt = self.data_root / language
        if alt.exists() and alt.is_dir():
            # attempt to find any file that looks like the split
            for p in alt.glob(f'*{split}*'):
                try:
                    return self._read_pairs(p)
                except Exception:
                    continue

        raise FileNotFoundError(f'No {split} data found for language {language} in {self.data_root} or legacy dataset dir')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/samyak/scratch/temp/multilingual_antonym_detection/config/language_config.yaml')
    parser.add_argument('--languages', nargs='+', default=None)
    parser.add_argument('--baseline', choices=['mbert', 'xlmr', 'both'], default=None,
                        help='Run baseline probe(s): mbert, xlmr or both')
    parser.add_argument('--ablation', action='store_true', help='Run ablation grid (example)')
    args = parser.parse_args()
    config = load_config(args.config)

    languages = args.languages or list(config['languages'].keys())
    analyzer = Analyzer(config)

    bert_results = {}
    dual_results = {}

    for lang in languages:
        lang_cfg = config['languages'][lang]
        bert_model = lang_cfg['bert_model']

        # Collect embeddings for analysis
        analyzer.evaluate_bert(lang, lang_cfg.get('trained_bert_path', bert_model), bert_model)
        analyzer.evaluate_dual_encoder(lang, lang_cfg.get('trained_bert_path', bert_model), bert_model)

        # Try to read per-sample predictions emitted by trainers
        analysis_dir = analyzer.analysis_root / lang
        bert_pred = analysis_dir / f'{lang}_bert_predictions.csv'
        dual_pred = analysis_dir / f'{lang}_dual_predictions.csv'
        if bert_pred.exists():
            df = pd.read_csv(bert_pred)
            bert_results[lang] = accuracy_score(df['label'], df['pred'])
        if dual_pred.exists():
            df2 = pd.read_csv(dual_pred)
            dual_results[lang] = accuracy_score(df2['label'], df2['pred'])

        # Optionally run baseline probes
        if args.baseline in ('mbert', 'both'):
            analyzer.baseline_probe(lang, 'bert-base-multilingual-cased')
        if args.baseline in ('xlmr', 'both'):
            analyzer.baseline_probe(lang, 'xlm-roberta-base')

        # Optionally run an ablation grid (small default grid)
        if args.ablation:
            grid = {
                'hidden_dim': [64, 128],
                'margin_weight': [0.1, 0.5]
            }
            analyzer.run_ablation(lang, args.config, grid)

    # Compare across languages (may have missing values)
    analyzer.compare_across_languages(languages, bert_results, dual_results)
    logger.info('Analysis completed')
