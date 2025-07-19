#!/usr/bin/env python3
"""
Test connectivity and download a minimal dataset offline if needed
"""

import requests
import json
import os
from pathlib import Path

def test_connectivity():
    """Test if we can reach external services"""
    test_urls = [
        "https://httpbin.org/get",
        "https://github.com",
        "http://api.conceptnet.io/query?node=/c/en/hot&rel=/r/Antonym&limit=1"
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {url}: {response.status_code}")
            return True
        except Exception as e:
            print(f"✗ {url}: {e}")
    
    return False

def create_synthetic_dataset():
    """Create a synthetic antonym dataset for testing when offline"""
    synthetic_antonyms = {
        'german': [
            ('heiß', 'kalt'), ('groß', 'klein'), ('gut', 'schlecht'), 
            ('hell', 'dunkel'), ('schnell', 'langsam'), ('jung', 'alt'),
            ('stark', 'schwach'), ('neu', 'alt'), ('reich', 'arm'),
            ('glücklich', 'traurig'), ('leicht', 'schwer'), ('hoch', 'niedrig'),
            ('weit', 'nah'), ('dick', 'dünn'), ('laut', 'leise'),
        ],
        'french': [
            ('chaud', 'froid'), ('grand', 'petit'), ('bon', 'mauvais'),
            ('clair', 'sombre'), ('rapide', 'lent'), ('jeune', 'vieux'),
            ('fort', 'faible'), ('nouveau', 'ancien'), ('riche', 'pauvre'),
            ('heureux', 'triste'), ('léger', 'lourd'), ('haut', 'bas'),
            ('loin', 'près'), ('épais', 'mince'), ('bruyant', 'silencieux'),
        ],
        'spanish': [
            ('caliente', 'frío'), ('grande', 'pequeño'), ('bueno', 'malo'),
            ('claro', 'oscuro'), ('rápido', 'lento'), ('joven', 'viejo'),
            ('fuerte', 'débil'), ('nuevo', 'viejo'), ('rico', 'pobre'),
            ('feliz', 'triste'), ('ligero', 'pesado'), ('alto', 'bajo'),
            ('lejos', 'cerca'), ('grueso', 'delgado'), ('ruidoso', 'silencioso'),
        ],
        'italian': [
            ('caldo', 'freddo'), ('grande', 'piccolo'), ('buono', 'cattivo'),
            ('chiaro', 'scuro'), ('veloce', 'lento'), ('giovane', 'vecchio'),
            ('forte', 'debole'), ('nuovo', 'vecchio'), ('ricco', 'povero'),
            ('felice', 'triste'), ('leggero', 'pesante'), ('alto', 'basso'),
            ('lontano', 'vicino'), ('spesso', 'sottile'), ('rumoroso', 'silenzioso'),
        ],
        'russian': [
            ('горячий', 'холодный'), ('большой', 'маленький'), ('хороший', 'плохой'),
            ('светлый', 'тёмный'), ('быстрый', 'медленный'), ('молодой', 'старый'),
            ('сильный', 'слабый'), ('новый', 'старый'), ('богатый', 'бедный'),
            ('счастливый', 'грустный'), ('лёгкий', 'тяжёлый'), ('высокий', 'низкий'),
            ('далёкий', 'близкий'), ('толстый', 'тонкий'), ('громкий', 'тихий'),
        ]
    }
    
    # Create datasets directory
    datasets_dir = Path("../datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    for lang, pairs in synthetic_antonyms.items():
        lang_dir = datasets_dir / lang
        lang_dir.mkdir(exist_ok=True)
        
        # Create combined file
        combined_file = lang_dir / "combined_antonyms.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for word1, word2 in pairs:
                f.write(f"{word1}\t{word2}\n")
        
        # Create splits
        train_split = pairs[:int(len(pairs) * 0.7)]
        val_split = pairs[int(len(pairs) * 0.7):int(len(pairs) * 0.85)]
        test_split = pairs[int(len(pairs) * 0.85):]
        
        # Save splits
        for split_name, split_data in [('train', train_split), ('val', val_split), ('test', test_split)]:
            split_file = lang_dir / f"{split_name}.txt"
            with open(split_file, 'w', encoding='utf-8') as f:
                for word1, word2 in split_data:
                    f.write(f"{word1}\t{word2}\n")
        
        print(f"Created synthetic dataset for {lang}: {len(pairs)} pairs")
        print(f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

if __name__ == "__main__":
    print("Testing connectivity...")
    if test_connectivity():
        print("✓ Network is available - can proceed with real downloads")
    else:
        print("✗ Network issues detected - creating synthetic datasets for testing")
        create_synthetic_dataset()
        print("\n✓ Synthetic datasets created successfully!")
        print("You can now proceed with BERT downloads and training")
