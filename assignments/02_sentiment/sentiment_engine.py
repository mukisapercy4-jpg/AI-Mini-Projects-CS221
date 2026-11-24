"""
Sentiment Engine & Synthetic Dataset Utilities
Provides labeled movie review samples and feature utilities for CS221 Assignment 2.
"""

import random
from typing import List, Tuple, Dict, Any

SAMPLE_REVIEWS = [
    ("This movie was an absolute masterpiece with stellar performances and brilliant directing", 1),
    ("Incredible emotional depth and captivating storytelling throughout the entire film", 1),
    ("A triumph of modern cinema, beautifully paced and thoroughly engaging", 1),
    ("Loved the dialogue and the breathtaking visual cinematography", 1),
    ("Exceptional soundtrack and compelling character arcs make this a must-watch", 1),
    ("A wonderful, heartwarming experience that left everyone in tears of joy", 1),
    ("Brilliant acting and sharp witty humor make it an instant classic", 1),
    ("Superb direction and fantastic writing from start to finish", 1),
    ("A truly remarkable film that exceeds all high expectations", 1),
    ("Tremendous suspense, thrilling twists, and unforgettable climax", 1),
    ("Dreadful acting, abysmal writing, and complete waste of time", -1),
    ("Boring, disjointed, and utterly painful to sit through until the end", -1),
    ("Terrible script with cardboard characters and zero emotional resonance", -1),
    ("Horrendous plot holes and laughably bad special effects ruin everything", -1),
    ("A clumsy, uninspired mess that insults the intelligence of the audience", -1),
    ("Completely disappointed by the awful directing and sluggish pacing", -1),
    ("Lacks originality, poorly acted, and hopelessly predictable from the opening scene", -1),
    ("One of the worst films of the year, utterly unwatchable and tedious", -1),
    ("Dull, lifeless, and painfully monotonous dialogue throughout", -1),
    ("A chaotic disaster with no coherent storyline whatsoever", -1),
]

def get_movie_review_dataset(
    num_train: int = 60,
    num_dev: int = 20,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Generates training and validation datasets with positive (+1) and negative (-1) reviews.
    """
    random.seed(seed)
    # Augment base pool
    pool = list(SAMPLE_REVIEWS)
    # Duplicate with minor noise to expand size
    expanded_pool = []
    for text, label in pool:
        expanded_pool.append((text, label))
        words = text.split()
        if len(words) > 4:
            sub = " ".join(words[:len(words)-2])
            expanded_pool.append((sub, label))

    random.shuffle(expanded_pool)
    train_set = expanded_pool[:num_train]
    dev_set = expanded_pool[num_train:num_train + num_dev]
    return train_set, dev_set
