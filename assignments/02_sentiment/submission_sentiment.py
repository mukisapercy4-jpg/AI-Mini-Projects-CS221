import random
import collections
from typing import List, Tuple, Dict, Callable

def extract_word_features(x: str) -> Dict[str, float]:
    features = collections.defaultdict(float)
    for word in x.split():
        features[word] += 1.0
    return dict(features)

def dot_product(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    if len(d1) > len(d2):
        d1, d2 = d2, d1
    return sum(val * d2.get(key, 0.0) for key, val in d1.items())

def increment_sparse_vector(d1: Dict[str, float], scale: float, d2: Dict[str, float]) -> None:
    for key, val in d2.items():
        d1[key] = d1.get(key, 0.0) + scale * val

def learn_predictor_sgd(train_data, dev_data, num_epochs=15, eta=0.05):
    w = {}
    train_errors, dev_errors = [], []
    
    def evaluate(w, data):
        errors = 0
        for phi, y in data:
            score = dot_product(w, phi)
            pred = 1 if score >= 0 else -1
            if pred != y:
                errors += 1
        return errors / len(data)

    for epoch in range(num_epochs):
        shuffled = list(train_data)
        random.shuffle(shuffled)
        for phi, y in shuffled:
            margin = dot_product(w, phi) * y
            if margin < 1.0:
                increment_sparse_vector(w, eta * y, phi)
        train_errors.append(evaluate(w, train_data))
        dev_errors.append(evaluate(w, dev_data))
    return w, train_errors, dev_errors

def extract_character_features(n: int) -> Callable[[str], Dict[str, float]]:
    def extract(x: str) -> Dict[str, float]:
        clean_x = x.replace(" ", "").replace("\t", "")
        features = collections.defaultdict(float)
        for i in range(len(clean_x) - n + 1):
            features[clean_x[i:i+n]] += 1.0
        return dict(features)
    return extract

def kmeans_clustering(points, k=2, num_iters=10, seed=42):
    random.seed(seed)
    if not points: return [], [], []
    
    centroids = [dict(random.choice(points)) for _ in range(k)]
    assignments = [-1] * len(points)
    example_norms = [sum(v*v for v in ex.values()) for ex in points]
    distortions = []

    for _ in range(num_iters):
        centroid_norms = [sum(v*v for v in c.values()) for c in centroids]
        new_assignments = []
        
        for i, ex in enumerate(points):
            best_k, min_dist = 0, float('inf')
            for c_idx in range(k):
                c = centroids[c_idx]
                dot = sum(v * c.get(f, 0.0) for f, v in ex.items())
                dist = example_norms[i] + centroid_norms[c_idx] - 2 * dot
                if dist < min_dist:
                    min_dist, best_k = dist, c_idx
            new_assignments.append(best_k)

        if new_assignments == assignments: break
        assignments = new_assignments

        new_centroids = [{} for _ in range(k)]
        counts = [0] * k
        for i, ex in enumerate(points):
            c_idx = assignments[i]
            counts[c_idx] += 1
            for f, v in ex.items():
                new_centroids[c_idx][f] = new_centroids[c_idx].get(f, 0.0) + v
                
        for c_idx in range(k):
            if counts[c_idx] > 0:
                for f in new_centroids[c_idx]:
                    new_centroids[c_idx][f] /= counts[c_idx]
            else:
                new_centroids[c_idx] = dict(random.choice(points))
        centroids = new_centroids

        total_loss = 0.0
        final_c_norms = [sum(v*v for v in c.values()) for c in centroids]
        for i, ex in enumerate(points):
            c = centroids[assignments[i]]
            dot = sum(v * c.get(f, 0.0) for f, v in ex.items())
            total_loss += example_norms[i] + final_c_norms[assignments[i]] - 2 * dot
        distortions.append(total_loss)

    return centroids, assignments, distortions
