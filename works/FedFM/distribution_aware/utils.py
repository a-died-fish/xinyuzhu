import numpy as np
import sys


def get_distribution_difference(client_cls_counts, participation_clients, metric):
    global_distribution = np.ones(client_cls_counts.shape[1])/client_cls_counts.shape[1]
    local_distributions = client_cls_counts[np.array(participation_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]
    similarity_scores = local_distributions.dot(global_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(global_distribution))
    if metric=='cosine':
        difference = 1.0 - similarity_scores + 0.1
    elif metric=='only_iid':
        difference = np.where(similarity_scores>0.99, 0.01, float('inf'))
    return difference
