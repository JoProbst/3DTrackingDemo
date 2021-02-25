import numpy as np


def cosineSimilarity(A, B):
    assert len(A) == len(B), "Vectors need to be of same length"
    d = np.dot(A, B)
    s = np.around(np.linalg.norm(A) * np.linalg.norm(B), decimals=3)
    return d / s