import stumpy
import numpy as np

def past_only_matrix_profile(T, m, start_idx=0, normalize=False):
    n = len(T)
    profile = np.full(n, 0.0)
    profile_idx = np.full(n, -1)

    for i in range(m, n):
        if i > start_idx:
            Q = T[i-m:i]
            D = stumpy.mass(Q, T[:i-m], normalize=normalize)
            # D = stumpy.mass(Q, T[:int(0.4*len(T))], normalize=normalize)

            # Find best match in the past only
            min_idx = np.argmin(D)
            profile[i] = D[min_idx]
            profile_idx[i] = min_idx

    return profile, profile_idx