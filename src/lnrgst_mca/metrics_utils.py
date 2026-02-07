import numpy as np
from fsl.transform import affine
from itertools import product
from collections import Counter
from itertools import combinations


def jaccard_similarity(arr1, arr2):

    set1 = set(arr1)
    set2 = set(arr2)

    return len(set1.intersection(set2)) / len(set1.union(set2))


def frequency_vectors(arr1, arr2):

    counter1 = Counter(arr1)
    counter2 = Counter(arr2)

    all_categories = set(list(arr1) + list(arr2))

    vector1 = [counter1.get(category, 0) for category in all_categories]
    vector2 = [counter2.get(category, 0) for category in all_categories]

    return vector1, vector2


def cosine_similarity(A, B):
    """Compute cosine similarity, guarding against zero-norm vectors."""
    A = np.asarray(A)
    B = np.asarray(B)
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    if norm_A == 0 or norm_B == 0:
        return 0.0
    return float(dot_product) / float(norm_A * norm_B)


def print_metrics(groups, all_ravel):

    a = list(range(len(groups)))
    combos = list(combinations(a, 2))

    for combo in combos:

        g1, g2 = combo[0], combo[1]
        print(f"{groups[g1]} vs {groups[g2]}")

        vector1, vector2 = frequency_vectors(all_ravel[g1], all_ravel[g2])

        print(f"cosine similarity: {cosine_similarity(vector1, vector2) :4f}")

        print(f"jaccard similarity: {jaccard_similarity(all_ravel[g1],all_ravel[g2]) :4f}")

        print("*********\n")


def decompose_tensor(tensor: np.ndarray) -> tuple:
    """
    Decompose affine matrices into (scales, translations, angles, shears)

    Parameters:
        tensor: np.ndarray
            - (N, 4, 4) 
            - (N, M, 4, 4)
    Returns:
        scales: np.ndarray (N, M, 3)
        translations: np.ndarray (N, M, 3)
        angles: np.ndarray (N, M, 3)
        shears: np.ndarray (N, M, 3)

    """

    if tensor.ndim == 3 and tensor.shape[1:] == (4, 4):
        tensor = tensor[:, np.newaxis, :, :]
    elif tensor.ndim != 4 or tensor.shape[2:] != (4, 4):
        raise ValueError("Input tensor must be of shape (N, 4, 4) or (N, M, 4, 4)")

    n, m, _, _ = tensor.shape
    scales = np.zeros((n, m, 3), dtype=tensor.dtype)
    translations = np.zeros((n, m, 3), dtype=tensor.dtype)
    angles = np.zeros((n, m, 3), dtype=tensor.dtype)
    shears = np.zeros((n, m, 3), dtype=tensor.dtype)

    for i in range(n):
        for j in range(m):
            s, t, a, sh = affine.decompose(tensor[i, j])
            scales[i, j] = s
            translations[i, j] = t
            angles[i, j] = np.degrees(a)
            shears[i, j] = sh

    return (scales, translations, angles, shears)


def framewise_displacment_all_subjects_vectorized(
        translation_mca: np.ndarray,
        angles_mca: np.ndarray,
        translation_ieee: np.ndarray = None,
        angles_ieee: np.ndarray = None,
        r: float = 50.0,
        mode: str = "degree",       
) -> np.ndarray:
    """
    Vectorized computation of framewise displacement for all subjects and MCA runs.
    Computes ||∆t|| + r * ||∆θ|| for each subject and MCA run, where ∆t is the translation difference and ∆θ is the angle difference.
    Relative to per-subject IEEE reference (if provided) or zero if not.

    Inputs:
        translation_mca: (N, M, 3)
        angles_mca:      (N, M, 3)
        translation_ieee: (N, 1, 3) or None
        angles_ieee:      (N, 1, 3) or None
    """

    if translation_ieee is not None and translation_ieee.ndim != 3:
        raise ValueError("translation_ieee must be a 3D array of shape (n_subjects, 1, 3) or None")
    if angles_ieee is not None and angles_ieee.ndim != 3:
        raise ValueError("angles_ieee must be a 3D array of shape (n_subjects, 1, 3) or None")
    
    n, n_mca, dims_t = translation_mca.shape
    _, _, dims_a = angles_mca.shape
    if dims_t != 3 or dims_a != 3:
        raise ValueError("translation_mca and angles_mca must have last dimension of size 3 (x, y, z)")
    
    if translation_ieee is None:
        translation_ieee = np.zeros((n, 1, 3), dtype=translation_mca.dtype)

    if angles_ieee is None:
        angles_ieee = np.zeros((n, 1, 3), dtype=angles_mca.dtype)

    d_translation = np.linalg.norm(translation_mca - translation_ieee, axis=2)
    d_angles = np.linalg.norm(angles_mca - angles_ieee, axis=2)

    if mode == "degree":
        d_rotation = (r * np.pi / 180) * d_angles
    elif mode == "radian":
        d_rotation = r * d_angles
    else:
        raise ValueError("Invalid mode. Mode should be either 'degree' or 'radian'.")
    
    return d_translation + d_rotation
    


def mean_absolute_difference(FD_mca, FD_ieee):

    """Calculate Mean Absolute Difference (MAD) between MCA and IEEE framewise displacement across all subjects and runs.
    
    Inputs:
        FD_mca: (N_subjects, N_MCA) array of framewise displacement for MCA runs
        FD_ieee: (N_subjects, 1)  or (N_subjects,) array of framewise displacement for IEEE reference.
    
    Returns:
        mad: (N_subjects,) array of mean absolute differences for each subject
    """

    if FD_ieee.ndim == 1:
        FD_ieee = FD_ieee[:, np.newaxis]
    
    
    return np.mean(np.abs(FD_mca - FD_ieee), axis=1)
