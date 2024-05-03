import numpy as np
from itertools import product

# from pathlib import Path
# import significantdigits as sd
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


def Cosine_similarity(A, B):

    dot_product = np.dot(A, B)

    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    return dot_product / (norm_A * norm_B)


def print_metrics(groups, all_ravel):

    a = list(range(len(groups)))
    combos = list(combinations(a, 2))

    for combo in combos:

        g1, g2 = combo[0], combo[1]
        print(f"{groups[g1]} vs {groups[g2]}")

        vector1, vector2 = frequency_vectors(all_ravel[g1], all_ravel[g2])

        print(f"cosine similarity: {Cosine_similarity(vector1, vector2) :4f}")

        print(f"jaccard similarity: {jaccard_similarity(all_ravel[g1],all_ravel[g2]) :4f}")

        print("*********\n")


def framewise_displacement(translation, angles, previous_translation=np.array([0, 0, 0]), previous_angles=np.array([0, 0, 0]), r=50, mode="degree"):

    try:
        if mode == "degree":
            d_rotation = (r * np.pi / 180) * np.sqrt(np.sum((angles - previous_angles) ** 2))

        elif mode == "radian":
            d_rotation = r * np.sqrt(np.sum((angles - previous_angles) ** 2))

        else:
            raise ValueError("Invalid mode. Mode should be either 'degree' or 'radian'.")

        d_translation = np.sqrt(np.sum((translation - previous_translation) ** 2))

        return d_rotation + d_translation

    except Exception as e:
        print(f"An error occurred: {e}")


def FD_all_subjects(translation_mca, angles_mca, translation_ieee=None, angles_ieee=None):

    n, n_mca, dims = translation_mca.shape
    FD_results = np.zeros((n, n_mca))

    if translation_ieee is None:
        translation_ieee = np.zeros((n, dims))

    if angles_ieee is None:
        angles_ieee = np.zeros((n, dims))

    for i, j in product(range(n), range(n_mca)):
        FD_results[i, j] = framewise_displacement(translation_mca[i, j], angles_mca[i, j], translation_ieee[i], angles_ieee[i])

    return FD_results


def random_point_on_sphere_surface(radius, center):
    np.random.seed(0)  # Set the seed for the random number generator

    x, y, z = np.random.uniform(-1, 1, 3)

    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm, y_norm, z_norm = x / norm, y / norm, z / norm

    x_surface = radius * x_norm + center[0]
    y_surface = radius * y_norm + center[1]
    z_surface = radius * z_norm + center[2]

    return np.array((x_surface, y_surface, z_surface, 1))


def improved_FD(transformation, n=50):

    for i in range(n):
        distances = np.zeros((n,))
        reversed_transformation = np.linalg.inv(transformation)
        p = random_point_on_sphere_surface(50, (0, 0, 0))
        intial_p = np.dot(reversed_transformation, p)
        distances[i] = np.linalg.norm(p - intial_p)

    return np.mean(distances)


def FD_all_subjects_improved(transformations, n_random=50):
    n, n_mca, _, _ = transformations.shape
    FD_results = np.zeros((n, n_mca))

    for i, j in product(range(n), range(n_mca)):
        FD_results[i, j] = improved_FD(transformations[i, j], n_random)

    return FD_results


# def improved_fd(translations, angles, shears, scales):
#     d_rotation = 2* np.sqrt(np.sum((angles) ** 2))
#     d_translation = np.sqrt(np.sum((translations)**2))
#     d_shears = np.sqrt(np.sum((shears)**2)) * 400
#     d_scales = np.sqrt(np.sum((scales)**2)) * 20
#     print("translations", d_rotation)
#     print("angles", d_translation)
#     print("shears", d_shears)
#     print("scales", d_scales)
#     return d_shears + d_scales + d_translation + d_rotation


# def FD_all_subjects_improved(translations, angles, shears, scales):
#     n, n_mca, _ = translations.shape
#     FD_results = np.zeros((n, n_mca))

#     for i,j in product(range(n), range(n_mca)):
#         FD_results[i,j] = improved_fd(translations[i,j], angles[i,j], shears[i,j], scales[i,j])
#     return FD_results
