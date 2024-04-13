import numpy as np

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
            # angles = np.mod(angles, 360)
            # previous_angles = np.mod(previous_angles, 360)
            d_rotation = (r * np.pi / 180) * np.sqrt(np.sum((angles - previous_angles) ** 2))
        elif mode == "radian":
            # angles = np.mod(angles, 2 * np.pi)
            # previous_angles = np.mod(previous_angles, 2 * np.pi)
            d_rotation = r * np.sqrt(np.sum((angles - previous_angles) ** 2))
        else:
            raise ValueError("Invalid mode. Mode should be either 'degree' or 'radian'.")

        d_translation = np.sqrt(np.sum((translation - previous_translation) ** 2))

        return d_rotation + d_translation

    except Exception as e:
        print(f"An error occurred: {e}")
