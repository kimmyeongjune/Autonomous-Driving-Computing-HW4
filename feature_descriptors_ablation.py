
import numpy as np
import cv2
from m2bk import *
import time
from matplotlib import pyplot as plt


def visualize_matches(image1, kp1, image2, kp2, matches, title):
    result = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(result)
    plt.show()


def feature_experiment(image1, image2, method="ORB"):
    if method == "ORB":
        feature_detector = cv2.ORB_create()
    elif method == "SIFT":
        feature_detector = cv2.SIFT_create()
    elif method == "BRISK":
        feature_detector = cv2.BRISK_create()
    else:
        raise ValueError("Invalid method selected!")

    start_time = time.time()

    kp1, des1 = feature_detector.detectAndCompute(image1, None)
    kp2, des2 = feature_detector.detectAndCompute(image2, None)

    if method in ["ORB", "BRISK"]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    end_time = time.time()

    exec_time = end_time - start_time

    return kp1, kp2, matches, exec_time

dataset_handler = DatasetHandler()
image1 = dataset_handler.images[0] 
image2 = dataset_handler.images[1]  

methods = ["ORB", "SIFT", "BRISK"]

results = {}
for method in methods:
    kp1, kp2, matches, exec_time = feature_experiment(image1, image2, method=method)
    results[method] = {
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
        "exec_time": exec_time,
    }
    print(f"{method} - Number of matches: {len(matches)}, Execution time: {exec_time:.4f} seconds")

    visualize_matches(image1, kp1, image2, kp2, matches[:20], f"{method} Matches")

for method in methods:
    print(f"{method}:")
    print(f"- Number of Matches: {len(results[method]['matches'])}")
    print(f"- Execution Time: {results[method]['exec_time']:.4f} seconds")
