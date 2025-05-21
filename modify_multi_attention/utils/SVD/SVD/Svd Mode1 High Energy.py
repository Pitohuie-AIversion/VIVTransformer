import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import svd
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity


def load_difference_matrix(file_path):
    return np.loadtxt(file_path, delimiter=",")


def perform_svd(difference_matrix):
    U, S, Vh = svd(difference_matrix, full_matrices=False)
    return U, S, Vh


def extract_mode1_2D(U, S, Vh):
    return S[0] * np.outer(U[:, 0], Vh[0, :])


def extract_high_energy_coords(mode1_2D, percentile=95):
    threshold = np.percentile(np.abs(mode1_2D), percentile)
    coords = np.argwhere(np.abs(mode1_2D) >= threshold)  # [(y, x)]
    return coords


def plot_difference_with_overlay(difference_matrix, high_energy_coords, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(difference_matrix, cmap='hot')
    plt.colorbar(im, ax=ax)
    ax.set_title("Difference Map with Mode-1 High Energy Overlay")

    for (y, x) in high_energy_coords:
        ax.scatter(x, y, color='blue', marker='x')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_single_file(file_path, output_dir):
    diff = load_difference_matrix(file_path)
    U, S, Vh = perform_svd(diff)
    mode1_2D = extract_mode1_2D(U, S, Vh)
    high_energy_coords = extract_high_energy_coords(mode1_2D)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, os.path.basename(file_path).replace(".csv", "_mode1_overlay.png"))
    plot_difference_with_overlay(diff, high_energy_coords, out_file)


def batch_process(input_dir, output_dir):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    for f in files:
        print(f"Processing: {f}")
        process_single_file(f, output_dir)


if __name__ == "__main__":
    input_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/difference_results"
    output_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/svd_results/mode1_overlay"
    batch_process(input_dir, output_dir)
