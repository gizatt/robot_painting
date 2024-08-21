import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def fit_spline_and_calculate_rss(x, y, smoothing=1):
    # Fit a B-spline to the data
    print(len(x))
    tck, u = splprep([x, y], s=smoothing, k=3)

    # Evaluate the B-spline at these parameters
    new_points = splev(u, tck)

    # Calculate the Residual Sum of Squares (RSS)
    rss = np.sum((x - new_points[0])**2 + (y - new_points[1])**2)

    return rss

def process_npz_file(file_path, smoothing):
    # Load the npy file
    lines = np.load(file_path).values()
    print(f"Loaded {len(lines)} lines from {file_path}")
    print(lines)

    # Collect RSS errors for each line
    rss_errors = []
    for line in lines:
        # Cut out unused thickness value.
        if len(line) == 0:
            continue
        line = np.unique(line[:, :2], axis=1)
        if len(line) < 10:
            print(f"Rejecting line that's too long ({len(line)})")
            continue
        
        rss_errors.append(fit_spline_and_calculate_rss(line[:, 0], line[:, 1], smoothing=smoothing))

    return rss_errors

def plot_histogram(rss_errors):
    plt.hist(rss_errors, bins=20, edgecolor='black')
    plt.title('Histogram of RSS Errors')
    plt.xlabel('RSS Error')
    plt.ylabel('Frequency')
    plt.show()

def main():
    file_path = 'strokes.npz'  # Replace with your actual file path

    # Process the npz file
    rss_errors = process_npz_file(file_path, smoothing=1.0)

    # Plot the histogram of RSS errors
    plot_histogram(rss_errors)

if __name__ == "__main__":
    main()