import os
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from math import factorial

# --- Config ---
WATCH_FOLDER = os.path.join(os.getcwd(), "data")
FILE_EXTENSION = ".csv"
WINDOW_SIZE = 11
POLY_ORDER = 3

# --- "Custom" Savitzky-Golay Implementation ---
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter."""
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# --- File Watcher ---
class RamanHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(FILE_EXTENSION):
            print(f"[+] New file detected: {event.src_path}")
            time.sleep(1)  # wait for it to be fully written
            self.process_file(event.src_path)

    def process_file(self, path):
        try:
            df = pd.read_csv(path)
            x = df.iloc[:, 0].to_numpy()
            y = df.iloc[:, 1].to_numpy()

            y_filtered = savitzky_golay(y, window_size=WINDOW_SIZE, order=POLY_ORDER)

            # Save filtered output
            filtered_path = path.replace(".csv", "_filtered.csv")
            pd.DataFrame({"Wavelength": x[:len(y_filtered)], "Filtered Intensity": y_filtered}).to_csv(filtered_path, index=False)
            print(f"[+] Filtered data saved to: {filtered_path}")

            # Plot
            plt.figure()
            plt.plot(x, y, label="Raw")
            plt.plot(x[:len(y_filtered)], y_filtered, label="Filtered", linestyle="--")
            plt.xlabel("Wavelength")
            plt.ylabel("Intensity")
            plt.title("Raman Spectra (Filtered)")
            plt.legend()
            plt.show()

        except Exception as e:
            print("[-] Failed to process file:", e)

# --- Run File Watcher ---
if __name__ == "__main__":
    print(f"[*] Watching folder: {WATCH_FOLDER}")
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    observer = Observer()
    observer.schedule(RamanHandler(), path=WATCH_FOLDER, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
