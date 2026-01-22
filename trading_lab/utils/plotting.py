import matplotlib.pyplot as plt


def plot_series(series, title: str, ylabel: str = ""):
    plt.figure(figsize=(12, 5))
    plt.plot(series)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
