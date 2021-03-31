import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

artificial_dataset = pd.read_csv("examples/datasets/ArtificialDataset.csv", header = None)
artificial_dataset = artificial_dataset.rename(columns = {0: "At1", 1: "At2", 2: "Class"})

sns.set_theme(context = "paper", 
    style = "white",
    font_scale = 1.5, 
    rc = {
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "axes.grid": True,
    "axes.grid.which": 'both',

    "patch.antialiased": True,

    "xtick.bottom": True,
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    "xtick.labelsize": "large",

    "ytick.left": True,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    "ytick.labelsize": "large",

    "font.family": "serif",
    "font.serif": "Computer Modern Roman",

    "path.simplify": True,

    "lines.markersize": 10,

})

markers = {1: "^", 2: "o"}
sns.scatterplot(data = artificial_dataset, y = "At1", x = "At2", style = "Class", markers = markers)

print(plt.style.available)
print(sns.axes_style())
plt.tight_layout()
plt.savefig("examples/original_artificial.png")