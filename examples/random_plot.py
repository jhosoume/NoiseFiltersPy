import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from NoiseFiltersPy.RandomInjector import RandomInjector

artificial_dataset = pd.read_csv("examples/datasets/ArtificialDataset.csv", header = None)
artificial_dataset = artificial_dataset.rename(columns = {0: "At1", 1: "At2", 2: "Class"})

ri = RandomInjector(artificial_dataset.iloc[:, :2].values, artificial_dataset.Class.values, rate = 0.3)
ri.generate()
artificial_dataset.Class = ri.labels
artificial_dataset["color"] = [1 if indx in ri.noise_indx else 0 for indx in range(artificial_dataset.shape[0])]

sns.set_theme(context = "paper", 
    style = "white",
    font_scale = 1.5, 
    rc = {
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "axes.grid": True,
    "axes.grid.which": "major",

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

    "lines.markersize": 12,
})

fig, scatter = plt.subplots(figsize = (10,8), dpi = 100)

markers = {1: "^", 2: "o"}
scatter = sns.scatterplot(
    data = artificial_dataset, 
    y = "At1", 
    x = "At2", 
    style = "Class", 
    hue="color",
    palette=["black", "red"],
    legend = False,
    markers = markers
)

# scatter.set_title("Artificial Dataset", fontsize = 20, y = 1.05)
scatter.set(ylim = (0.95, 1.85))
scatter.set(xlim = (2.9, 5.2))

plt.tight_layout()
plt.savefig("examples/random_artificial.png")