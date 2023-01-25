import matplotlib.pyplot as plt
from os import path, makedirs

def handle_options(ax, options):
    if "xticks" in options.keys():
        plt.setp(ax.get_xticklabels(), visible=options.pop("xticks"))
    if "yticks" in options.keys():
        plt.setp(ax.get_yticklabels(), visible=options.pop("yticks"))
    if "xlabel" in options.keys():
        ax.set_xlabel(options.pop("xlabel"))
    if "ylabel" in options.keys():
        ax.set_ylabel(options.pop("ylabel"))
    if "ylim" in options.keys():
        ax.set_ylim(options.pop("ylim"))
    if "xlim" in options.keys():
        ax.set_xlim(options.pop("xlim"))

def scatterplot(ax, panel):
    handle_options(ax, panel)
    data = panel.pop("data")
    timestamps = []
    neurons = []
    for n, ts in enumerate(data):
        if len(ts) > 0:
            for t in ts: 
                timestamps.append(t)
                neurons.append(n)
    ax.scatter(timestamps, neurons, s=10, facecolors='none', edgecolors="gray")

def Figure(infos):
    n_panels = 0
    for k in infos.keys():
        if "plot" in k:
            n_panels += 1
    width, height = 6, 2 * n_panels + 1.2
    fig, axes = plt.subplots(n_panels, figsize=(width, height))

    for k, panel in infos.items():
        if "plot" in k:
            idx = int(k[-1])-1
            scatterplot( axes[idx], panel)
    
    options = infos["figure"]
    if "title" in options:
        fig.suptitle(options["title"], fontsize="large")
    if "annotations" in options:
        plt.figtext(0.01, 0.01, options["annotations"], fontsize=6, verticalalignment='bottom')

    if "save" in options and options["save"]:
        try:
            dirname = path.dirname(options["saveas"])
            if dirname and not path.exists(dirname):
                makedirs(dirname)
            fig.savefig(options["saveas"])
        except KeyError:
            raise 'Error: the figure was not saved as "saveas" was not defined'
    plt.tight_layout()
    