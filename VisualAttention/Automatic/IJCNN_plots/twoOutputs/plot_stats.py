import argparse
from filesReader import read_file
from tools import getConfidenceEllipse, computeDensities
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from loadData import loadData, getFormat

parser = argparse.ArgumentParser(description="Compute statistics for detected ROI")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str, default='none')
parser.add_argument("--time", "-t", help="Time length of the simulation (if simulation run on whole sample, time set to -1)", metavar="T", type=float, default=-1)
parser.add_argument("--plot-values", "-pv", help="Plot with real values", action='store_true')
parser.add_argument("--plot-ratio", "-pr", help="Plot with ratio output/ground truth", action='store_true')
args = parser.parse_args()

datasets = [
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-191011/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-170026/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-112147/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-155610/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-094442/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-181249/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230121-145833/'
]

if args.dataset != 'none':
    if not args.dataset.endswith('/'):
        args.dataset += '/'
    datasets = [args.dataset]

plot_all = False
if not args.plot_values and not args.plot_ratio:
    plot_all = True

data_to_plot_samples = {}
data_to_plot_output = {}
stat_labels = ['nb events', 'temporal density','spatial density']
for ds in datasets:
    data, references, statistics = read_file(ds)

    for d in tqdm(data):
        ref = references[d['sample name']]

        # get output decision
        stat_left = d['nb events left'] / d['nb events']
        stat_right = d['nb events right'] / d['nb events']
        if stat_right > 0.9: 
            decision = 'left'
        elif stat_left > 0.9:
            decision = 'right'
        
        # data to plot sample
        sample_labels = [l for l in ref.keys() if 'sample' in l]
        for sample in sample_labels:
            target = int(ref[sample].split('/')[-2])
            if target not in data_to_plot_samples.keys():
                data_to_plot_samples[target] = {
                    k: []
                    for k in stat_labels
                }
            
            if args.time == -1:
                for k in stat_labels:
                    data_to_plot_samples[target][k].append( statistics[ref[sample]][k] )
            else:
                ev = loadData(ref[sample])
                format_ev = getFormat(ev)

                # adapt to time length
                min_t = min(ev[:,format_ev.index('t')])
                ev[:,format_ev.index('t')] -= min_t
                ev = ev[ev[:,format_ev.index('t')] <= args.time*1e3]
                
                # get statistics
                data_to_plot_samples[target]['nb events'].append( len(ev) )
                spatial_density, temporal_density = computeDensities(ev, format_ev, div=1, combination='none')
                data_to_plot_samples[target]['spatial density'].append( spatial_density )
                data_to_plot_samples[target]['temporal density'].append( temporal_density )

            if decision in sample:
                decision_target = target
        
        # data to plot output
        if target not in data_to_plot_output.keys():
            data_to_plot_output[target] = {
                k: []
                for k in stat_labels
            }
        data_to_plot_output[target]['nb events'].append( d['nb events'] )

        ev = loadData(os.path.join(ds, d['sample name']))
        format_ev = getFormat(ev)
        spatial_density, temporal_density = computeDensities(ev, format_ev, div=4)
        data_to_plot_output[target]['spatial density'].append( spatial_density )
        data_to_plot_output[target]['temporal density'].append( temporal_density )


if plot_all or args.plot_values:

    # FIGURE GROUNDTRUTH

    # get xlim and ylim
    maximum = {k:0 for k in stat_labels}
    for target in data_to_plot_samples.keys():
        for k in stat_labels:
            if max(data_to_plot_samples[target][k]) > maximum[k]:
                maximum[k] = max(data_to_plot_samples[target][k]) 
    target_labels = sorted(list(data_to_plot_samples.keys()))
    target_labels.remove(0)
    print(maximum)

    fig, axes = plt.subplots(1,len(target_labels))
    # plot
    for i,target in enumerate(target_labels):
        axes[i].scatter([np.mean(data_to_plot_samples[target]['nb events'])], [np.mean(data_to_plot_samples[target]['temporal density'])], c='orange')
        getConfidenceEllipse(data_to_plot_samples[target]['nb events'], data_to_plot_samples[target]['temporal density'], axes[i], facecolor='orange', alpha=0.2)
        
        axe2 = axes[i].twinx()
        axe2.scatter([np.mean(data_to_plot_samples[target]['nb events'])], [np.mean(data_to_plot_samples[target]['spatial density'])], c='purple')
        getConfidenceEllipse(data_to_plot_samples[target]['nb events'], data_to_plot_samples[target]['spatial density'], axe2, facecolor='purple', alpha=0.2)

        axes[i].set_xlabel('Number of events\nTarget '+str(target))
        if i == 0:
            axes[i].set_ylabel('Temporal density')
        elif i == len(target_labels) - 1:
            axe2.set_ylabel('Spatial density')
        axes[i].set_xlim(0,maximum['nb events'])
        axe2.set_xlim(0,maximum['nb events'])
        axes[i].set_ylim(0,maximum['temporal density'])
        axe2.set_ylim(0,maximum['spatial density'])

    # legend
    axes[i].scatter([],[],c='purple', label='Spatial density')
    axes[i].scatter([],[],c='orange', label='Temporal density')
    fig.legend(loc='lower center', ncol=2)
    fig.suptitle('Statistics per target - grountruth')
    fig.tight_layout()


    # FIGURE OUTPUT

    # get xlim and ylim
    maximum = {k:0 for k in stat_labels}
    for target in data_to_plot_output.keys():
        for k in stat_labels:
            if max(data_to_plot_output[target][k]) > maximum[k]:
                maximum[k] = max(data_to_plot_output[target][k]) 
    target_labels = sorted(list(data_to_plot_output.keys()))
    target_labels.remove(0)
    print(maximum)

    fig, axes = plt.subplots(1,len(target_labels))
    # plot
    for i,target in enumerate(target_labels):
        axes[i].scatter([np.mean(data_to_plot_output[target]['nb events'])], [np.mean(data_to_plot_output[target]['temporal density'])], c='orange')
        getConfidenceEllipse(data_to_plot_output[target]['nb events'], data_to_plot_output[target]['temporal density'], axes[i], facecolor='orange', alpha=0.2)
        
        axe2 = axes[i].twinx()
        axe2.scatter([np.mean(data_to_plot_output[target]['nb events'])], [np.mean(data_to_plot_output[target]['spatial density'])], c='purple')
        getConfidenceEllipse(data_to_plot_output[target]['nb events'], data_to_plot_output[target]['spatial density'], axe2, facecolor='purple', alpha=0.2)

        axes[i].set_xlabel('Number of events\nTarget '+str(target))
        if i == 0:
            axes[i].set_ylabel('Temporal density')
        elif i == len(target_labels) - 1:
            axe2.set_ylabel('Spatial density')
        axes[i].set_xlim(0,maximum['nb events'])
        axe2.set_xlim(0,maximum['nb events'])
        axes[i].set_ylim(0,maximum['temporal density'])
        axe2.set_ylim(0,maximum['spatial density'])

    # legend
    axes[i].scatter([],[],c='purple', label='Spatial density')
    axes[i].scatter([],[],c='orange', label='Temporal density')
    fig.legend(loc='lower center', ncol=2)
    fig.suptitle('Statistics per target - saliency output')
    fig.tight_layout()


    # FIGURE GROUNDTRUTH VS OUTPUT

    # get xlim and ylim
    maximum = {k:0 for k in stat_labels}
    for target in data_to_plot_output.keys():
        for k in stat_labels:
            if np.mean(data_to_plot_output[target][k]) > maximum[k]:
                maximum[k] = np.mean(data_to_plot_output[target][k])
            if np.mean(data_to_plot_samples[target][k]) > maximum[k]:
                maximum[k] = np.mean(data_to_plot_samples[target][k])
            print(target, k, np.mean(data_to_plot_output[target][k]),np.mean(data_to_plot_samples[target][k]))
    target_labels = sorted(list(data_to_plot_output.keys()))
    target_labels.remove(0)
    print(maximum)

    fig, axes = plt.subplots(1,len(target_labels))
    # plot
    for i,target in enumerate(target_labels):
        axes[i].scatter([np.mean(data_to_plot_output[target]['nb events'])], [np.mean(data_to_plot_output[target]['temporal density'])], c='orange',marker='^')
        axes[i].scatter([np.mean(data_to_plot_samples[target]['nb events'])], [np.mean(data_to_plot_samples[target]['temporal density'])], c='orange',marker='o')
        getConfidenceEllipse(data_to_plot_output[target]['nb events'], data_to_plot_output[target]['temporal density'], axes[i], color='orange', alpha=0.1, hatch='xx')
        getConfidenceEllipse(data_to_plot_samples[target]['nb events'], data_to_plot_samples[target]['temporal density'], axes[i], facecolor='orange', alpha=0.1)
        
        axe2 = axes[i].twinx()
        axe2.scatter([np.mean(data_to_plot_output[target]['nb events'])], [np.mean(data_to_plot_output[target]['spatial density'])], c='purple', marker='^')
        axe2.scatter([np.mean(data_to_plot_samples[target]['nb events'])], [np.mean(data_to_plot_samples[target]['spatial density'])], c='purple', marker='o')
        getConfidenceEllipse(data_to_plot_output[target]['nb events'], data_to_plot_output[target]['spatial density'], axe2, color='purple', alpha=0.1, hatch='xx')
        getConfidenceEllipse(data_to_plot_samples[target]['nb events'], data_to_plot_samples[target]['spatial density'], axe2, facecolor='purple', alpha=0.1)

        axes[i].set_xlabel('Number of events\nTarget '+str(target))
        if i == 0:
            axes[i].set_ylabel('Temporal density')
        elif i == len(target_labels) - 1:
            axe2.set_ylabel('Spatial density')
        axes[i].set_xlim(0,maximum['nb events'])
        axe2.set_xlim(0,maximum['nb events'])
        axes[i].set_ylim(0,maximum['temporal density'])
        axe2.set_ylim(0,maximum['spatial density'])

    # legend
    axes[i].plot([],[],c='purple', label='Spatial density')
    axes[i].plot([],[],c='orange', label='Temporal density')
    axes[i].scatter([],[],marker='^', color="black", label='Saliency output')
    axes[i].scatter([],[],marker='o', color='black', label='Ground truth')
    fig.legend(loc='lower center', ncol=4)
    fig.suptitle('Statistics per target')
    fig.tight_layout()

if plot_all or args.plot_ratio:

    # FIGURE RATIO - GROUND TRUTH VS OUTPUT

    # get xlim and ylim
    maximum = {k:0 for k in stat_labels}
    for target in data_to_plot_output.keys():
        for k in stat_labels:
            if np.mean(data_to_plot_output[target][k]) / np.mean(data_to_plot_samples[target][k]) > maximum[k]:
                maximum[k] = 100*np.mean(data_to_plot_output[target][k]) / np.mean(data_to_plot_samples[target][k])
    target_labels = sorted(list(data_to_plot_output.keys()))
    target_labels.remove(0)
    
    fig, axes = plt.subplots()
    axe2 = axes.twinx()
    # plot
    for target in target_labels:
        axes.scatter(
            [100 * np.mean(data_to_plot_output[target]['nb events']) / np.mean(data_to_plot_samples[target]['nb events'])],
            [100 * np.mean(data_to_plot_output[target]['temporal density']) / np.mean(data_to_plot_samples[target]['temporal density'])], 
            c='orange',marker="x"
        )        
        axe2.scatter(
            [100 * np.mean(data_to_plot_output[target]['nb events']) / np.mean(data_to_plot_samples[target]['nb events'])], 
            [100 * np.mean(data_to_plot_output[target]['spatial density']) / np.mean(data_to_plot_samples[target]['spatial density'])], 
            c='purple', marker="x"
        )

    axes.set_xlabel('Number of events (in %)')
    axes.set_ylabel('Temporal density (in %)')
    axe2.set_ylabel('Spatial density (in %)')
    axes.set_xlim(0, 100) #maximum['nb events'])
    axe2.set_xlim(0, 100) #maximum['nb events'])
    axes.set_ylim(0, 100) #maximum['temporal density'])
    axe2.set_ylim(0, 100) #maximum['spatial density'])

    # legend
    axes.scatter([],[],c='purple', marker="x", label='Spatial density')
    axes.scatter([],[],c='orange', marker="x", label='Temporal density')
    axes.legend(loc='upper right', ncol=1)
    fig.suptitle('Statistics per target')
    fig.tight_layout()


plt.show()