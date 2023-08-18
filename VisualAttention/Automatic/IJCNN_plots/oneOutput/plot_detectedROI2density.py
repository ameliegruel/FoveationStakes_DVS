import argparse
from filesReader import read_file
from tools import getConfidenceEllipse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Compute statistics for detected ROI")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str, default='none')
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

data_to_plot = {}
stat_labels = ['nb events', 'temporal density','spatial density']
for ds in datasets:
    data, references, statistics = read_file(ds)

    for d in data:
        ref = references[d['sample name']]
        if 'shift' in ref.keys():
            sh = int(ref['shift'])
            if sh not in data_to_plot.keys():
                data_to_plot[sh] = {
                    decision: {
                        number: {k: [] for k in stat_labels}
                        for number in ['first', 'second']
                    } for decision in ['correct', 'incorrect']
                }

            stat_left = d['nb events left'] / d['nb events']
            stat_right = d['nb events right'] / d['nb events']
            # if the left sample is shifted, the right sample should be detected first and events should majorly be present on the right
            # if the right sample is shifted, the left sample should be detected first and events should majorly be present on the left
            if (ref['shifted'] == 'left' and stat_right > 0.9) or (ref['shifted'] == 'right' and stat_left > 0.9):
                decision = 'correct'
            else:
                decision = 'incorrect'
            
            # if the left sample is shifted, the right sample should be detected first and events should majorly be present on the right
            if ref['shifted'] == 'left' :
                for k in stat_labels:
                    data_to_plot[sh][decision]['first'][k].append( statistics[ref['right sample']][k] )
                    data_to_plot[sh][decision]['second'][k].append( statistics[ref['left sample']][k] )
            
            # if the right sample is shifted, the left sample should be detected first and events should majorly be present on the left
            elif ref['shifted'] == 'right' :
                for k in stat_labels:
                    data_to_plot[sh][decision]['first'][k].append( statistics[ref['left sample']][k] )
                    data_to_plot[sh][decision]['second'][k].append( statistics[ref['right sample']][k] )
        
        else: 
            if 0 not in data_to_plot.keys():
                data_to_plot[0] = {
                    decision : {
                        side: {k: [] for k in stat_labels}
                        for side in ['left', 'right']
                    } for decision in ['left', 'right']
                }

            stat_left = d['nb events left'] / d['nb events']
            stat_right = d['nb events right'] / d['nb events']
            if stat_right > 0.9: 
                decision = 'left'
            elif stat_left > 0.9:
                decision = 'right'

            for k in stat_labels:
                data_to_plot[0][decision]['right'][k].append( statistics[ref['right sample']][k] )
                data_to_plot[0][decision]['left'][k].append( statistics[ref['left sample']][k] )

print(data_to_plot[0])

# get xlim and ylim
maximum = {k:0 for k in stat_labels}
for decision in ['correct','incorrect']:
    for k in stat_labels:
        if max(data_to_plot[sh][decision]['first'][k]) > maximum[k]:
            maximum[k] = max(data_to_plot[sh][decision]['first'][k]) 
shift_labels = sorted(list(data_to_plot.keys()))
shift_labels.remove(0)

# FIGURE SCATTER
fig, axes = plt.subplots(len(shift_labels), len(stat_labels))
# plot
for j,stat in enumerate(stat_labels):
    for i,sh in enumerate(shift_labels):
        axes[i,j].scatter(data_to_plot[sh]['correct']['first'][stat], data_to_plot[sh]['correct']['second'][stat], c='g')
        axes[i,j].scatter(data_to_plot[sh]['incorrect']['first'][stat], data_to_plot[sh]['incorrect']['second'][stat], c='r')
        axes[i,j].set_xlim(0,maximum[stat])
        axes[i,j].set_ylim(0,maximum[stat])

        if j == 0:
            axes[i,j].set_ylabel(str(sh)+' shift')
        if i == len(shift_labels) - 1:
            axes[i,j].set_xlabel(stat)

# legend
axes[i,j].scatter([],[],c='g', label='First detected')
axes[i,j].scatter([],[],c='r', label='Second detected')
fig.legend(loc='lower center', ncol=2)
fig.suptitle('Detection according to statistics and shift')
fig.text(0,0.5,'Second sample',va='center', rotation='vertical')
fig.text(0.5,0,'First sample',ha='center')
fig.tight_layout()


# FIGURE MEAN
fig, axes = plt.subplots(len(shift_labels), len(stat_labels))
# plot
for j,stat in enumerate(stat_labels):
    for i,sh in enumerate(shift_labels):
        axes[i,j].scatter([np.mean(data_to_plot[sh]['correct']['first'][stat])], [np.mean(data_to_plot[sh]['correct']['second'][stat])], c='g')
        getConfidenceEllipse(data_to_plot[sh]['correct']['first'][stat], data_to_plot[sh]['correct']['second'][stat], axes[i,j], facecolor='g', alpha=0.2)
        axes[i,j].scatter([np.mean(data_to_plot[sh]['incorrect']['first'][stat])], [np.mean(data_to_plot[sh]['incorrect']['second'][stat])], c='r')
        getConfidenceEllipse(data_to_plot[sh]['incorrect']['first'][stat], data_to_plot[sh]['incorrect']['second'][stat], axes[i,j], facecolor='r',alpha=0.2)
        axes[i,j].set_xlim(0,maximum[stat])
        axes[i,j].set_ylim(0,maximum[stat])

        if j == 0:
            axes[i,j].set_ylabel(str(sh)+' shift')
        if i == len(shift_labels) - 1:
            axes[i,j].set_xlabel(stat)

# legend
axes[i,j].scatter([],[],c='g', label='First detected')
axes[i,j].scatter([],[],c='r', label='Second detected')
fig.legend(loc='lower center', ncol=2)
fig.suptitle('Detection according to statistics and shift')
fig.text(0,0.5,'Second sample',va='center', rotation='vertical')
fig.text(0.5,0,'First sample',ha='center')
fig.tight_layout()


# FIGURE NO SHIFT
fig, axes = plt.subplots(1, len(stat_labels))
# plot
for j,stat in enumerate(stat_labels):
    axes[j].scatter([np.mean(data_to_plot[0]['right']['left'][stat])], [np.mean(data_to_plot[0]['right']['right'][stat])], c='b')
    getConfidenceEllipse(data_to_plot[0]['right']['left'][stat], data_to_plot[0]['right']['right'][stat], axes[j], facecolor='b', alpha=0.2)
    axes[j].scatter([np.mean(data_to_plot[0]['left']['left'][stat])], [np.mean(data_to_plot[0]['left']['right'][stat])], c='orange')
    getConfidenceEllipse(data_to_plot[0]['left']['left'][stat], data_to_plot[0]['left']['right'][stat], axes[j], facecolor='orange', alpha=0.2)
    axes[j].set_xlabel(stat)
    axes[j].set_xlim(0,maximum[stat])
    axes[j].set_ylim(0,maximum[stat])

# legend
axes[j].scatter([],[],c='b', label='Right detected')
axes[j].scatter([],[],c='orange', label='Left detected')
fig.legend(loc='lower center', ncol=2)
fig.suptitle('Detection according to statistics solely')
fig.text(0,0.5,'Right sample',va='center', rotation='vertical')
fig.text(0.5,0,'Left sample',ha='center')
fig.tight_layout()
plt.show()