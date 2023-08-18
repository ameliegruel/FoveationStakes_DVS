import argparse
from filesReader import read_file
from tools import getConfidenceEllipse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Compute statistics for detected ROI")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str, default='none')
args = parser.parse_args()

datasets = [
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-190118/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-131502/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-144524/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-161338/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-173524/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-203255/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_twoOutput_eventcount_20230123-232314/'
]

if args.dataset != 'none':
    if not args.dataset.endswith('/'):
        args.dataset += '/'
    datasets = [args.dataset]

data_to_plot = {}
stat_labels = ['nb events', 'temporal density','spatial density']
detected_labels = ['right','left','both']
side_labels = ['right', 'left']
for ds in datasets:
    data, references, statistics = read_file(ds)

    last_sample_name = None
    nb_last_sample = 0

    for d in data:
        ref = references[d['sample name']]
        if 'shift' in ref.keys():
            sh = int(ref['shift'])
        else:
            sh = 0

        if sh not in data_to_plot.keys():
            data_to_plot[sh] = {
                detected: {
                    side: {
                        k: [] for k in stat_labels
                    } for side in side_labels
                } for detected in detected_labels
            }
        
        if last_sample_name == None or nb_last_sample == 0:
            last_sample_name = d['sample name']
            ref = references[last_sample_name]
            dico = {
                side: {k: [] for k in stat_labels}
                for side in side_labels
            }
            sides = set(side_labels)
        nb_last_sample += 1
        

        stat_left = d['nb events left'] / d['nb events']
        stat_right = d['nb events right'] / d['nb events']
        # if the left sample is shifted, the right sample should be detected first and events should majorly be present on the right
        if stat_right > 0.9:
            detected_side = 'right'
        # if the right sample is shifted, the left sample should be detected first and events should majorly be present on the left
        if stat_left > 0.9:
            detected_side = 'left'
        sides.discard(detected_side)


        if nb_last_sample == 2:
            nb_last_sample = 0

            if len(sides) == 1:
                detected = detected_side
            elif len(sides) == 0:
                detected = 'both'

            # for k in stat_labels:
            #     print(ref)
            #     dico[d['output']][k].append( statistics[ref[detected_side+' sample']][k] )
            for side in side_labels:
                for k in stat_labels:
                    data_to_plot[sh][detected][side][k].append( statistics[ref[side+' sample']][k] )


# get xlim and ylim
maximum = {k:0 for k in stat_labels}
for detected in detected_labels:
    for o in side_labels:
        for k in stat_labels:
            if max(data_to_plot[sh][detected][o][k]) > maximum[k]:
                maximum[k] = max(data_to_plot[sh][detected][o][k]) 
shift_labels = sorted(list(data_to_plot.keys()))

# FIGURE SCATTER
fig, axes = plt.subplots(len(shift_labels), len(stat_labels))
# plot
for j,stat in enumerate(stat_labels):
    for i,sh in enumerate(shift_labels):
        # data to plot[shift][number of obj detected][output 1 or 2][statistic]
        axes[i,j].scatter(data_to_plot[sh]['both']['right'][stat], data_to_plot[sh]['both']['left'][stat], c='g')
        axes[i,j].scatter(data_to_plot[sh]['right']['right'][stat], data_to_plot[sh]['right']['left'][stat], c='orange')
        axes[i,j].scatter(data_to_plot[sh]['left']['right'][stat], data_to_plot[sh]['left']['left'][stat], c='purple')
        axes[i,j].set_xlim(0,maximum[stat])
        axes[i,j].set_ylim(0,maximum[stat])

        if j == 0:
            axes[i,j].set_ylabel(str(sh)+' shift')
        if i == len(shift_labels) - 1:
            axes[i,j].set_xlabel(stat)


# legend
axes[i,j].scatter([],[],c='g', label='Two objects of interest detected')
axes[i,j].scatter([],[],c='orange', label='Right object of interest detected')
axes[i,j].scatter([],[],c='purple', label='Left object of interest detected')
fig.legend(loc='lower center', ncol=3)
fig.suptitle('Detection according to statistics and shift')
fig.text(0.5,0,'Output 1',ha='center')
fig.text(0,0.5,'Output 2',va='center', rotation='vertical')
fig.tight_layout()


# FIGURE MEAN
fig, axes = plt.subplots(len(shift_labels), len(stat_labels))
# plot
for j,stat in enumerate(stat_labels):
    for i,sh in enumerate(shift_labels):
        # data to plot[shift][number of obj detected][output 1 or 2][statistic]
        axes[i,j].scatter([np.mean(data_to_plot[sh]['both']['right'][stat])], [np.mean(data_to_plot[sh]['both']['left'][stat])], c='g')
        # getConfidenceEllipse(data_to_plot[sh]['both']['right'][stat], data_to_plot[sh]["both"]['left'][stat], axes[i,j], facecolor='g', alpha=0.2)
        axes[i,j].scatter([np.mean(data_to_plot[sh]['right']['right'][stat])], [np.mean(data_to_plot[sh]['right']['left'][stat])], c='orange')
        # getConfidenceEllipse(data_to_plot[sh]['right']['right'][stat], data_to_plot[sh]['right']['left'][stat], axes[i,j], facecolor='orange',alpha=0.2)
        axes[i,j].scatter([np.mean(data_to_plot[sh]['left']['right'][stat])], [np.mean(data_to_plot[sh]['left']['left'][stat])], c='purple')
        # getConfidenceEllipse(data_to_plot[sh]['left']['right'][stat], data_to_plot[sh]['left']['left'][stat], axes[i,j], facecolor='purple',alpha=0.2)
        axes[i,j].set_xlim(0,maximum[stat])
        axes[i,j].set_ylim(0,maximum[stat])

        if j == 0:
            axes[i,j].set_ylabel(str(sh)+' shift')
        if i == len(shift_labels) - 1:
            axes[i,j].set_xlabel(stat)


# legend
axes[i,j].scatter([],[],c='g', label='Two objects of interest detected')
axes[i,j].scatter([],[],c='orange', label='Right object of interest detected')
axes[i,j].scatter([],[],c='purple', label='Left object of interest detected')
fig.legend(loc='lower center', ncol=3)
fig.suptitle('Detection according to statistics and shift')
fig.text(0,0.5,'Output 2',va='center', rotation='vertical')
fig.text(0.5,0,'Output 1',ha='center')
fig.tight_layout()

plt.show()