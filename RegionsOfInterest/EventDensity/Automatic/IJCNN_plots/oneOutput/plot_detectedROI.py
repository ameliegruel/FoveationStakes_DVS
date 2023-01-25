import argparse
from filesReader import read_file
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compute statistics for detected ROI")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str, default='none')
parser.add_argument("--plot-figure", "-pf", help="Plot figure", action='store_true')
args = parser.parse_args()

datasets = [
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-191011/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-170026/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-112147/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-155610/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-094442/',
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/saliencyFilter_oneOutput_eventcount_20230120-181249/'
]

if args.dataset != 'none':
    if not args.dataset.endswith('/'):
        args.dataset += '/'
    datasets = [args.dataset]

data_to_plot = {}
for ds in datasets:
    data, references, statistics = read_file(ds)

    correct_right = 0 
    total_right = 0
    correct_left = 0
    total_left = 0

    for d in data:
        ref = references[d['sample name']]
        stat_left = d['nb events left'] / d['nb events']
        stat_right = d['nb events right'] / d['nb events']
        if ref['shifted'] == 'left':
            total_left += 1
            # if the left sample is shifted, the right sample should be detected first and events should majorly be present on the right
            if stat_right > 0.9:
                correct_left += 1

        elif ref['shifted'] == 'right':
            total_right +=1
            # if the right sample is shifted, the left sample should be detected first and events should majorly be present on the left
            if stat_left > 0.9:
                correct_right += 1


    print("Left:",correct_left, "correctly detected on",total_left,"total\n-> Accuracy:",correct_left/total_left)
    print("Right:",correct_right, "correctly detected on",total_right,"total\n-> Accuracy:",correct_right/total_right)
    print("Total accuracy:",(correct_left+correct_right)/(total_left+total_right))
    print()

    data_to_plot[int(ref['shift'])] = {
        'left':     correct_left/total_left,
        'right':    correct_right/total_right,
        'accuracy': (correct_left+correct_right)/(total_left+total_right)
    }

if args.plot_figure:
    fig = plt.subplots()
    label_X = sorted(list(data_to_plot.keys()))
    plt.bar([x-0.25 for x in range(len(data_to_plot))], [data_to_plot[shift]['left'] for shift in label_X], color = 'b', width = 0.25, label='Left')
    plt.bar([x for x in range(len(data_to_plot))], [data_to_plot[shift]['right'] for shift in label_X], color = 'g', width = 0.25, label='Right')
    plt.bar([x+0.25 for x in range(len(data_to_plot))], [data_to_plot[shift]['accuracy'] for shift in label_X], color = 'r', width = 0.25, label='Accuracy')
    plt.title('Accuracy of detection by shift')
    plt.xticks(list(range(len(data_to_plot))), label_X)
    plt.xlabel('Shift (in microseconds)')
    plt.ylabel('Accuracy (in %)')
    plt.legend()
    plt.show()