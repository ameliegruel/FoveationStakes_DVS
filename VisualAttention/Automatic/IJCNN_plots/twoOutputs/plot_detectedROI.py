import argparse
from filesReader import read_file
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compute statistics for detected ROI")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str, default='none')
parser.add_argument("--plot-figure", "-pf", help="Plot figure", action='store_true')
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
for ds in datasets:
    data, references, statistics = read_file(ds)

    nb_zero = 0
    nb_one = 0
    nb_two = 0
    nb_tot = 0

    last_sample_name = None
    nb_last_sample = 0

    for d in data:
        if last_sample_name == None or nb_last_sample == 0:
            last_sample_name = d['sample name']
            ref = references[last_sample_name]
            sides = set(['right', 'left'])
        nb_last_sample += 1
        
        stat_left = d['nb events left'] / d['nb events']
        stat_right = d['nb events right'] / d['nb events']
        
        # if the left sample is shifted, the right sample should be detected first and events should majorly be present on the right
        if stat_right > 0.9:
            sides.discard('right')

        # if the right sample is shifted, the left sample should be detected first and events should majorly be present on the left
        if stat_left > 0.9:
            sides.discard('left')
        
        if nb_last_sample == 2:
            if len(sides) == 2:
                nb_zero += 1
            elif len(sides) == 1:
                nb_one += 1
            elif len(sides) == 0:
                nb_two += 1
            nb_last_sample = 0
            nb_tot += 1

    print("Zero:",nb_zero, "detected on",nb_tot,"total")
    print("One:",nb_one, "correctly detected on",nb_tot,"total")#\n-> Accuracy:",correct_right/total_right)
    print("Two:",nb_two, "correctly detected on",nb_tot,"total")
    print("-> Accuracy:",nb_two / nb_tot)
    print()

    if 'shift' in ref.keys():
        k = int(ref['shift'])
    else:
        k = 0
    data_to_plot[k] = {
        'zero': nb_zero,
        'one' : nb_one,
        'two' : nb_two,
        'tot' : nb_tot
    }

if args.plot_figure:
    fig = plt.subplots()
    label_X = sorted(list(data_to_plot.keys()))
    plt.bar([x-0.2 for x in range(len(data_to_plot))], [data_to_plot[shift]['one'] for shift in label_X], color = 'r', width = 0.4, label='One detected')
    plt.bar([x+0.2 for x in range(len(data_to_plot))], [data_to_plot[shift]['two'] for shift in label_X], color = 'g', width = 0.4, label='Two detected')
    plt.title('Number of objects of interest detected according to shift')
    plt.xticks(list(range(len(data_to_plot))), label_X)
    plt.xlabel('Shift (in microseconds)')
    plt.ylabel('Number of objects of interest detected')
    plt.legend()

    fig = plt.subplots()
    label_X = sorted(list(data_to_plot.keys()))
    plt.plot(list(range(len(data_to_plot))), [data_to_plot[shift]['two']/data_to_plot[shift]['tot'] for shift in label_X])
    plt.title('Detection of 2 objects of interest')
    plt.xticks(list(range(len(data_to_plot))), label_X)
    plt.xlabel('Shift (in microseconds)')
    plt.ylabel('Accuracy (in %)')
    plt.show()