from matplotlib import use
import esim_py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
from getTimestamps import getTS
from skimage import io, img_as_ubyte
from skimage.transform import rescale
import shutil

# set up parser
parser = argparse.ArgumentParser(description="Get timestamps for an image dataset given as input")
parser.add_argument("dataset", metavar="D", type=str, nargs="+", help="Input dataset")
parser.add_argument("--contrast_threshold", "-ct", help="Define the contrast threshold used to compute the events", nargs=1, metavar="CT", type=int, default=[0.25])
parser.add_argument("--frame_interval", "-fi", help="Define the time interval between frames", nargs=1, metavar="F", type=float, default=[50])
parser.add_argument("--high_resolution", "-HR", help="Get the events with a high resolution (True by default)", action="store_true", default=True)
parser.add_argument("--low_resolution", "-LR", help="Get the events with a low resolution (False by default)", action='store_true', default=False)
parser.add_argument("--reduction_coeff", "-rc", help="Define the reduction coefficient for the reduction into low resolution", nargs=1, default=[0.25], type=float, metavar="RC")
parser.add_argument("--output", "-o", help="Name of output file, where events will be saved", nargs=1, metavar="O", type=str, default=None)
parser.add_argument("--figure", "-f", help="Visualize the events", action='store_true', default=False)
args = parser.parse_args()


# function provided by Gehrig et al (CVPR 2019) to visualise the events
def viz_events(events, resolution):
    pos_events = events[events[:,-1]==1]
    neg_events = events[events[:,-1]==-1]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), pos_events[:,-1]**2)
    np.add.at(image_neg, (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), neg_events[:,-1]**2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution), 
            image_neg.reshape(resolution), 
            np.zeros(resolution, dtype="uint8") 
        ], -1
    ) * 50

    return image_rgb    


### HIGH AND LOW RESOLUTION DATA

# get low resolution
if args.dataset[0][-1] == "/":
    dataset_name = args.dataset[0].split("/")[-2]+"/"
    dataset_directory = args.dataset[0] 
else : 
    dataset_name = args.dataset[0].split("/")[-1]+"/"
    dataset_directory = args.dataset[0]+"/"
if args.low_resolution :
    new_dataset_name = dataset_name[:-1]+"_lowResolution/" 
    new_dataset_directory = dataset_directory[:-1]+"_lowResolution/"
    if not os.path.exists(new_dataset_directory):
        os.makedirs(new_dataset_directory)
    list_images = os.listdir(dataset_directory)
    for image in list_images:
        io.imsave(new_dataset_directory+image, img_as_ubyte(rescale(io.imread(dataset_directory+image), args.reduction_coeff[0], anti_aliasing=False)))
    dataset_directory = new_dataset_directory
    
# get data
image_folder = os.path.join(os.path.dirname(__file__), dataset_directory)
timestamps_file = "timestamps_"+dataset_name[:-1]+"_tmp.txt"
getTS(dataset_directory, timestamps_file, frame_interval=args.frame_interval[0])
timestamps_file = os.path.join(os.path.dirname(__file__), timestamps_file)


### PRODUCE EVENTS

# define parameters 
contrast_threshold_positive = contrast_threshold_negative = args.contrast_threshold[0]
refractory_period = 1e-4
log_eps = 1e-3
use_log = True
H, W = io.imread(dataset_directory+os.listdir(dataset_directory)[0]).shape  # heigth and wideness of images (high resolution)

event_simulator = esim_py.EventSimulator(contrast_threshold_positive, contrast_threshold_negative, refractory_period, log_eps, use_log)
events = event_simulator.generateFromFolder(image_folder, timestamps_file)

# save events
reso = "_HR.npy"
if args.low_resolution:
    reso = "_LR.npy"
if args.output == None:
    output_file = "events_"+dataset_name[:-1]+reso
else :
    output_file = args.output[0]
np.save(output_file, events)
print("Events produced from "+dataset_name+" saved as "+output_file)


### DISPLAY EVENTS 

if args.figure:
    # get optimal parameters
    contrast_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    number_events_per_plot = [5000, 10000, 33000, 66000, 100000]
    
    fig, ax = plt.subplots(ncols=len(contrast_thresholds), nrows=len(number_events_per_plot), figsize=(10,10), sharex=True, sharey=True)
    fig.supxlabel("Number of events in the plot")
    fig.supylabel("Positive and negative contrast thresholds")

    for x, ct in enumerate(contrast_thresholds):
        event_simulator.setParameters(ct, ct, refractory_period, log_eps, use_log)
        events = event_simulator.generateFromFolder(image_folder, timestamps_file)
        print("* For contrast threshold of "+str(ct)+" : Number of events : "+str(len(events))+" with shape", events.shape)
        
        for y, nb_events in enumerate(number_events_per_plot):
            ax[x,y].imshow(viz_events(events[:nb_events], [H, W]))
            if x == len(contrast_thresholds)-1:
                ax[x,y].set_xlabel(str(nb_events))
            if y == 0:
                ax[x,y].set_ylabel(str(ct))

    plt.savefig("contrast_train.png")
    

# clean up
os.remove(timestamps_file)
if args.low_resolution :
    shutil.rmtree(new_dataset_directory)