import numpy as np 
import argparse

# set up parser
parser = argparse.ArgumentParser(description="Transform the events with [x,y,p,t] formalism to Ev-SegNet formalism")
parser.add_argument("events", metavar="E", type=str, nargs="+", help="Input events")
parser.add_argument("--time_window", "-tw", help="Define the time window (in ms) used to integrate the event information", nargs=1, metavar="TW", type=int, default=[50])
parser.add_argument("--output", "-o", help="Name of output file, where events will be saved", nargs=1, metavar="O", type=str, default=None)
parser.add_argument("--image_size", "-s", help="Provide size of the sensor, as '-s x y'", nargs=2, metavar="S", type=int, default=[346, 200])
args = parser.parse_args()

# get events
events = np.load(args.events[0]) # shape(nb_events, nb_channels) with nb_channels = 4 (0-1: x-y coordinates, 2: timestamp in ms, 3: polarity (+1 or -1))

neg_events_by_timewindow = [[]]
pos_events_by_timewindow = [[]]

newFormalism_by_timewindow = [
    [
        [
            # default values
            -1,  # 2D histogram for negative events
            -1,  # 2D histogram for positive events
            0,   # mean for negative events
            0,   # mean for positive events
            0,   # standard deviation for negative events
            0    # standard deviation for positive events
        ] for y in range(args.image_size[1])
    ] for x in range(args.image_size[0])
]

newFormalism = 
{
    "2Dhisto_neg": [],
    "2Dhisto_pos": [],
    "mean_neg": [],
    "mean_pos": [],
    "std_neg": [],
    "std_pos": []
}

timewindow_limit = args.time_window[0]

for ev in events: 
    if ev[2] < timewindow_limit:
        # save events in current timewindow
        if ev[3] > 0:
            # save positive events
            pos_events_by_timewindow[-1].append(ev)
        elif ev[3] < 0:
            # save negative events 
            neg_events_by_timewindow[-1].append(ev)
    else :
        # compute 2D histogram
        
        
        # set next timewindow
        pos_events_by_timewindow.append([])
        neg_events_by_timewindow.append([])
        timewindow_limit = timewindow_limit+args.time_window[0]

