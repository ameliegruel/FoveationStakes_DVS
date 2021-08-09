import numpy as np 
np.seterr('raise')
import argparse
import os

### PARSER

parser = argparse.ArgumentParser(description="Transform the events with [x,y,p,t] formalism to Ev-SegNet formalism")
parser.add_argument("events", metavar="E", type=str, nargs="+", help="Input events")
parser.add_argument("--time_window", "-tw", help="Define the time window (in ms) used to integrate the event information", nargs=1, metavar="TW", type=int, default=[50])
parser.add_argument("--output", "-o", help="Name of output directory, where events will be saved", nargs=1, metavar="O", type=str, default=None)
parser.add_argument("--image_size", "-s", help="Provide size of the sensor, as '-s x y'", nargs=2, metavar="S", type=int, default=[346, 200])
args = parser.parse_args()

if args.output == None : 
    args.output = ["formalism_"+args.events[0][:-4]]
os.mkdir(args.output[0])

### FUNCTIONS 

def get_newFormalism_by_timewindow(image_size=args.image_size): 
    # initializes with default values, for no event recorded 
    return [
        [
            [
                -1,  # 2D histogram for negative events
                -1,  # 2D histogram for positive events
                0,   # mean for negative events
                0,   # mean for positive events
                0,   # standard deviation for negative events
                0    # standard deviation for positive events
            ] for y in range(image_size[1]) # columns
        ] for x in range(image_size[0])     # lines
    ]

def get_events_timestamps(events, x, y, p):
    return [e[2] for e in events if e[0]==x and e[1]==y and e[3]==p]

def get_2DHistogram(timestamps):
    return len(timestamps)

def get_mean(timestamps, histogram):
    return 1/histogram * np.sum(timestamps)

def get_std(timestamps, histogram, mean):
    if histogram == 1:
        return 0
    return np.sqrt( np.sum( np.square(np.array(timestamps) - mean) ) / (histogram-1) )



### MAIN

# get events
events = np.load(args.events[0]) # shape(nb_events, nb_channels) with nb_channels = 4 (0-1: x-y coordinates, 2: timestamp in ms, 3: polarity (+1 or -1))
print("Events correctly loaded from "+args.events[0]+"\n")

timewindow_limit = args.time_window[0]
events_by_timewindow = [[]]
print("First time window initialized")

# new formalism for all event dataset 
newFormalism = [get_newFormalism_by_timewindow()]


for ev in events: 
    if ev[2] < timewindow_limit:
        # save events in current timewindow
        events_by_timewindow[-1].append(ev)
        
    else :
        # compute new formalism for the ending time window
        for x in range(args.image_size[0]):      # lines
            for y in range(args.image_size[1]):  # columns
                
                ### positive events 
                neg_ts = get_events_timestamps(events_by_timewindow[-1], x, y, -1)  # get timestamps
                if len(neg_ts) > 0:
                    # compute 2D histogram of events in timewindow
                    neg_2DH = get_2DHistogram(neg_ts)
                    # print("Negative 2D histogram : ",neg_2DH)
                    newFormalism[-1][x][y][0] = neg_2DH
                    # compute mean of normalized timestamps of events in timewindow
                    neg_mean = get_mean(neg_ts, neg_2DH)
                    newFormalism[-1][x][y][2] = neg_mean
                    # compute standard deviation of normalized timestamps of events in timewindow
                    neg_std = get_std(neg_ts, neg_2DH, neg_mean)
                    newFormalism[-1][x][y][4] = neg_std
                
                ### negative events
                pos_ts = get_events_timestamps(events_by_timewindow[-1], x, y, 1)  # get timestamps
                if len(pos_ts) > 0:
                    # compute 2D histogram of events in timewindow
                    pos_2DH = get_2DHistogram(pos_ts)
                    # print("Positive 2D histogram : ",pos_2DH)
                    newFormalism[-1][x][y][1] = pos_2DH
                    # compute mean of normalized timestamps of events in timewindow
                    pos_mean = get_mean(pos_ts, pos_2DH)
                    newFormalism[-1][x][y][3] = pos_mean
                    # compute standard deviation of normalized timestamps of events in timewindow
                    pos_std = get_std(pos_ts, pos_2DH, pos_mean)
                    newFormalism[-1][x][y][5] = pos_std

        # save events
        print("Treatment done")
        np.save(args.output[0]+"/events_for_timewindow"+str(len(events_by_timewindow))+".npy",np.array(newFormalism[-1]))
        print("Events from this timewindow saved under new formalism as "+args.output[0]+"/events_for_timewindow"+str(len(events_by_timewindow))+".npy")
        print("Onto the next time window\n")
        
        # set next timewindow
        events_by_timewindow.append([])
        newFormalism.append(get_newFormalism_by_timewindow())
        timewindow_limit = timewindow_limit+args.time_window[0]
