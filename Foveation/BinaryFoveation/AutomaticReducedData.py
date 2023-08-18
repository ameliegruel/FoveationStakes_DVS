from os import walk, path, makedirs
import numpy as np
import h5py as h
import argparse
from getFoveatedData import LR2HR, loadData, getFormat
from datetime import datetime as d

parser = argparse.ArgumentParser(description="Automatically reduce data")
parser.add_argument("--dataset", "-da", nargs="+", help="Dataset repertory", metavar="D", type=str)
parser.add_argument("--divider","-div", metavar="d", type=int, help="Dividing factor", default=4)
parser.add_argument("--method", "-m", help="Reduction method (between 'funelling','eventcount', 'cubic', 'linear')", metavar="m", type=str)
args = parser.parse_args()

# Datasets :
# "/home/amelie/Scripts/Data/DVS128Gesture/DVS128G_classifier_data/",
# "/home/amelie/Scripts/Data/DDD17_datasets/Ev-SegNet_xypt/"

assert args.method in ['funelling','eventcount', 'cubic', 'linear']

if args.method == 'funelling':
    from reduceEvents import SpatialFunnelling as EventReduction
elif args.method == 'eventcount':
    from reduceEvents import EventCount as EventReduction
elif args.method == 'cubic' or args.method == 'linear':
    from reduceEvents import LogLuminance as EventReduction


def adaptToEventReductionFormat(events, format_og):
    X = events[:,format_og.index('x')]
    Y = events[:,format_og.index('y')]
    P = events[:,format_og.index('p')]
    T = events[:,format_og.index('t')]

    events = np.column_stack((X,Y,P,T))
    return events

def saveEvents(events, event_file, save_to): #, format_HR):
    if not path.exists(save_to):
        makedirs(save_to)

    if event_file.endswith('.npz') or event_file.endswith('.npy'):
        np.save(
            path.join( save_to , event_file.replace("npz","npy")),
            events
        )
    
    elif event_file.endswith('hdf5'):
        f = h.File(path.join( save_to, event_file), "w")
        f.create_dataset("event", data=events)
        f.close()

not_done = []

for dr in args.dataset:
    print("\n\n>>",dr)

    if 'DVS128Gesture' in dr:
        original_repertory = dr+"DVSGesture/ibmGestureTest/"
        reduced_repertory = dr+"reduced_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(128,128)
    elif 'DDD17' in dr:
        original_repertory = dr+"test/"
        reduced_repertory = dr+"reduced_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(346,260)


    init=False
    
    i = 0
    for (rep_path, _, files) in walk(original_repertory):
        repertory=rep_path.replace(original_repertory, "")
        
        if len(files) > 0: 
            
            for event_file in files :
                print(repertory, event_file, end=" ")

                if not path.exists(path.join( reduced_repertory, repertory, event_file.replace("npz","npy")) ): 

                    start = d.now()
                    
                    HR = loadData(path.join( rep_path, event_file))
                    format_HR = getFormat(HR)
                    HR = adaptToEventReductionFormat(HR, format_HR)

                    if args.method == 'linear':
                        reduction = EventReduction(
                            input_ev=HR,
                            coord_t=3,
                            div=args.divider,
                            width=size[0],
                            height=size[1],
                            cubic_interpolation=False
                        )                    
                    else : 
                        reduction = EventReduction(
                            input_ev=HR,
                            coord_t=3,
                            div=args.divider,
                            width=size[0],
                            height=size[1]
                        )
                    
                    reduction.reduce()
                    reduced_events = reduction.events
                    
                    reduced_events = LR2HR(reduced_events, args.divider, format_HR)

                    print(d.now() - start )
                    
                    saveEvents(
                        reduced_events,
                        event_file,
                        path.join( reduced_repertory, repertory)
                    )
                else : 
                    print()
                    not_done.append(event_file)

print("All files not done yet:",not_done)