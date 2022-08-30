from os import walk, path, makedirs
import numpy as np
import h5py as h
from reduceEvents import StochasticStructural as EventReduction
from getFoveatedData import loadData, getFormat
from datetime import datetime as d
import argparse


parser = argparse.ArgumentParser(description="Automatically reduce data")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str)
args = parser.parse_args()

# Datasets :
# "/home/amelie/Scripts/Data/DVS128Gesture/DVS128G_classifier_data/",
# "/home/amelie/Scripts/Data/DDD17_datasets/Ev-SegNet_xypt/"

def adaptToFormat(events, format_og, new_format):
    X = events[:,format_og.index('x')]
    Y = events[:,format_og.index('y')]
    P = events[:,format_og.index('p')]
    T = events[:,format_og.index('t')]

    correct_events = np.zeros((len(X), 4))
    correct_events[:,new_format.index('x')] = X
    correct_events[:,new_format.index('y')] = Y
    correct_events[:,new_format.index('p')] = P
    correct_events[:,new_format.index('t')] = T

    return correct_events

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
Div = [80] #[5,10,20,40,60,80]

if 'DVS128Gesture' in args.dataset:
    size=(128,128)
elif 'DDD17' in args.dataset:
    size=(346,260)

for div in Div:

    print('> div:',div,"%")

    global_start = d.now()
    new_repertory = args.dataset[:-1] + "_" + str(div)+"%/"
    
    for (rep_path, _, files) in walk(args.dataset):
        repertory=rep_path.replace(args.dataset, "")
        
        if len(files) > 0:
            
            for event_file in files :
                print(args.dataset,div,"% -",repertory, event_file, end=" ")
                data = loadData(path.join( rep_path, event_file))
                # print(len(data))
                # print(data[:10])

                # try :
                
                if not path.exists(path.join( new_repertory, repertory, event_file.replace("npz","npy")) ): 

                    start = d.now()
                    
                    data = loadData(path.join( rep_path, event_file))

                    if len(data) != 0:
                        format_data = getFormat(data)
                        data = adaptToFormat(data, format_data,'xypt')

                        reduction = EventReduction(
                            input_ev=data,
                            coord_t=3,
                            div=div,
                            width=size[0],
                            height=size[1]
                        )
                        reduction.reduce()
                        reduced_events = adaptToFormat(reduction.events, 'xypt', format_data)
                        
                        
                    else : 
                        reduced_events = np.zeros((0,4))

                    print(d.now() - start )
                    saveEvents(
                        reduced_events,
                        event_file,
                        path.join( new_repertory, repertory)
                    )

                else : 
                    print()
                
                # except ValueError:
                #     print()
                #     pass
        
    print("Runtime for",div,'% of', args.dataset,":", d.now() - global_start,"\n")