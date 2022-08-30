from os import walk, path, makedirs
import numpy as np
import h5py as h
from getFoveatedData import getPuzzle, loadData
from datetime import datetime as d
import argparse

parser = argparse.ArgumentParser(description="Automatically reduce data")
parser.add_argument("--dataset", "-da", nargs="+", help="Dataset repertory", metavar="D", type=str)
parser.add_argument("--divider","-div", metavar="d", type=int, help="Dividing factor", default=4)
parser.add_argument("--method", "-m", help="Reduction method (between 'funelling','eventcount', 'cubic', 'linear')", metavar="m", type=str)
parser.add_argument("--ROI", "-roi", help="Path to ROI repertory", metavar="m", type=str)
args = parser.parse_args()

if "funelling" in args.ROI:
    end = 'funelling'
elif 'eventcount' in args.ROI:
    end = 'eventcount'
elif 'linear' in args.ROI:
    end = 'linear'
elif 'cubic' in args.ROI:
    end = 'cubic'

# Datasets :
# "/home/amelie/Scripts/Data/DVS128Gesture/DVS128G_classifier_data/",
# "/home/amelie/Scripts/Data/DDD17_datasets/Ev-SegNet_xypt/"

assert args.method in ['funelling','eventcount', 'cubic', 'linear']

def saveEvents(events, event_file, save_to):
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
        ROI_repertory = args.ROI
        fovea_repertory = dr+"foveated_data_"+args.method+"_div"+str(args.divider)+"_ROI"+end+"/"
        reduced_repertory = dr+"reduced_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(128,128)
    elif 'DDD17' in dr:
        original_repertory = dr+"test/"
        ROI_repertory = args.ROI
        fovea_repertory = dr+"foveated_data_"+args.method+"_div"+str(args.divider)+"_ROI"+end+"/"
        reduced_repertory = dr+"reduced_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(346,260)

    assert path.exists(reduced_repertory)

    init=False
    
    i = 0
    for (rep_path, _, files) in walk(original_repertory):
        repertory=rep_path.replace(original_repertory, "")
        
        if len(files) > 0: 
            
            for event_file in files :
                print(repertory, event_file, end=" ")

                if not path.exists(path.join( fovea_repertory, repertory, event_file.replace("npz","npy")) ) and path.exists(path.join( ROI_repertory, repertory, event_file.replace("npz","npy")) ):
                    
                    start = d.now()
                    HR  = loadData(path.join( rep_path, event_file ))
                    LR  = loadData(path.join( reduced_repertory, repertory, event_file ))
                    ROI = loadData(path.join( ROI_repertory, repertory, event_file.replace("npz","npy") ))
                    
                    
                    if len(LR) == 0 and len(ROI) == 0:
                        foveated_events = np.zeros((0,4))

                    elif len(ROI) == 0:
                        foveated_events = LR
                    
                    else :
                    # print(event_file,"yessss",ROI.shape, LR.shape)
                        foveated_events = getPuzzle(
                            insert_image=HR,
                            ROI=ROI,
                            frame_image=LR,
                            div=4,
                            ROI_div=4*5,
                            size=size,
                            ROI_latency=1e6,
                            method=args.method
                        )

                    print(d.now() - start )
                        
                    saveEvents(
                        foveated_events,
                        event_file,
                        path.join( fovea_repertory, repertory)
                    )
                else : 
                    print()
                    not_done.append(event_file)

print("All files not done yet:",not_done)