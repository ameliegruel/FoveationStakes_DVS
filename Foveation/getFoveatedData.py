from operator import index
import numpy as np
# from reduceEvents import EventCount as EventReduction
import h5py as h
from datetime import datetime as d
from tqdm import tqdm

def loadData(file_name):
    if file_name.endswith('npy'):
        ev = np.load(file_name)
    elif file_name.endswith('npz'):
        ev = np.load(file_name)
        ev = np.concatenate((
            ev["x"].reshape(-1,1),
            ev["y"].reshape(-1,1),
            ev["p"].reshape(-1,1),
            ev["t"].reshape(-1,1)
        ), axis=1).astype('float64')
    elif file_name.endswith('hdf5'):
        ev = h.File(file_name,'r')
        assert 'event' in ev.keys()
        ev = np.array(ev['event'])
    return ev

def getFormat(ev):
    string_coord = [None]*4
    set_coord = [2,3]

    if max(ev[:,2]) in [0,1]:
        coord_p = 2
        set_coord.remove(2)
    elif max(ev[:,3]) in [0,1]:
        coord_p = 3
        set_coord.remove(3)
    string_coord[coord_p] = 'p'

    if max(ev[:,0]) > 1e6:
        coord_ts = 0
        coord_x = 1
        coord_y = set_coord[0]
    else:
        coord_ts = set_coord[0]
        coord_x = 0
        coord_y = 1

    string_coord[coord_x] = 'x'
    string_coord[coord_y] = 'y'
    string_coord[coord_ts] = 't'

    return ''.join(string_coord)

def getCoordLR(
    coordHR,    # value or numpy array
    div
    ):
    return np.floor(coordHR/div)

def getCoordHR(
    coordLR,    # value or numpy array
    div
    ):
    return np.floor(coordLR*div)

def LR2HR(LRimage, div, format):
    X = getCoordHR(LRimage[:,0], div)
    Y = getCoordHR(LRimage[:,1], div)
    P = LRimage[:,2]
    T = LRimage[:,3]

    X_ = []
    Y_ = []
    P_ = []
    T_ = []

    for x,y,p,t in zip(X,Y,P,T):
        X_ += list(np.random.random_integers(low=x,high=min(max(X),x+div-1),size=div*div))
        Y_ += list(np.random.random_integers(low=y,high=min(max(Y),y+div-1),size=div*div))
        P_ += [p]*div*div
        T_ += [t]*div*div


    HRimage = np.zeros((len(X)*16,len(format)))
    HRimage[:,format.index('x')] = X_
    HRimage[:,format.index('y')] = Y_
    HRimage[:,format.index('p')] = P_
    HRimage[:,format.index('t')] = T_

    return HRimage

def getMinMax(coord, insert_size, fig_size):
    c_min = max(coord - insert_size/2, 0)
    c_max = min(c_min + insert_size, fig_size)
    return c_min, c_max


def transformRoi(
    ROI,
    ROI_div,
    size,
    div=4
    ):
    '''
    Transform ROI data, originally a 4 channels np.array as xypt (default) in low resolution, into a 5 channels np.array as x_min,x_max,y_min,y_max,t in high resolution
    '''
    format_ROI = getFormat(ROI)
    X = ROI[:,format_ROI.index('x')]
    Y = ROI[:,format_ROI.index('y')]
    T = ROI[:,format_ROI.index('t')]

    transformed_ROI_HR = np.zeros((0,5))

    for x,y,t in zip(X,Y,T):

        # min and max coordinates of insert part in HR
        x_min_HR, x_max_HR = getMinMax(getCoordHR(x, ROI_div), ROI_div, size[0])
        y_min_HR, y_max_HR = getMinMax(getCoordHR(y, ROI_div), ROI_div, size[1])

        current_ROI = np.array([[x_min_HR, x_max_HR, y_min_HR, y_max_HR, t]])

        transformed_ROI_HR = np.vstack((
            transformed_ROI_HR,
            current_ROI
        ))

    return transformed_ROI_HR


def getFrameImage(
    X, Y, P, T,
    div
    ):

    original_events = np.column_stack((X,Y,P,T))

    event_reduction = EventReduction(input_ev=original_events, coord_t=3,div=div)
    event_reduction.reduce()
    return event_reduction.events



def getPuzzle(
    insert_image,   # high resolution events as np.array
    ROI,            # place of ROIs as np.array, in low resolution
    frame_image = [],
    div = 4,
    ROI_div = 5*4,
    size = (128,128),
    ROI_latency = 1000, # microsecond
    method = 'eventcount'
    ):
    
    # import corresponding method
    assert method in ['funelling','eventcount', 'cubic', 'linear']

    global EventReduction
    if method == 'funelling':
        from reduceEvents import SpatialFunnelling as EventReduction
    elif method == 'eventcount':
        from reduceEvents import EventCount as EventReduction
    elif method == 'cubic' or method == 'linear':
        from reduceEvents import LogLuminance as EventReduction

    # get format of insert image (HR)
    format_HR = getFormat(insert_image)

    # adapt HR to ROI's length
    try:
        max_t = max(ROI[:,-1])
        min_t = min(insert_image[:,format_HR.index('t')])
        insert_image = insert_image[insert_image[:,format_HR.index('t')] < max_t + min_t]
    except ValueError:
        insert_image = np.zeros((0,4))

    insert_image = insert_image[insert_image[:,format_HR.index('t')].argsort()]
    X = insert_image[:,format_HR.index('x')]
    Y = insert_image[:,format_HR.index('y')]
    P = insert_image[:,format_HR.index('p')]
    T = insert_image[:,format_HR.index('t')]

    # get frame image (LR)
    if len(frame_image) == 0:
        frame_image = getFrameImage(X,Y,P,T,div)
        # translate frame image (LR) into HR coordinates
        frame_image = LR2HR(frame_image, div, format_HR)

    frame_image = frame_image[frame_image[:,format_HR.index('t')].argsort()]
    T_frame = frame_image[:,format_HR.index('t')]

    # get size of frame image (HR)
    x_size_frame, y_size_frame = size

    # get transformed ROIs min and max x and y coordinates, respectively in LR and HR
    transformed_ROI_HR = transformRoi(ROI, ROI_div, size, div)
    t_ROI = transformed_ROI_HR[:,-1]

    # launch puzzle
    last_t_delay = ROI_latency + 1
    final_image = []

    idx_insert=0
    idx_frame=0
    last_t = 0
    last_ROI = []

    idx_roi = 0

    for t in tqdm(np.unique(T_frame)):

        # get insert at t
        while idx_insert < len(T) and T[idx_insert] < last_t:
            idx_insert += 1
        prev_idx_insert = idx_insert
        while idx_insert < len(T) and T[idx_insert] <= t:
            idx_insert += 1
        insert = insert_image[prev_idx_insert:idx_insert]

        # get frame at t
        prev_idx_frame = idx_frame
        while idx_frame < len(T_frame) and T_frame[idx_frame] == t:
            idx_frame += 1
        frame = frame_image[prev_idx_frame:idx_frame]

        # get any t in t_roi which happened between now and last loop
        while idx_roi <len(t_ROI) and t_ROI[idx_roi] < last_t:
            idx_roi += 1
        prev_idx_roi = idx_roi
        while idx_roi <len(t_ROI) and t_ROI[idx_roi] <= t:
            idx_roi += 1
        current_ROI = transformed_ROI_HR[prev_idx_roi:idx_roi]

        if len(current_ROI) > 0:
            last_t_delay = 0
            last_ROI = current_ROI

        if last_t_delay <= ROI_latency:
            global_insert = np.zeros((0,4))
            
            for _x_min_HR_,_x_max_HR_, _y_min_HR_, _y_max_HR_, _ in last_ROI :

                insert_ = insert[
                    (insert[:,format_HR.index('x')] > _x_min_HR_) &
                    (insert[:,format_HR.index('x')] < _x_max_HR_) &
                    (insert[:,format_HR.index('y')] > _y_min_HR_) &
                    (insert[:,format_HR.index('y')] < _y_max_HR_)
                ]
                global_insert = np.vstack((global_insert, insert_))
            
                # update mask
                frame = frame[~(
                    (frame[:,format_HR.index('x')] >= _x_min_HR_) &
                    (frame[:,format_HR.index('x')] <= _x_max_HR_) &
                    (frame[:,format_HR.index('y')] >= _y_min_HR_) &
                    (frame[:,format_HR.index('y')] <= _y_max_HR_)
                )]
            
            final_image.append(frame)
            final_image.append(global_insert)

        else :
            final_image.append(frame)

        last_t_delay += 1
        last_t = t

    final_image = np.concatenate(final_image)
    return final_image



### MAIN ###
"""
if len(sys.argv) > 1 :
    # initialise parser
    import argparse

    parser = argparse.ArgumentParser(description="Form a puzzle from LR and HR data")
    parser.add_argument("image", metavar="HR", type=str, help="Input events at high resolution (insert image)")
    parser.add_argument("ROI", help="Detected ROIs", metavar="ROI", type=str)
    parser.add_argument("--insert-size", "-s", help="Size of the insert image, in pixels and in LR", metavar="S", type=int, default=4*5)
    parser.add_argument("--divider", "-div", help="Reduction coefficient", metavar="RC", type=int, default=4)
    args = parser.parse_args()

    HR = loadData(args.image)
    ROI = loadData(args.ROI)

    image = getPuzzle(
        insert_image=HR,
        ROI=ROI,
        div=args.divider,
        ROI_div=args.insert_size,
        size=(345,235),
        ROI_latency=1e6
    )
    np.save("events_puzzle.npy", image)
    print("Image correctly saved as events_puzzle.npy")
"""