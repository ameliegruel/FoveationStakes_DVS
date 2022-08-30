import numpy as np
import sys
# from reduceEvents import EventCount as EventReduction
import h5py as h

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
    insert_size,
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
        x_min_HR, x_max_HR = getMinMax(getCoordHR(x, div), insert_size, getCoordHR(size[0], div))
        y_min_HR, y_max_HR = getMinMax(getCoordHR(y, div), insert_size, getCoordHR(size[1], div))

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

    X = insert_image[:,format_HR.index('x')]
    Y = insert_image[:,format_HR.index('y')]
    P = insert_image[:,format_HR.index('p')]
    T = insert_image[:,format_HR.index('t')]
    
    # get frame image (LR)
    if len(frame_image) == 0:
        frame_image = getFrameImage(X,Y,P,T,div)
        # translate frame image (LR) into HR coordinates
        frame_image = LR2HR(frame_image, div, format_HR)

    # get size of frame image (HR)
    x_size_frame, y_size_frame = size

    # get transformed ROIs min and max x and y coordinates, respectively in LR and HR
    transformed_ROI_HR = transformRoi(ROI, ROI_div, size, div)

    # launch puzzle
    last_t = 0
    x_min_HR = y_min_HR = [-1]
    x_max_HR = [x_size_frame]
    y_max_HR = [y_size_frame]

    final_image = np.zeros((0,4))

    for t in T:

        insert = insert_image[T == t]
        frame = frame_image.copy()
    
        if t in transformed_ROI_HR[:,-1]:
            last_t = 0
            x_min_HR, x_max_HR, y_min_HR, y_max_HR = transformed_ROI_HR[
                transformed_ROI_HR[:,-1] == t
            ][:,:-1].T

        elif last_t > ROI_latency:
            x_min_HR = y_min_HR = [-1]
            x_max_HR = [x_size_frame]
            y_max_HR = [y_size_frame]
        
        global_insert = np.zeros((0,4))
        
        if len(x_min_HR) > 1:

            for _x_min_HR_,_x_max_HR_, _y_min_HR_, _y_max_HR_ in zip(x_min_HR,x_max_HR, y_min_HR, y_max_HR) :

                # get insert part from insert image (HR)      
                insert_ = insert[
                    (insert[:,format_HR.index('x')] > _x_min_HR_) &
                    (insert[:,format_HR.index('x')] < _x_max_HR_) &
                    (insert[:,format_HR.index('y')] > _y_min_HR_) &
                    (insert[:,format_HR.index('y')] < _y_max_HR_)
                ]
                global_insert = np.vstack((global_insert, insert_))

                # update mask
                frame = frame[~(
                    (frame[:,0] >= _x_min_HR_) &
                    (frame[:,0] <= _x_max_HR_) &
                    (frame[:,1] >= _y_min_HR_) &
                    (frame[:,1] <= _y_max_HR_)
                )]
        
        else : 
            # get insert part from insert image (HR)      
            insert_ = insert[
                (insert[:,format_HR.index('x')] > x_min_HR[0]) &
                (insert[:,format_HR.index('x')] < x_max_HR[0]) &
                (insert[:,format_HR.index('y')] > y_min_HR[0]) &
                (insert[:,format_HR.index('y')] < y_max_HR[0])
            ]
            global_insert = np.vstack((global_insert, insert_))

            # update mask
            frame = frame[~(
                (frame[:,0] >= x_min_HR[0]) &
                (frame[:,0] <= x_max_HR[0]) &
                (frame[:,1] >= y_min_HR[0]) &
                (frame[:,1] <= y_max_HR[0])
            )]

        # puzzle frame image and LR image
        final_image = np.vstack((
            final_image,
            frame[frame[:,-1] == t],
            global_insert
        ))

        last_t += 1

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