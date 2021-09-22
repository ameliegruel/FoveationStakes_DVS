import numpy as np

def getCoordLR(
    coordHR,    # value or numpy array
    reduction_coeff
    ):
    return np.floor(coordHR*reduction_coeff)

def getCoordHR(
    coordLR,    # value or numpy array
    reduction_coeff
    ):
    return np.floor(coordLR/reduction_coeff)

def LR2HR(LRimage, reduction_coeff):
    LRimage.T[0] = getCoordHR(LRimage.T[0], reduction_coeff)
    LRimage.T[1] = getCoordHR(LRimage.T[1], reduction_coeff)
    return LRimage

def getMinMax(coord, insert_size, fig_size):
    c_min = max(coord - insert_size, 0)
    c_max = min(c_min + insert_size, fig_size)
    return c_min, c_max

def getPuzzle(
    frame_image,   # low resolution events as np.array
    insert_image,  # high resolution events as np.array
    x_coord,       # in LR
    y_coord,       # in LR
    insert_size = 5,  # size of one side of the insert, in pixels and in LR
    reduction_coeff = 0.25
    ):

    # get size of frame image (LR)
    x_size_frame = max(frame_image.T[0])
    y_size_frame = max(frame_image.T[1])

    # min and max coordinates of insert part in HR
    x_min_HR, x_max_HR = getMinMax(getCoordHR(x_coord, reduction_coeff), getCoordHR(insert_size, reduction_coeff), getCoordHR(x_size_frame, reduction_coeff))
    y_min_HR, y_max_HR = getMinMax(getCoordHR(y_coord, reduction_coeff), getCoordHR(insert_size, reduction_coeff), getCoordHR(y_size_frame, reduction_coeff))
    print("Region of interest's coordonnates:")
    print("> en x :", x_min_HR, x_max_HR)
    print("> en y :", y_min_HR, y_max_HR)
    
    # min and max coordinates of insert part in LR
    x_min_LR, x_max_LR = getMinMax(x_coord, insert_size, x_size_frame)
    y_min_LR, y_max_LR = getMinMax(y_coord, insert_size, y_size_frame)
    
    # get insert part from insert image (HR)
    insert = insert_image[
        (insert_image.T[0] > x_min_HR) &
        (insert_image.T[0] <= x_max_HR) &
        (insert_image.T[1] > y_min_HR) &
        (insert_image.T[1] <= y_max_HR)
    ]

    # remove insert part from frame image (LR)
    frame_image = frame_image[
        (frame_image.T[0] < x_min_LR) |
        (frame_image.T[0] >= x_max_LR) |
        (frame_image.T[1] < y_min_LR) |
        (frame_image.T[1] >= y_max_LR)
    ]

    # translate frame image (LR) into HR coordinates
    frame_image = LR2HR(frame_image, reduction_coeff)

    # puzzle frame image and LR image
    final_image = np.concatenate((frame_image, insert))
    return final_image



### MAIN ###
import argparse

parser = argparse.ArgumentParser(description="Form a puzzle from LR and HR data")
parser.add_argument("low_resolution_image", metavar="LR", type=str, nargs=1, help="Input events at low resolution (frame image)")
parser.add_argument("high_resolution_image", metavar="HR", type=str, nargs=1, help="Input events at high resolution (insert image)")
parser.add_argument("coordinates", help="Coordinates of insert image", nargs=2, metavar="C", type=int)
parser.add_argument("--insert_size", "-s", help="Size of the insert image, in pixels and in LR", nargs=1, metavar="S", type=int, default=[5])
parser.add_argument("--reduction_coefficient", "-rc", help="Reduction coefficient", nargs=1, metavar="RC", type=float, default=[0.25])
args = parser.parse_args()

LR = np.load(args.low_resolution_image[0])
HR = np.load(args.high_resolution_image[0])

image = getPuzzle(
    frame_image=LR, 
    insert_image=HR,
    x_coord=args.coordinates[0],
    y_coord=args.coordinates[1],
    insert_size=args.insert_size[0],
    reduction_coeff=args.reduction_coefficient[0]
)
np.save("events_puzzle.npy", image)