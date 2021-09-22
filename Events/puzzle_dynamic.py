import numpy as np
from numpy.lib.function_base import select

class Puzzle():

    def __init__(
        self,
        frame_image,        # low resolution events as np.array
        insert_image,       # high resolution events as np.array
        insert_size = 5,    # size of one side of the insert, in pixels and in LR
        reduction_coeff = 0.25,
        time_window = 10    # length of time window in ms
        ):  
        
        self.frame_image = frame_image    # low resolution events as np.array
        self.insert_image = insert_image  # high resolution events as np.array
        self.reduction_coeff = reduction_coeff
        
        # LR
        self.insert_size_LR = insert_size    # size of one side of the insert, in pixels and in LR
        # get size of frame image (LR)
        self.x_size_frame_LR = max(self.frame_image.T[0])
        self.y_size_frame_LR = max(self.frame_image.T[1])

        # HR
        self.insert_size_HR = self.getCoordHR(insert_size)    # size of one side of the insert, in pixels and in LR
        # get size of frame image (LR)
        self.x_size_frame_HR = max(self.insert_image.T[0])
        self.y_size_frame_HR = max(self.insert_image.T[1])

        # ROI coordonates
        self.x_coord = -1
        self.y_coord = -1

        self.x_min_LR = self.x_max_LR = self.x_min_HR = self.x_max_HR = -1
        self.y_min_LR = self.y_max_LR = self.y_min_HR = self.y_max_HR = -1

        # time window
        self.min_tw = 0
        self.max_tw = time_window
        self.run = self.max_tw < np.max(self.insert_image.T[2])  # True as long as all the events haven't been processed

        # final puzzle image
        self.final_puzzle_image = None
    

    def setROIcoord(
        self,
        xROI,
        yROI
    ): 
        self.x_coord = xROI
        self.y_coord = yROI
        

    def getCoordLR(
        self,
        coordHR    # value or numpy array
        ):
        return np.floor(coordHR*self.reduction_coeff)

    def getCoordHR(
        self,
        coordLR    # value or numpy array
        ):
        return np.floor(coordLR/self.reduction_coeff)

    def LR2HR(self, LRimage):
        LRimage.T[0] = self.getCoordHR(LRimage.T[0])
        LRimage.T[1] = self.getCoordHR(LRimage.T[1])
        return LRimage

    def getMinMax(self, coord, insert_size, fig_size):
        c_min = max(coord - insert_size, 0)
        c_max = min(c_min + insert_size, fig_size)
        return c_min, c_max

    def setFinalImage(self, puzzle_for_current_time_window):
        if self.final_puzzle_image == None : 
            self.final_puzzle_image = puzzle_for_current_time_window
        else :
            self.final_puzzle_image = np.concatenate((self.final_puzzle_image, puzzle_for_current_time_window))

    def getPuzzle(
        self, 
        x_coord,      # in LR
        y_coord       # in LR
        ):

        # handle case with no region of interest
        if x_coord == -1 or y_coord == -1:
            self.setFinalImage(self.frame_image)

        if x_coord != self.x_coord :
            self.x_coord = x_coord
            # min and max coordinates of insert part in HR
            self.x_min_HR, self.x_max_HR = self.getMinMax(self.getCoordHR(x_coord), self.insert_size_HR, self.x_size_frame_HR)
            print("en x :", self.x_min_HR, self.x_max_HR)
            # min and max coordinates of insert part in LR
            self.x_min_LR, self.x_max_LR = self.getMinMax(x_coord, self.insert_size_LR, self.x_size_frame_LR)
        
        if y_coord != self.y_coord :
            self.y_coord = y_coord
            # min and max coordinates of insert part in HR
            self.y_min_HR, self.y_max_HR = self.getMinMax(self.getCoordHR(y_coord), self.insert_size_HR, self.y_size_frame_HR)
            print("en y :", self.y_min_HR, self.y_max_HR)
            # min and max coordinates of insert part in LR
            self.y_min_LR, self.y_max_LR = self.getMinMax(y_coord, self.insert_size_LR, self.y_size_frame_LR)
        
        # get insert part from insert image (HR)
        insert = self.insert_image[
            (self.insert_image.T[0] >  self.x_min_HR) &
            (self.insert_image.T[0] <= self.x_max_HR) &
            (self.insert_image.T[1] >  self.y_min_HR) &
            (self.insert_image.T[1] <= self.y_max_HR) &
            (self.insert_image.T[2] >  self.min_tw) &
            (self.insert_image.T[2] <= self.max_tw)
        ]

        # remove insert part from frame image (LR)
        frame_image = self.frame_image[
            ((self.frame_image.T[0] < self.x_min_LR) |
            (self.frame_image.T[0] >= self.x_max_LR) |
            (self.frame_image.T[1] <  self.y_min_LR) |
            (self.frame_image.T[1] >= self.y_max_LR)) &
            (self.frame_image.T[2] >  self.min_tw) &
            (self.frame_image.T[2] <= self.max_tw)
        ]

        # translate frame image (LR) into HR coordinates
        frame_image = self.LR2HR(frame_image)

        # puzzle frame image and LR image
        final_image_for_current_timewindow = np.concatenate((frame_image, insert))
        self.setFinalImage(final_image_for_current_timewindow)
        



### MAIN ###
import argparse

parser = argparse.ArgumentParser(description="Form a puzzle from LR and HR data")
parser.add_argument("low_resolution_image", metavar="LR", type=str, nargs=1, help="Input events at low resolution (frame image)")
parser.add_argument("high_resolution_image", metavar="HR", type=str, nargs=1, help="Input events at high resolution (insert image)")
# parser.add_argument("coordinates", help="Coordinates of insert image", nargs=2, metavar="C", type=int)
parser.add_argument("--insert_size", "-s", help="Size of the insert image, in pixels and in LR", nargs=1, metavar="S", type=int, default=[5])
parser.add_argument("--reduction_coefficient", "-rc", help="Reduction coefficient", nargs=1, metavar="RC", type=float, default=[0.25])
parser.add_argument("--time_window", "-tw", help="Time window length (in ms)", nargs=1, metavar="TW", type=float, default=[10])
args = parser.parse_args()


LR = np.load(args.low_resolution_image[0])
HR = np.load(args.high_resolution_image[0])

events_puzzle = Puzzle(
    frame_image=LR, 
    insert_image=HR,
    insert_size=args.insert_size[0],
    reduction_coeff=args.reduction_coefficient[0],
    time_window=args.time_window[0]
)

x_ROI = int(input("x frame size: "+str(events_puzzle.x_size_frame_LR)+" ==> ROI x coordinate : "))
y_ROI = int(input("y frame size: "+str(events_puzzle.y_size_frame_LR)+" ==> ROI y coordinate : "))
events_puzzle.getPuzzle(
    x_coord=x_ROI,
    y_coord=y_ROI
)
np.save("events_puzzle.npy", events_puzzle.final_puzzle_image)