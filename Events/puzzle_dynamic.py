import numpy as np
import threading

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
        self.x_ROI = -1
        self.y_ROI = -1

        self.x_min_LR = self.x_max_LR = self.x_min_HR = self.x_max_HR = -1
        self.y_min_LR = self.y_max_LR = self.y_min_HR = self.y_max_HR = -1

        # time window
        self.setTimeWindow(0, time_window)

        # final puzzle image
        self.final_puzzle_image = None

        self.count = 0 # tmp, to debug


    def setROIcoord(
        self,
        type,   # type of coordinate : "x" or "y"
        coord   # value of coordinate
    ): 
        if type == "x":
            self.x_ROI = coord
        if type == "y":
            self.y_ROI = coord

    
    def setTimeWindow(self, min_tw, max_tw):
        self.min_tw = min_tw
        self.max_tw = max_tw
        self.run = self.max_tw < np.max(self.insert_image.T[2]) # True as long as all the events haven't been processed
        self.last_ROI = [self.x_ROI, self.y_ROI] # since this method is run only on initialisation and when a timewindow is over, it's a good place to set the last ROI
        

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
        c_min = max(coord - insert_size/2, 0)
        c_max = min(c_min + insert_size, fig_size)
        return c_min, c_max

    def setFinalImage(self, puzzle_for_current_time_window):
        try : 
            self.final_puzzle_image = np.concatenate((self.final_puzzle_image, puzzle_for_current_time_window))
        except ValueError :
            self.final_puzzle_image = puzzle_for_current_time_window

    def getPuzzle(self):
        # debug
        # self.count += 1
        # print("Puzzle "+str(self.count)+" - tw : "+str(self.max_tw))

        # handle option with no region of interest
        if self.x_ROI == -1 or self.y_ROI == -1:
            self.setFinalImage(self.frame_image)

        if self.x_ROI != self.last_ROI[0] :
            # min and max coordinates of insert part in HR
            self.x_min_HR, self.x_max_HR = self.getMinMax(self.getCoordHR(self.x_ROI), self.insert_size_HR, self.x_size_frame_HR)
            print("> en x :", self.x_min_HR, self.x_max_HR)
            # min and max coordinates of insert part in LR
            self.x_min_LR, self.x_max_LR = self.getMinMax(self.x_ROI, self.insert_size_LR, self.x_size_frame_LR)
        
        if self.y_ROI != self.last_ROI[1] :
            # min and max coordinates of insert part in HR
            self.y_min_HR, self.y_max_HR = self.getMinMax(self.getCoordHR(self.y_ROI), self.insert_size_HR, self.y_size_frame_HR)
            print("> en y :", self.y_min_HR, self.y_max_HR)
            # min and max coordinates of insert part in LR
            self.y_min_LR, self.y_max_LR = self.getMinMax(self.y_ROI, self.insert_size_LR, self.y_size_frame_LR)
        
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

        # update timewindow and continue (or not)
        self.setTimeWindow(self.max_tw, self.max_tw + self.max_tw-self.min_tw)
        if self.run :
            self.getPuzzle()



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
events_puzzle.setROIcoord("x", x_ROI)
events_puzzle.setROIcoord("y", y_ROI)
#events_puzzle.getPuzzle()

# threading 
thread_puzzle = threading.Thread(target=events_puzzle.getPuzzle)
thread_puzzle.daemon = True
thread_puzzle.start()

while events_puzzle.run:
    if input() == "x":
        events_puzzle.setROIcoord("x", int(input("> new value for ROI's x: ")))
    if input() == "y":
        events_puzzle.setROIcoord("y", int(input("> new value for ROI's y: ")))
        

np.save("events_puzzle.npy", events_puzzle.final_puzzle_image)
print("Max tw :",events_puzzle.max_tw)
