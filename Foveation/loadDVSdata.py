from DVSgesture_dataset import DVSGesture_dataset
dir = '/home/amelie/Scripts/Data/DVS128Gesture/tonic_data/ZIP files/'

# load data from test dataset
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'test',
    spatial_divider=1
)

# load data reduced using cubic
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'reduce',
    method='cubic'
)

# load 40% of data reduced using eventcount
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'reduce',
    method='eventcount', 
    structural_divider=40
)

# load data foveated using linear, with ROI obtained on data reduced using cubic
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'fovea',
    method='linear',
    roi_method='cubic'
)

# load 5% of data foveated using cubic, with ROI obtained on data reduced using funnelling
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'fovea',
    method='cubic',
    roi_method='funnelling',
    structural_divider=5
)

# load data foveated using eventcount, with ROI obtained on whole data (no reduction)
d = DVSGesture_dataset(
    repertory=dir, 
    type_data= 'fovea',
    method='eventcount'
)

# read first event of 99 first samples
for i, (events, target) in enumerate(iter(d)):
    print(i, target, events[0])
    if i > 99:
        break