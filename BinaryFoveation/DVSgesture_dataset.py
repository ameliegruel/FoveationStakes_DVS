import os
import sys
import numpy as np
import zipfile as z
from torchvision.datasets.vision import VisionDataset


class DVSGesture_dataset(VisionDataset):
    """DVSGesture <http://research.ibm.com/dvsgesture/> dataset, either reduced or foveated.
    arguments:
        repertory: root repertory where the different processed datasets are stored 
        type_data: must be either test (to load test data), reduce (to load reduced data) or fovea (to load foveated data)
        spatial_divider: spatial dividing factor (all data is divided by 4, no need to change default)
        structural_divider: structural dividing factor, to keep only a certain percentage of the processed data 
                    (if want to charge the whole processed data, set structural_divider to 100)
        method: spacial downscaling method 
                    (if type_data not 'test', roi_method must be either 'funnelling', 'eventcount', 'linear' or 'cubic')
        roi_method: ROI data to use during foveation 
                    if type_data is 'fovea' and roi is maintained to default None, then use data foveated on whole input data ; 
                    else, use data foveated on downscaled data using a certain method)
    """

    classes = [
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    ]

    sensor_size = (128, 128)
    ordering = "xypt"

    def __init__(
        self, repertory, type_data, spatial_divider=4, structural_divider=100, method=None, roi_method=None, 
    ):
        super(DVSGesture_dataset, self).__init__(repertory)
        assert type_data in ['test', 'reduce', 'fovea'], "Wrong 'type_data' argument"

        if type_data == 'test':
            self.zip_name = type_data
            self.folder_name = 'Test data/'
        
        elif type_data == 'reduce':
            assert method in ['funnelling', 'eventcount', 'linear', 'cubic'], "Wrong 'method' argument"
            self.zip_name = 'reduced_data_'+method+'_div'+str(spatial_divider)
            self.folder_name = 'Reduced data/Method - '+method+'/'
        
        else : 
            assert method in ['funnelling', 'eventcount', 'linear', 'cubic'], "Wrong 'method' argument"
            assert roi_method in ['funnelling', 'eventcount', 'linear', 'cubic', None], "Wrong 'roi_method' argument"
            if roi_method == None:
                roi_method = 'no reduc'
            self.folder_name = 'Foveated data/ROI data - '+roi_method+'/Method '+method+'/'
            self.zip_name = 'foveated_data_'+method+'_div'+str(spatial_divider)+'_ROI'+roi_method

        if structural_divider != 100:
            assert structural_divider in [5,10,20,40,60,80], "Wrong 'structural_divider' argument"
            self.zip_name += '_'+str(structural_divider)+'%'

        self.location_on_system = repertory
        self.data = []
        self.samples = []
        self.targets = []

        file_path = os.path.join(self.location_on_system, self.folder_name, self.zip_name)
        
        if not os.path.exists(file_path) and os.path.exists(file_path+'.zip'):
            
            print('Extracting into '+file_path+'...')
            with z.ZipFile(file_path+'.zip', 'r') as zip_dir :
                zip_dir.extractall(os.path.join(self.location_on_system, self.folder_name))
            print('Extraction done')

        if os.path.exists(file_path):
    
            for path, dirs, files in os.walk(file_path):
                dirs.sort()
                for file in files:
                    if file.endswith("npy"):
                        self.samples.append(path + "/" + file)
                        self.targets.append(int(file[:-4]))

        else: 
            print('Error: The folder '+file_path+' does not exist')
            sys.exit()

    def __getitem__(self, index):
        events = np.load(self.samples[index])
        events[:, 3] *= 1000  # convert from ms to us
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.samples)