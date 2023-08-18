import elephant
import pickle
import argparse
from tqdm import tqdm
import numpy as np

# set up parser
parser = argparse.ArgumentParser(description="Translate output data from pkl file into 3 csv files (one for each spike encoding)")
parser.add_argument("data", metavar="D", type=str, nargs=1, help="Input data")
parser.add_argument("shape", metavar="S", type=int, nargs=2, help="Height and width of the layer producing the input data")
args = parser.parse_args()


# open data
with open(args.data[0], "rb") as f:
    data = pickle.load(f)

spikes_neo = elephant.neo_tools.get_all_spiketrains(data)
spikes_np = np.zeros((0,3))

# generate data with x,y
for n in range(len(spikes_neo)):
    x,y = np.unravel_index(n, args.shape)
    spikes_np = np.vstack((
        spikes_np, 
        [[x,y,t.item()] for t in spikes_neo[n]]
    ))   

# save files
np.save(args.data[0][:-3]+".npy", spikes_np)