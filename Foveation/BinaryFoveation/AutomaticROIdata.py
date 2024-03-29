from os import walk, path, makedirs
import time

# from pyNN.utility import normalized_filename
# from pyNN.utility.plotting import Figure, Panel
import numpy as np
import sys
from quantities import ms
import neo
from math import ceil
import itertools as it
from datetime import datetime as d
import h5py as h
# from reduceEvents import SpatialFunnelling
import argparse
# from getSaliencySpiNN import LastSpikeRecorder, dynamicWeightAdaptation

from pyNN.parameters import ParameterSpace
import nest 
from pyNN.nest.conversion import make_sli_compatible
from pyNN.errors import InvalidDimensionsError

# GLOBAL PARAMETERS
plus = 0
events_downscale_factor_1D = 4
reaction_downscale_factor_1D = 5
input_type = None

# CLASSES

class LastSpikeRecorder(object):
    
    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self._spikes = np.zeros(self.population.size)

    def __call__(self, t):
        # start = d.now()
        if t > 0:
            self._spikes = map(lambda x: x[-1].item() if len(x) > 0 else 0, self.population.get_data("spikes").segments[0].spiketrains)
            self._spikes = np.fromiter(self._spikes, dtype=float)
        # print("last spike", d.now() - start)
        return t+self.interval


class dynamicWeightAdaptation(object):
    def __init__(self, sampling_interval, projection, spikesReaction):
        self.interval = sampling_interval
        self.projection = projection

        # source & target
        self.source = projection.pre
        self.source_first_id = self.source.first_id
        self.target = projection.post
        self.spikesReaction = spikesReaction

        # weights
        self.default_w = np.ones((self.source.size, self.target.size)) * 0.7
        self._weights = []
        self.increase = 0.01
        
        # initialise connections
        conn_ = nest.GetConnections(source=np.unique(self.projection._sources).tolist())
        self.connections = {}
        self.source_mask = {}
        for postsynaptic_cell in self.target.local_cells:
            self.connections[postsynaptic_cell] = tuple([c for c in conn_ if c[1] == postsynaptic_cell])
            self.source_mask[postsynaptic_cell] = [x[0] - self.source_first_id for x in self.connections[postsynaptic_cell]]
    
    def __call__(self, t):
        # start = d.now()

        firing_t = self.spikesReaction._spikes
        w = self.projection.get('weight', format='array', with_address=False)
        np.nan_to_num(w,copy=False)

        # Get weights
        w = np.where(
            np.logical_and(
                firing_t > t-1,
                firing_t != 0
            ),
            w + self.increase,
            np.where(
                firing_t < t-100,
                np.maximum(w, self.default_w),
                w
            )
        ).astype('float64')

        # Set weights with nest
        attributes = self.projection._value_list_to_array({'weight':w})
        parameter_space = ParameterSpace(attributes,
                                         self.projection.synapse_type.get_schema(),
                                         (self.source.size, self.target.size))
        parameter_space = self.projection.synapse_type.translate( self.projection._handle_distance_expressions(parameter_space))

        parameter_space.evaluate(mask=(slice(None), self.target._mask_local))  # only columns for connections that exist on this machine

        for postsynaptic_cell, connection_parameters in zip(self.target.local_cells, parameter_space.columns()):    
            value = make_sli_compatible(connection_parameters['weight'])
            nest.SetStatus(self.connections[postsynaptic_cell], 'weight', value[self.source_mask[postsynaptic_cell]])
        self._weights.append(w)

        # print("dynamic weight adaptation",d.now() - start)
        return t+self.interval
    
    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal



# FUNCTIONS

def GaussianConnectionWeight(x,y, w_max=50.0):
    size=x*y
    # get coordonates of neurons in matrix
    coordonates = np.repeat(
        [np.transpose(
            np.unravel_index(
                np.arange(size), (x,y)
        ))],
        size, axis=0)
    # get coordonates of neurons in the diagonal
    diag_indices = np.transpose(
        np.unravel_index(
            np.arange(size), (x,y)
    )).reshape((size, 1, 2))
    # get distances between neurons in diagonal and in matrix
    distances = np.array(
        [ [np.linalg.norm(arr) for arr in lines]   for lines in abs(coordonates - diag_indices) ]
    )
    
    w = np.maximum( np.minimum( np.exp( distances ) * (1/size) , w_max), 0)
    
    return w

def getSensorSize(events):
    return int(np.max(events[::,0]))+1,int(np.max(events[::,1]))+1


def getDownscaledSensorSize(width, height, div):
    return ceil(width/div), ceil(height/div)     # to keep modified ? 

def getTimeLength(events, coord_t):
    return int(np.max(events[:,coord_t]))

def getPolarityIndex(coord_t):
    return (set([2,3]) - set([coord_t])).pop()

def getNegativeEventsValue(events, coord_p):
    return np.unique(events[events[:,coord_p] < 1][:,coord_p]).item()

def newEvent(x,y,p,t, coord_t):
    if coord_t == 2:
        return [x,y,t,p]
    elif coord_t == 3:
        return [x,y,p,t]

def ev2spikes(events,coord_t,width,height):
    # print("\nTranslating events to spikes... ")
    if not 1<coord_t<4:
        raise ValueError("coord_t must equals 2 or 3")
    
    coord_t-=2
    
    spikes=[[] for _ in range(width*height)]
    for x,y,*r in events: # tqdm(events):
        coord = int(np.ravel_multi_index( (int(x),int(y)) , (width, height) ))
        spikes[coord].append(float(r[coord_t]))
    # print("Translation done\n")
    return spikes

def spikes2ev(spikes, width, height, coord_t, polarity=1):
    events = np.zeros((0,4))
    for n in range(len(spikes)):
        x,y = np.unravel_index(n, (width, height))
        pixel_events = np.array([ newEvent(x,y,polarity, t.item(), coord_t) for t in spikes[n]])
        # print(n, pixel_events)
        try : 
            events = np.vstack((
                events,
                pixel_events
            ))
        except ValueError:
            pass

    events = events[events[:,coord_t].argsort()]
    return events

def getEvents(path_ev, size, spatial_reduce=False, time_reduce=False, time_factor=0.001):

    global input_type
    if path_ev.endswith("npz"):
        ev = np.load(path_ev)
        ev = np.concatenate((
            ev["x"].reshape(-1,1),
            ev["y"].reshape(-1,1),
            ev["p"].reshape(-1,1),
            ev["t"].reshape(-1,1)
        ), axis=1).astype('float64')
        ev = ev.copy()
        input_type = "numpy"
    
    elif path_ev.endswith("npy"):
        ev = np.load(path_ev)
        ev = ev.copy()
        input_type = "numpy"
    
    elif path_ev.endswith('hdf5'):
        f = h.File(path_ev, "r")
        ev = np.concatenate((
            f['event'][:,1].reshape(-1,1),
            f['event'][:,2].reshape(-1,1),
            f['event'][:,3].reshape(-1,1),
            f['event'][:,0].reshape(-1,1)
        ), axis=1).astype('float64')
        input_type = "hdf5"

    else:
        print("Error: The input file has to be of format .npy, .npz or .hdf5")
        sys.exit()

    # adapt to the 2 possible formalisms (x,y,p,t) or (x,y,t,p)
    if max(ev[:,2] == 1):
        coord_ts = 3
    elif max(ev[:,3] == 1):
        coord_ts = 2

    # adapt to time unit
    if time_reduce:
        ev[:,coord_ts] *= time_factor
    
    min_t = min(ev[:,coord_ts])
    ev[:,coord_ts] -= min_t
    ev[:,coord_ts] += 1.5
    # print(ev[:,coord_ts])

    (x_input, y_input) = size

    if spatial_reduce :
        global events_downscale_factor_1D
        events_downscale_factor_1D = args.divider
        if args.method == 'linear':
            ev_reduction = EventReduction(
                input_ev=ev, 
                coord_t=coord_ts, 
                div=events_downscale_factor_1D,
                width=size[0],
                height=size[1],
                cubic_interpolation=False
            )        
        else :
            ev_reduction = EventReduction(
                input_ev=ev, 
                coord_t=coord_ts, 
                div=events_downscale_factor_1D,
                width=size[0],
                height=size[1]
            )
        ev_reduction.run()
        ev = ev_reduction.events
        events_downscale_factor_1D = 1
        x_input, y_input = ev_reduction.width_downscale, ev_reduction.height_downscale
    
    if len(ev) == 0:
        return ev, x_input, y_input, 0, coord_ts

    sim_length = ceil(max(ev[:,coord_ts]))
    
    ev = ev2spikes( ev, coord_t=coord_ts, width=x_input, height=y_input )

    return ev, x_input, y_input, sim_length, coord_ts


def saveEvents(SNN_events, event_file, save_to):
    global input_type
    if not path.exists(save_to):
        makedirs(save_to)
    
    if input_type == 'numpy':
        np.save(
            path.join( save_to , event_file.replace("npz","npy")),
            SNN_events
        )
    
    elif input_type == 'hdf5':
        f = h.File(path.join( save_to, event_file), "w")
        f.create_dataset("event", data=SNN_events)
        f.close()


def init_SNN(sim, spikes, x_input, y_input):
    
    x_event = ceil(x_input/events_downscale_factor_1D)
    y_event = ceil(y_input/events_downscale_factor_1D)
    x_reaction = ceil(x_event/reaction_downscale_factor_1D)
    y_reaction = ceil(y_event/reaction_downscale_factor_1D)
       
    ######################## POPULATIONS ########################
    Input = sim.Population(
        x_input*y_input,
        sim.SpikeSourceArray(spike_times=spikes),
        label="Input")
    Input.record("spikes")

    Reaction_parameters = {
        'tau_m': 2.5,      # membrane time constant (in ms)
        'tau_refrac': 0.1,  # duration of refractory period (in ms)
        'v_reset': -100.0,   # reset potential after a spike (in mV)
        'v_rest': -65.0,    # resting membrane potential (in mV) !
        'v_thresh': -25,    # spike threshold (in mV)
    }

    Reaction = sim.Population(
        x_reaction * y_reaction,
        sim.IF_cond_exp(**Reaction_parameters),
        label="Reaction"
    )
    Reaction.record(("spikes","v"))

    ######################## CONNECTIONS ########################
    input2reaction_weights = []

    for X,Y in it.product(range(x_reaction), range(y_reaction)):
        idx = np.ravel_multi_index( (X,Y) , (x_reaction, y_reaction) ) 
        input2reaction_weights += [
            (
                np.ravel_multi_index( (x,y) , (x_input, y_input) ) , 
                idx
            )
            for x in range(int(reaction_downscale_factor_1D*X), int(reaction_downscale_factor_1D*(X+1))) if x < x_input
            for y in range(int(reaction_downscale_factor_1D*Y), int(reaction_downscale_factor_1D*(Y+1))) if y < y_input
        ]

    input2reaction = sim.Projection(
        Input, Reaction,
        connector=sim.FromListConnector(input2reaction_weights),
        synapse_type=sim.StaticSynapse(weight=1),
        receptor_type="excitatory",
        label="Connection events region to reaction"
    )

    # lateral inhibition on ROI detection
    WTA = sim.Projection(
        Reaction, Reaction,
        connector=sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_reaction, y_reaction, w_max=1)),
        receptor_type="inhibitory",
        label="Winner-Takes-All"
    )
    
    return Input, Reaction, input2reaction, x_reaction, y_reaction



def SNN_downscale(
    events,
    coord_t,
    Input,
    Reaction,
    input2reaction,
    x_reaction, y_reaction,
    sim_length=1e3,
    time_reduce=True,
    time_factor=1000
):
    """
    Arguments: 
    - events (numpy array): events to be reduced
    - coord_t (int): index of the timestamp coordinate 
    Optionnal arguments:
    - div (int): by how much to divide the events (spacial reduction)
    - density (float between 0 and 1): density of the downscaling
    - keep_polarity (boolean): wether to keep the polarity of events or ignore them (all downscaled events are positive)
    """

    Input.set(spike_times=events)

    spikesReaction = LastSpikeRecorder(sampling_interval=1.0, pop=Reaction)
    weightRuleInput2reaction = dynamicWeightAdaptation(sampling_interval=1.0, projection=input2reaction, spikesReaction=spikesReaction)
    callbacks = [spikesReaction, weightRuleInput2reaction]
    sim.run(sim_length, callbacks = callbacks)

    spikes = Reaction.get_data("spikes").segments[-1].spiketrains
    
    events = spikes2ev(spikes, x_reaction, y_reaction, coord_t)
    
    ###########################################################################################################################################

    events = np.vstack((np.zeros((0,4)), events))  # handles case where no spikes produced by simulation

    if time_reduce:
        events[:,coord_t] *= time_factor

    return events


# main
import pyNN.nest as sim
sim.setup(timestep=1)

parser = argparse.ArgumentParser(description="Automatically reduce data")
parser.add_argument("--dataset", "-da", nargs="+", help="Dataset repertory", metavar="D", type=str)
parser.add_argument("--divider","-div", metavar="d", type=int, help="Dividing factor", default=4)
parser.add_argument("--method", "-m", help="Reduction method (between 'funelling','eventcount', 'cubic', 'linear','none')", metavar="m", type=str)
args = parser.parse_args()

# Datasets :
# "/home/amelie/Scripts/Data/DVS128Gesture/DVS128G_classifier_data/",
# "/home/amelie/Scripts/Data/DDD17_datasets/Ev-SegNet_xypt/"

assert args.method in ['funelling','eventcount', 'cubic', 'linear','none']

if args.method == 'funelling':
    from reduceEvents import SpatialFunnelling as EventReduction
elif args.method == 'eventcount':
    from reduceEvents import EventCount as EventReduction
elif args.method == 'cubic' or args.method == 'linear':
    from reduceEvents import LogLuminance as EventReduction

if args.method == 'none':
    SpReduce = False
else : 
    SpReduce = True


for dr in args.dataset:
    print("\n\n>>",dr)

    if 'DVS128Gesture' in dr:
        original_repertory = dr+"DVSGesture/ibmGestureTest/"
        SNN_repertory = dr+"ROI_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(128,128)
        time_reduce = False
    elif 'DDD17' in dr:
        original_repertory = dr+"test/"
        SNN_repertory = dr+"ROI_data_"+args.method+"_div"+str(args.divider)+"/"
        size=(346,260)
        time_reduce = True

    # init=False
    
    for (rep_path, _, files) in walk(original_repertory):
        repertory=rep_path.replace(original_repertory, "")

        if len(files) > 0: # and repertory in ["test/0","test/1","train/0","train/1"]:
            
            for event_file in files :
                print(repertory, event_file, end=" ")

                if not path.exists(path.join( SNN_repertory, repertory, event_file.replace("npz","npy")) ):

                    i = 0
                    # while True:
                    original_events, x_input, y_input, sim_length, coord_ts = getEvents(
                        path.join(rep_path, event_file),
                        size,
                        spatial_reduce=SpReduce, 
                        time_reduce=time_reduce, 
                        time_factor=0.001
                    )

                    if len(original_events) == 0:
                        SNN_events = np.zeros((0,4))

                    else:
                        sim.end()
                        sim.setup(timestep=1)
                        Input, Reaction, input2reaction, x_reaction, y_reaction = init_SNN(sim, original_events,x_input,y_input)
                        print("Network has been correctly initialised", end=" ")

                        # SNN
                        start = d.now()
                        SNN_events = SNN_downscale(
                            original_events,
                            coord_ts,
                            Input, 
                            Reaction, 
                            input2reaction, 
                            x_reaction, y_reaction,
                            sim_length=sim_length,
                            time_reduce=time_reduce,
                            time_factor=1000
                        )
                        print(d.now() - start,end=" ")
                    
                    saveEvents(
                        SNN_events,
                        event_file,
                        path.join( SNN_repertory, repertory)
                    )
                    print("saved")
                
                else : 
                    print()