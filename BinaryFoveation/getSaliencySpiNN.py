#!/bin/python3

"""
Spiking neural network for saliency detection in event-camera data
Author : Amélie Gruel - Université Côte d'Azur, CNRS/i3S, France - amelie.gruel@i3s.unice.fr
Run as : $ python3 getSaliency.py spiNNaker --events [events]
Use of DDD17 (DVS Driving Dataset 2017) annotated by [Alonso, 2017]
"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
from events2spikes import ev2spikes
from reduceEvents import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from quantities import ms
import neo
from math import ceil
import itertools as it
from datetime import datetime as d
from reduceEvents import SpatialFunnelling


################################################### FUNCTIONS #################################################################

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
    
    w = np.maximum( np.minimum( np.exp( distances ) * (1/size) , w_max), 0).reshape(-1)
    return w




################################################### MAIN #################################################################

# Configure simulator
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--events","Get input file with inputs"),
                             ("--reduce", "Spatially reduce input data", {"action": "store_true"}),
                             ("--unit", "Temporally adapt input data", {"default": "milli"}),
                             ("--save", "Save the data output by all layers as npy files", {"action":"store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}))

if options.debug:
    init_logging(None, debug=True)

sim_ = "spiNNaker"

sim.setup(timestep=10)

events_downscale_factor_1D = 4
reaction_downscale_factor_1D = 5

######################## GET DATA ########################
try :
    ev = np.load(options.events)

    # adapt to the 2 possible formalisms (x,y,p,t) or (x,y,t,p)
    if max(ev[:,2] == 1):
        coord_p = 2
        coord_ts = 3
    elif max(ev[:,3] == 1):
        coord_p = 3
        coord_ts = 2

    # adapt to time unit
    if options.unit == "s":
        ev[:,coord_ts] *= 1000
    elif options.unit == "nano":
        ev[:,coord_ts] *= 0.001

except (ValueError, FileNotFoundError, TypeError):

    try : 
        ev = np.load(options.events)
        ev = np.concatenate((
            ev["x"].reshape(-1,1),
            ev["y"].reshape(-1,1),
            ev["p"].reshape(-1,1),
            ev["t"].reshape(-1,1)
        ), axis=1).astype('float64')

        # adapt to the 2 possible formalisms (x,y,p,t) or (x,y,t,p)
        if max(ev[:,2] == 1):
            coord_p = 2
            coord_ts = 3
        elif max(ev[:,3] == 1):
            coord_p = 3
            coord_ts = 2

        # adapt to time unit
        if options.unit == "s":
            ev[:,coord_ts] *= 1000
        elif options.unit == "nano":
            ev[:,coord_ts] *= 0.001
        
        min_t = min(ev[:,coord_ts])
        ev[:,coord_ts] -= min_t

    except: 
        print("Error: The input file has to be of format .npy")
        sys.exit()

x_input, y_input = None, None
if options.reduce :
    ev_reduction = SpatialFunnelling(input_ev=ev, coord_t=coord_ts, div=events_downscale_factor_1D)
    ev_reduction.run()
    ev = ev_reduction.events
    events_downscale_factor_1D = 1

max_time = int(np.max(ev[:,coord_ts]))
print("\nTime length of the data :",max_time)
start_time = 0
stop_time = 1e6

try :
    ev = ev[np.logical_and( ev[:,coord_ts]>start_time , ev[:,coord_ts]<stop_time )]
    ev[:,coord_ts] -= start_time
    time_data = int(np.max(ev[:,coord_ts]))
    ev, x_input, y_input = ev2spikes( ev )
    print("Simulation will be run on", len(ev), "events\n")

except ValueError:
    
    print("Error: The start and stop time you defined for the simulation are not coherent with the data.")
    sys.exit(0)

x_event = ceil(x_input/events_downscale_factor_1D)
y_event = ceil(y_input/events_downscale_factor_1D)
x_reaction = ceil(x_event/reaction_downscale_factor_1D)
y_reaction = ceil(y_event/reaction_downscale_factor_1D)

################################################ NETWORK ################################################


######################## POPULATIONS ########################
print("Initiating populations...")

Input = sim.Population(
    x_input*y_input,
    sim.SpikeSourceArray(spike_times=ev),
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
    label="Reaction",
    # structure=Grid2D(dx=reaction_downscale_factor_1D, dy=reaction_downscale_factor_1D, z=50, aspect_ratio=x_event/y_event)
)

Reaction.record(("spikes","v"))
print("\nSize of populations :\n> Input", Input.size, "with shape",(x_input, y_input), "\n> Reaction", Reaction.size, "with shape", (x_reaction, y_reaction), end="\n\n")


######################## CONNECTIONS ########################
print("Initiating connections...")

input2reaction_connections = []

for X,Y in it.product(range(x_reaction), range(y_reaction)):
    idx = np.ravel_multi_index( (X,Y) , (x_reaction, y_reaction) ) 
    input2reaction_connections += [
        (
            np.ravel_multi_index( (x,y) , (x_input, y_input) ) , 
            idx
        )
        for x in range(int(reaction_downscale_factor_1D*X), int(reaction_downscale_factor_1D*(X+1))) if x < x_input
        for y in range(int(reaction_downscale_factor_1D*Y), int(reaction_downscale_factor_1D*(Y+1))) if y < y_input
    ]

timing_rule = sim.SpikePairRuleWeightAdaptation(t_max=100)
weight_rule = sim.AdditiveWeightDependenceWeightAdaptation()

WeightAdaptation_model = sim.WeightAdaptationMechanism(
    timing_rule, weight_rule,
    weight=1
)

input2reaction = sim.Projection(
    Input, Reaction,
    connector=sim.FromListConnector(input2reaction_connections),
    synapse_type=WeightAdaptation_model,
    receptor_type="excitatory",
    label="Connection events region to reaction"
)

# lateral inhibition on ROI detection
WTA = sim.Projection(
    Reaction, Reaction,
    connector=sim.AllToAllConnector(allow_self_connections=True),#False),
    synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_reaction, y_reaction, w_max=1)), #w_max=0.05)),
    receptor_type="inhibitory",
    label="Winner-Takes-All"
)

print("Network initialisation OK")


######################## RUN SIMULATION ########################

class WeightRecorder(object):

    def __init__(self, sampling_interval, proj):
        self.interval = sampling_interval
        self.projection = proj

        self._weights = []
       
    def __call__(self, t):
        if t > 0:
            weight = self.projection.get('weight',format='array', with_address=False)
            # np.nan_to_num(weight,copy=False)
            weight = weight[~np.isnan(weight)]
            self._weights.append(weight)
        return t+self.interval

    def update_weights(self, w):
        assert self._weights[-1].shape == w.shape
        self._weights[-1] = w

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms, name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal

class visualiseTime(object):
    def __init__(self, sampling_interval):
        self.interval = sampling_interval

    def __call__(self, t):
        print(t)
        return t + self.interval


visualise_time = visualiseTime(sampling_interval=10.0)
weightRuleInput2reaction = WeightRecorder(sampling_interval=1.0, proj=input2reaction)

print("\nStart simulation ...")
start = d.now()
sim.run(time_data, callbacks=[visualise_time,weightRuleInput2reaction])
print("Simulation done in", d.now() - start ,"\n")


######################## SIMULATION RESULTS ########################

Input_data = Input.get_data().segments[0]
Reaction_data = Reaction.get_data().segments[0]
weights = weightRuleInput2reaction.get_weights()

figure_filename = normalized_filename("Results", "Saliency detection", "png", options.simulator)

if options.plot_figure :
    fig = plt.figure()
    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, xlabel="ROI detector spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),

        # # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI detector layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # evolution of the synaptic weights with time
        Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Intermediate Output Weight",
                legend=False, xlim=(0, time_data)),
            
        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ).save(figure_filename)
    print("Figure correctly saved as", figure_filename)
    plt.show()


######################## SAVE DATA ########################

def newEvent(x,y,p,t, coord_t):
    if coord_t == 2:
        return [x,y,t,p]
    elif coord_t == 3:
        return [x,y,p,t]

def spikes2ev(spikes, width, height, coord_t, polarity=1):
    events = np.zeros((0,4))
    for n in range(len(spikes)):
        x,y = np.unravel_index(n, (width, height))
        pixel_events = np.array([ newEvent(x,y,polarity,t.item(), coord_t) for t in spikes[n]])
        try :
            events = np.vstack((
                events,
                pixel_events
            ))
        except ValueError:
            pass
    
    # adapt to time unit
    if options.unit == "s":
        events[:,coord_ts] /= 1000
    elif options.unit == "nano":
        events[:,coord_ts] /= 0.001

    events = events[events[:,coord_t].argsort()]
    return events

if options.save : 
    Reaction_filename = normalized_filename("Results", "Saliency detection - ROI detector", "npy", options.simulator)
    np.save(
        Reaction_filename,
        spikes2ev(Reaction_data.spiketrains, x_reaction, y_reaction, coord_ts)
    )
    print("Output data from ROI detector layer correctly saved as", Reaction_filename)

sim.end()