#!/bin/python3

"""
Spiking neural network for saliency detection in event-camera data
Author : Amélie Gruel - Université Côte d'Azur, CNRS/i3S, France - amelie.gruel@i3s.unice.fr
Run as : $ python3 getSaliency.py nest
Use of DDD17 (DVS Driving Dataset 2017) annotated by [Alonso, 2017]
"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
import sys
from quantities import ms
import neo
from math import ceil
import itertools as it
from datetime import datetime as d
import h5py as h
from tqdm import tqdm

from pyNN.parameters import ParameterSpace
import nest
from pyNN.nest.conversion import make_sli_compatible

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

    w = np.maximum( np.minimum( np.exp( distances ) * (1/size) , w_max), 0)
    if sim_ == "spiNNaker" :
        w = w.reshape(-1)

    return w

def ev2spikes(events,coord_t, width,height):
    # print("\nTranslating events to spikes... ")
    # if not 1<coord_t<4:
    #     raise ValueError("coord_t must equals 2 or 3")

    if coord_t == 0:
        coord_x, coord_y = 1,2
    else: 
        coord_x, coord_y = 0,1

    spikes=[[] for _ in range(width*height)]
    for ev in events:
        coord = int(np.ravel_multi_index( (int(ev[coord_x]),int(ev[coord_y])) , (width, height) ))
        spikes[coord].append(float(ev[coord_t]))
    return spikes

def getEvents(path_ev, size, spatial_reduce=False, time_reduce=False, time_factor=0.001, method='eventcount'):

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
    
    elif path_ev.endswith('txt'):
        ev = []
        f = open(path_ev, 'r')
        i = 0
        for line in tqdm(f.readlines()):
            if line != '':
                ev.append([[float(e) for e in line.split()]])
                i += 1
        ev = np.concatenate(ev).astype('float')
        input_type = 'txt'

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
    if max(ev[:,0]) > size[0] or max(ev[:,0]) < size[0] - 5:
        coord_ts = 0
    elif max(ev[:,2] == 1):
        coord_ts = 3
    elif max(ev[:,3] == 1):
        coord_ts = 2

    # adapt to time unit
    if input_type == 'txt':
        ev[:,coord_ts] *= 1e6

    if time_reduce:
        ev[:,coord_ts] *= time_factor
    min_t = min(ev[:,coord_ts])
    ev[:,coord_ts] -= min_t
    ev[:,coord_ts] += 1.5
    # print(ev[:,coord_ts])

    (x_input, y_input) = size

    if spatial_reduce :
        global events_downscale_factor_1D
        if method == 'linear':
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

    sim_length = 1e3
    if len(ev) != 0 and max(ev[:,coord_ts]) < sim_length :
        sim_length = ceil(max(ev[:,coord_ts]))

    ev = ev2spikes( ev, coord_t=coord_ts, width=x_input, height=y_input )

    return ev, x_input, y_input, sim_length, coord_ts


################################################### MAIN #################################################################

# Configure simulator
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--events","Get input file with inputs"),
                             ("--method", "Reduction method (between 'funelling','eventcount', 'cubic', 'linear','none')", {"default": 'none'}),
                             ("--save", "Save the data output by all layers as npy files", {"action":"store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}))

if options.debug:
    init_logging(None, debug=True)

sim_ = ""
if "nest" in sim.__file__:
    sim_ = "nest"
elif "spiNNaker" in sim.__file__:
    sim_ = "spiNNaker"

dt = 1/80000
sim.setup(timestep=1)

events_downscale_factor_1D = 2 #4
reaction_downscale_factor_1D = 5

assert options.method in ['funelling','eventcount', 'cubic', 'linear','none']

time_reduce = True
spatial_reduce = True
if options.method == 'funelling':
    from reduceEvents import SpatialFunnelling as EventReduction
elif options.method == 'eventcount':
    from reduceEvents import EventCount as EventReduction
elif options.method == 'cubic' or options.method == 'linear':
    from reduceEvents import LogLuminance as EventReduction
else:
    spatial_reduce = False

if 'DVS128Gesture' in options.events:
    size=(128,128)
elif 'gesture' in options.events:
    size=(128*3,128)
elif 'DDD17' in options.events:
    size=(346,260)
elif 'shapes' in options.events:
    size = (240,180)

######################## GET DATA ########################

ev, x_input, y_input, time_data, coord_ts = getEvents(
    path_ev=options.events,
    size=size,
    spatial_reduce=spatial_reduce,
    time_reduce=time_reduce
)
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

Output_parameters = {
    'tau_m': 25,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV)
    'v_rest': -65.0,    # resting membrane potential (in mV) !
    'v_thresh': -20,    # spike threshold (in mV)
}

Output1 = sim.Population(
    x_input*y_input,
    sim.IF_cond_exp(**Output_parameters),
    label='Output layer 1'
)
Output1.record('spikes','v')

input2reaction_weights = []
subregions_Output1 = []
for X,Y in it.product(range(x_reaction), range(y_reaction)):

    idx = np.ravel_multi_index( (X,Y) , (x_reaction, y_reaction) )
    coordinates = []
    weights = []

    for x in range(int(reaction_downscale_factor_1D*X), int(reaction_downscale_factor_1D*(X+1))):
        if x < x_input:
            for y in range(int(reaction_downscale_factor_1D*Y), int(reaction_downscale_factor_1D*(Y+1))):
                if y < y_input:

                        A = np.ravel_multi_index( (x,y) , (x_input, y_input) )
                        coordinates.append(A)
                        weights.append( ( A, idx ) )

    input2reaction_weights += weights
    subregions_Output1.append(
        sim.PopulationView( Output1, np.array(coordinates) )
    )

print("\nSize of populations :\n> Input", Input.size, "with shape",(x_input, y_input), "\n> Reaction", Reaction.size, "with shape", (x_reaction, y_reaction), "\n> Output1", Output1.size, "with shape", (x_input, y_input), end="\n\n")


######################## CONNECTIONS ########################
print("Initiating connections...")

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
    synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_reaction, y_reaction, w_max=1)), #w_max=0.05)),
    receptor_type="inhibitory",
    label="Winner-Takes-All"
)

# output activation by input
input2output1 = sim.Projection(
    Input, Output1,
    connector=sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=1),
    receptor_type='excitatory',
    label='Connection from Input to Output layer 1'
)

# WTA on output layers
WTA_output1 = sim.Projection(
    Output1,Output1,
    connector=sim.AllToAllConnector(allow_self_connections=False),
    synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_input, y_input)), #w_max=0.05)),
    receptor_type="inhibitory",
    label="Winner-Takes-All on Output layer 1"
)

print("Network initialisation OK")


######################## RUN SIMULATION ########################

class LastSpikeRecorder(object):

    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self._spikes = np.zeros(self.population.size)

    def __call__(self, t):
        if t > 0:
            self._spikes = map(
                lambda x: x[-1].item() if len(x) > 0 else 0,
                self.population.get_data("spikes").segments[0].spiketrains
            )
            self._spikes = np.fromiter(self._spikes, dtype=float)
        return t+self.interval


class dynamicWeightAdaptation(object):
    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection

        # source & target
        self.source = projection.pre
        self.source_first_id = self.source.first_id
        self.target = projection.post

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
        firing_t = spikesReaction._spikes
        w = self.projection.get('weight', format='array', with_address=False)
        np.nan_to_num(w,copy=False)

        # Get weights
        null_w = np.where(w == 0)
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

        w[null_w] = 0

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

        return t+self.interval

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal


class dynamicThresholdAdaptation(object):
    def __init__(self, sampling_interval, activated_pop, modified_pop, activation, increase):
        self.interval = sampling_interval
        self.max_th = 0
        if type(modified_pop) == list:
            self.reset_th = modified_pop[0].get("v_reset")+1
        else :
            self.reset_th = modified_pop.get("v_reset")+1

        self._thresholds = []
        self.MP = modified_pop    # list of sub regions in modified population (list of PopulationView)
        self.AP = activated_pop
        # Reaction
        if activated_pop.label == "Reaction":
            self.APspikes = spikesReaction
        else:
            self.APspikes = spikesOutput1

        # if activation = "excitation", threshold are lower // if activation = "inhibition", threshold are increased$
        assert activation in ["excitation", "inhibition"]
        self.activation = activation
        if activation == "excitation":
            self.increase = increase
        elif activation == "inhibition":
            self.increase = -increase


    def update_threshold(self, pop, firing_t, t, th):
        if type(self.MP) == list:
            firing_t = [firing_t]*pop.size
        firing_t = np.array(firing_t)

        if self.activation == "excitation":
            global_th = np.where(
                firing_t > t - 1,
                th - self.increase,

                np.where(
                    firing_t < t-100,
                    th + self.increase,
                    th
                )
            )
            global_th = np.where(
                global_th > self.max_th,
                self.max_th,
                np.where(
                    global_th < self.reset_th,
                    self.reset_th,
                    global_th
                )
            )

        elif self.activation == "inhibition":
            global_th = np.where(
                firing_t > t - 1,
                self.max_th,
                np.where(
                    firing_t < t-100,
                    th + self.increase,
                    th
                )
            )
            global_th = np.where(
                global_th < self.reset_th,
                self.reset_th,
                global_th
            )

        return global_th


    def __call__(self,t):
        if t > 0:
            firing_t = self.APspikes._spikes
            i=0
            global_th = []
            if type(self.MP) == list:
                for subr in self.MP:

                    # get threshold value of subregion
                    native_parameter_space = subr._get_parameters('V_th')
                    parameter_space = subr.celltype.reverse_translate(native_parameter_space)
                    parameter_space.evaluate()
                    parameters = dict(parameter_space.items())
                    th = parameters['v_thresh']

                    Th = self.update_threshold(subr, firing_t[i], t, th)
                    global_th += list(set(Th))


                    # subr.set(v_thresh=Th)
                    parameter_space = ParameterSpace({'v_thresh': Th},
                                                    subr.celltype.get_schema(),
                                                    (subr.size,),
                                                    subr.celltype.__class__)
                    parameter_space = subr.celltype.translate(parameter_space)
                    subr._set_parameters(parameter_space)

                    i+=1
            
            else :
                # th = self.MP.get('v_thresh', gather=True)
                native_parameter_space = self.MP._get_parameters('V_th')
                parameter_space = self.MP.celltype.reverse_translate(native_parameter_space)
                parameter_space.evaluate()
                parameters = dict(parameter_space.items())
                th = parameters['v_thresh']

                global_th = self.update_threshold(self.MP, firing_t, t, th)
                self.MP.set(v_thresh=global_th)
            self._thresholds.append( global_th )
        return t + self.interval

    def get_thresholds(self):
        signal = neo.AnalogSignal(self._thresholds, units='nA', sampling_period=self.interval * ms, name="threshold")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._thresholds[0])))
        return signal


class visualiseTime(object):
    def __init__(self, sampling_interval):
        self.interval = sampling_interval

    def __call__(self, t):
        print(t)
        return t + self.interval


# visualise_time = visualiseTime(sampling_interval=1.0)

spikesReaction = LastSpikeRecorder(sampling_interval=1.0, pop=Reaction)
spikesOutput1 = LastSpikeRecorder(sampling_interval=1.0, pop=Output1)

weightRuleInput2reaction = dynamicWeightAdaptation(sampling_interval=1.0, projection=input2reaction)
thresholdRuleOutput1 = dynamicThresholdAdaptation(sampling_interval=1.0, activated_pop=Reaction, modified_pop=subregions_Output1, activation='excitation', increase=10)

callbacks = [spikesReaction, weightRuleInput2reaction, thresholdRuleOutput1]

print("\nStart simulation ...")
start = d.now()
sim.run(time_data, callbacks=callbacks)
print("Simulation done in", d.now() - start ,"\n")

######################## SIMULATION RESULTS ########################

Input_data = Input.get_data().segments[0]
Reaction_data = Reaction.get_data().segments[0]
Output1_data = Output1.get_data().segments[0]
# weights = weightRuleInput2reaction.get_weights()

w = np.array(weightRuleInput2reaction._weights)
s = w.shape
w = w[w > 0].reshape(s[0], -1).T

th = np.array(thresholdRuleOutput1._thresholds).T


if options.plot_figure :
    
    figure1_filename = normalized_filename("Results", "Adaptation rules", "png", options.simulator)
    figure2_filename = normalized_filename("Results", "Saliency detection", "png", options.simulator)

    fig1, (plt_weights, plt_thresholds) = plt.subplots(nrows=2, ncols=1)
    # weights
    pw = plt_weights.pcolormesh(w, cmap='viridis')
    fig1.colorbar(pw, ax=plt_weights)
    plt_weights.set_title('Weight adaptation rule - Input to ROI detector')
    # thresholds
    pt = plt_thresholds.pcolormesh(th, cmap='RdBu_r')
    fig1.colorbar(pt, ax=plt_thresholds)
    plt_thresholds.set_title('Threshold adaptation rule - Output1')

    plt.figtext(0.01, 0.01, "Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events, fontsize=6, verticalalignment='bottom')
    plt.savefig(figure1_filename)

    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, xlabel="ROI detector spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),
        # raster plot of the Output1 neurons spike times
        Panel(Output1_data.spiketrains, xlabel="Output1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),

        # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], xlabel="Membrane potential (mV)\nROI detector layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # evolution of the synaptic weights with time
        # Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Intermediate Output Weight",
        #         legend=False, xlim=(0, time_data)),
        
        # weight adaptation rule - Input to ROI detector
        # Panel(w, xlabel='Weight adaptation rule - Input to ROI detector', yticks=True, xlim=(0, time_data)),
        # # threshold adaptation rule -
        # Panel(th, xlabel='Threshold adaptation rule - Output1',yticks=True, xlim=(0, time_data)),

        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ).save(figure2_filename)
    print("Figures correctly saved as", figure1_filename, "and", figure2_filename)
    # plt.show()


######################## SAVE DATA ########################

def newEvent(x,y,p,t, coord_t):
    if coord_t == 2:
        return [x,y,t,p]
    elif coord_t == 3:
        return [x,y,p,t]
    elif coord_t == 0:
        return [t,x,y,p]

def spikes2ev(spikes, width, height, coord_t, polarity=1, time_reduce=False, time_factor=0.001):
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

    events = events[events[:,coord_t].argsort()]
    if time_reduce:
        events[:,coord_t] *= time_factor
    return events

if options.save :
    Reaction_filename = normalized_filename("Results", "Saliency detection - ROI detector", "npy", options.simulator)
    np.save(
        Reaction_filename,
        spikes2ev(Reaction_data.spiketrains, x_reaction, y_reaction, coord_ts, time_reduce=time_reduce)
    )
    print("Output data from ROI detector layer correctly saved as", Reaction_filename)

    Output1_filename = normalized_filename("Results", "Saliency detection - Output", "npy", options.simulator)
    np.save(
        Output1_filename,
        spikes2ev(Output1_data.spiketrains, x_input, y_input, coord_ts, time_reduce=time_reduce)
    )
    print("Output data from Output layer correctly saved as", Output1_filename)

sim.end()