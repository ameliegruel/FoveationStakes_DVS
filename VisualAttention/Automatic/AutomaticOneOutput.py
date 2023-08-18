from os import walk, path, makedirs

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
        self.global_spikes = [[] for _ in range(self.population.size)]

    def __call__(self, t):
        if t > 0:
            self._spikes = map(
                lambda x: x[-1].item() if len(x) > 0 else 0,
                self.population.get_data("spikes", clear=True).segments[0].spiketrains
            )
            self._spikes = np.fromiter(self._spikes, dtype=float)

            assert len(self._spikes) == len(self.global_spikes)
            if len(np.unique(self._spikes)) > 1:
                idx = np.where(self._spikes != 0)[0]
                for n in idx:
                    self.global_spikes[n].append(self._spikes[n])

        return t+self.interval
    
    def get_spikes(self):
        return self.global_spikes


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
        firing_t = self.spikesReaction._spikes
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
    def __init__(self, sampling_interval, activated_pop, modified_pop, activation, increase, spikesReaction):
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
        self.APspikes = spikesReaction

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


# FUNCTIONS

def ExponentialConnectionWeight(x,y, w_max=50.0):
    size=x*y
    # get coordinates of neurons in matrix
    coordinates = np.repeat(
        [np.transpose(
            np.unravel_index(
                np.arange(size), (x,y)
        ))],
        size, axis=0)
    # get coordinates of neurons in the diagonal
    diag_indices = np.transpose(
        np.unravel_index(
            np.arange(size), (x,y)
    )).reshape((size, 1, 2))
    # get distances between neurons in diagonal and in matrix
    distances = np.array(
        [ [np.linalg.norm(arr) for arr in lines]   for lines in abs(coordinates - diag_indices) ]
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
        events_downscale_factor_1D = 4
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
    
    sim_length = args.time
    if max(ev[:,coord_ts]) < sim_length :
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


    ######################## CONNECTIONS ########################
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
        synapse_type=sim.StaticSynapse(weight=ExponentialConnectionWeight(x_reaction, y_reaction, w_max=1)),
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
        synapse_type=sim.StaticSynapse(weight=ExponentialConnectionWeight(x_input, y_input)),
        receptor_type="inhibitory",
        label="Winner-Takes-All on Output layer 1"
    )

    return Input, Reaction, Output1, subregions_Output1, input2reaction, x_reaction, y_reaction



def run_SNN(
    events,
    coord_t,
    Input,
    Reaction,
    Output1,
    subregions_Output1,
    input2reaction,
    x_reaction, y_reaction,
    sim_length=1e3,
    time_reduce=True,
    time_factor=1000,
    plot_figure=False,
    sample_name=None
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

    spikesInput = LastSpikeRecorder(sampling_interval=1.0, pop=Input)
    spikesReaction = LastSpikeRecorder(sampling_interval=1.0, pop=Reaction)
    spikesOutput = LastSpikeRecorder(sampling_interval=1.0, pop=Output1)

    weightRuleInput2reaction = dynamicWeightAdaptation(sampling_interval=1.0, projection=input2reaction, spikesReaction=spikesReaction)
    thresholdRuleOutput1 = dynamicThresholdAdaptation(sampling_interval=1.0, activated_pop=Reaction, modified_pop=subregions_Output1, activation='excitation', increase=10, spikesReaction=spikesReaction)
    callbacks = [spikesInput, spikesReaction, spikesOutput, weightRuleInput2reaction, thresholdRuleOutput1]

    sim.run(sim_length, callbacks = callbacks)

    spikes = spikesOutput.get_spikes()

    events = spikes2ev(spikes, x_input, y_input, coord_t)
    
    ###########################################################################################################################################

    events = np.vstack((np.zeros((0,4)), events))  # handles case where no spikes produced by simulation

    if time_reduce:
        events[:,coord_t] *= time_factor

    if plot_figure:
        assert sample_name != None

        ## hyperparameters
        filename = path.join(SNN_repertory,'Figures','adaptation_rules_'+sample_name.replace('.npy','.png').replace('.npz','.png'))
        w = np.array(weightRuleInput2reaction._weights)
        s = w.shape
        w = w[w > 0].reshape(s[0], -1).T
        th = np.array(thresholdRuleOutput1._thresholds).T

        # weights
        fig1, (plt_weights, plt_thresholds) = plt.subplots(nrows=2, ncols=1)
        pw = plt_weights.pcolormesh(w, cmap='viridis')
        fig1.colorbar(pw, ax=plt_weights)
        plt_weights.set_title('Weight adaptation rule - Input to ROI detector')
        # thresholds
        pt = plt_thresholds.pcolormesh(th, cmap='RdBu_r')
        fig1.colorbar(pt, ax=plt_thresholds)
        plt_thresholds.set_title('Threshold adaptation rule - Output')

        plt.figtext(0.01, 0.01, "Simulated with nest\nInput events from "+sample_name, fontsize=6, verticalalignment='bottom')
        plt.title('Adaptation rules')
        plt.savefig(filename)

        ## spikes 
        filename = path.join(SNN_repertory,'Figures','spikes_'+sample_name.replace('.npy','.png').replace('.npz','.png'))
        Input_data = spikesInput.get_spikes()
        Reaction_data = spikesReaction.get_spikes()
        figure_info = {
            "plot1": {
                "data": Input_data, "xlabel": "Input spikes", "yticks":True, "ylim":(0, Input.size)
            },
            "plot2": {
                "data": Reaction_data, "xlabel": "Reaction spikes", "yticks":True, "ylim":(0, Reaction.size)
            },
            "plot3": {
                "data": spikes, "xlabel": "Output spikes", "yticks":True, "ylim":(0, Output1.size)
            },
            "figure": {
                "title": 'Spikes', "annotations":"Simulated with nest\nInput events from "+sample_name, "save":True, "saveas": filename
            }
        }
        Figure(figure_info)
        
    return events


# main
import pyNN.nest as sim
sim.setup(timestep=1)

parser = argparse.ArgumentParser(description="Automatically reduce data")
parser.add_argument("--dataset", "-da", help="Dataset repertory", metavar="D", type=str)
parser.add_argument("--method", "-m", help="Reduction method (between 'funelling','eventcount', 'cubic', 'linear','none')", metavar="m", type=str)
parser.add_argument("--time", "-t", help="Length of simulation (in ms) - 1s by default", metavar="t", type=float, default=1e3)
parser.add_argument("--plot-figure", "-p", action='store_true', help="Plot output spikes and hyperparameters evolution")
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

info = 'original dataset;'


dr = args.dataset
print("\n>>",dr,'\n')
if not dr.endswith('/'):
    dr += '/'
date_time = d.now().strftime("%Y%m%d-%H%M%S")

if 'DVS128Gesture' in dr and 'combinations_duo' in dr:
    original_repertory = dr.split('/')[-2]
    info += original_repertory+';\n'
    SNN_repertory = dr.replace(original_repertory+'/','')+"saliencyFilter_oneOutput_"+args.method+"_"+date_time+"/"
    size=(128*2,128)
    middle = size[1] / events_downscale_factor_1D
    time_reduce = True
elif 'DVS128Gesture' in dr:
    original_repertory = '/'.join(dr.split('/')[-4:-1])
    info += original_repertory+';\n'
    SNN_repertory = dr.replace(original_repertory+'/','')+"saliencyFilter_oneOutput_"+args.method+"_"+date_time+"/"
    size=(128,128)
    middle = size[1]
    time_reduce = True
elif 'DDD17' in dr:
    original_repertory = dr+"test/"
    SNN_repertory = dr+"ROI_data_"+args.method+"_div"+str(events_downscale_factor_1D)+"_"+date_time+"/"
    size=(346,260)
    time_reduce = True

info += '\nsample name;time;nb events;nb events left;nb events right;\n'

if args.plot_figure :
    import matplotlib.pyplot as plt
    from plots import Figure
    makedirs(path.join(SNN_repertory,'Figures'))

init = True

for (rep_path, _, files) in walk(dr):
    repertory=rep_path.replace(original_repertory, "").replace('//','/')
    if len(files) > 0:
        
        for event_file in files :
            print(repertory, event_file, end=" ")

            if (event_file.endswith('npz') or event_file.endswith('npy')) and not path.exists(path.join( SNN_repertory, repertory, event_file.replace("npz","npy")) ):

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
                    if init == True:
                        # sim.end()
                        # sim.setup(timestep=1)
                        Input, Reaction, Output1, subregions_Output1, input2reaction, x_reaction, y_reaction = init_SNN(sim, original_events,x_input,y_input)
                        print("Network has been correctly initialised", end=" ")
                        init = False
                    
                    else: 
                        sim.reset()


                    # SNN
                    start = d.now()
                    SNN_events = run_SNN(
                        original_events,
                        coord_ts,
                        Input, 
                        Reaction,
                        Output1,
                        subregions_Output1,
                        input2reaction, 
                        x_reaction, y_reaction,
                        sim_length=sim_length,
                        time_reduce=time_reduce,
                        time_factor=1000,
                        plot_figure = args.plot_figure,
                        sample_name=event_file
                    )
                    time = d.now() - start
                    print(time,  end=" ")
                info += event_file+';'+str(time)+';'+str(len(SNN_events))+';'+str(len(SNN_events[SNN_events[:,0] < middle]))+';'+str(len(SNN_events[SNN_events[:,0] >= middle]))+';\n'
                
                saveEvents(
                    SNN_events,
                    event_file,
                    path.join( SNN_repertory)
                )
                print("saved")
                f = open(path.join(SNN_repertory, "info.csv"),"w")
                f.write(info)
                f.close()

            
            else : 
                print()

# f = open(path.join(SNN_repertory, "info.csv"),"w")
# f.write(info)
# f.close()
