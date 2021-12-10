#!/bin/python3

"""
Spiking neural network for saliency detection in event-camera data
Author : Amélie Gruel - Université Côte d'Azur, CNRS/i3S, France - amelie.gruel@i3s.unice.fr
Run as : $ python3 getSaliency.py nest
Use of DDD17 (DVS Driving Dataset 2017) annotated by [Alonso, 2017]
"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
from pyNN.space import Grid2D
from quantities.units.radiation import R
from events2spikes import ev2spikes
from reduceEvents import event_count, reduce
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import sys
from quantities import ms
from tqdm import tqdm
import neo

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
    # return np.maximum( np.minimum( np.log( distances ) / (0.1*size) , w_max), 0)
    return np.maximum( np.minimum( np.exp( distances ) * (1/size) , w_max), 0)




################################################### MAIN #################################################################

# Configure simulator
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--events","Get input file with inputs"),
                             ("--inter-layer", "Use intermediate layer between input and reaction", {"action": "store_true"}),
                             ("--reduce", "Spatailly reduce input data", {"action": "store_true"}),
                             ("--unit", "Temporally adapt input data", {"default": "milli"}),
                             ("--gaussian_feedback", "Implement Gaussian feedback between Input and intermediate events layer", {"action": "store_true"}),
                             ("--classif", "Simulation with the classifications layers", {"action": "store_true"}),
                             ("--save-all", "Save the data output by all layers as pkl files", {"action":"store_true"}),
                             ("--save-ROI", "Save the data output by ROI selection layer as pkl ", {"action":"store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}))

if options.debug:
    init_logging(None, debug=True)

if options.gaussian_feedback:
    options.inter_layer = True

if options.classif :
    nb_classifiers = 2

if sim == "nest":
    import pyNN.nest as sim
elif sim == "spinnaker":
    import pyNN.spiNNaker as sim

dt = 1/80000
sim.setup(timestep=0.01)

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
    print("Error: The input file has to be of format .npy")
    sys.exit()

if options.reduce :
    ev = event_count(ev, coord_t=coord_ts, div=events_downscale_factor_1D)
    events_downscale_factor_1D = 1

max_time = int(np.max(ev[:,coord_ts]))
print("\nTime length of the data :",max_time)
start_time = int(input("Simulation start time : "))
stop_time = int(input("Simulation stop time : "))

try :
    ev = ev[np.logical_and( ev[:,coord_ts]>start_time , ev[:,coord_ts]<stop_time )]
    ev[:,2] -= start_time
    time_data = int(np.max(ev[:,coord_ts]))
    ev, x_input, y_input = ev2spikes( ev, coord_t=coord_ts )
    print("Simulation will be run on", len(ev), "events\n")

except ValueError:
    print("Error: The start and stop time you defined for the simulation are not coherent with the data.")
    sys.exit(0)

x_event = int(x_input/events_downscale_factor_1D)
y_event = int(y_input/events_downscale_factor_1D)
x_reaction = int(x_event/reaction_downscale_factor_1D)
y_reaction = int(y_event/reaction_downscale_factor_1D)

################################################ NETWORK ################################################


######################## POPULATIONS ########################
print("Initiating populations...")

Input = sim.Population(
    x_input*y_input,
    sim.SpikeSourceArray(spike_times=ev),
    label="Input")
Input.record("spikes")

ROI_parameters = {
    'tau_m': 25,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV)
    'v_rest': -65.0,    # resting membrane potential (in mV) !
    'v_thresh': -40,    # spike threshold (in mV)
}
ROI = sim.Population(
    x_input*y_input,
    sim.IF_cond_exp(**ROI_parameters),
    label="Region of interest")
ROI.record(("spikes","v"))

inhib_parameters = {
    'tau_m': 15,
    'tau_refrac': 5,
    'v_reset': -65,
    'v_rest': -65,
    'v_thresh': -50
}

if options.classif :
    classifiers = []
    inhibitors_classifier = []

    for layer in range(nb_classifiers):
        classifiers.append( sim.Population(
            x_input*y_input,
            sim.IF_cond_exp(**ROI_parameters),
            label="Classifier "+str(layer+1))
        )
        classifiers[-1].record(("spikes","v"))

    for neuron in range(nb_classifiers-1):
        inhibitors_classifier.append( sim.Population(
            1,
            sim.IF_cond_exp(**inhib_parameters),
            label="Inhibitor "+str(neuron+1)+" of classifier "+str(neuron+2)
        ))
        inhibitors_classifier[-1].record(("spikes","v"))



Reaction_parameters = {
    'tau_m': 2.5,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV)
    'v_rest': -65.0,    # resting membrane potential (in mV) !
    'v_thresh': -25,    # spike threshold (in mV)
}

if options.inter_layer :
    Events_parameters = {
        'tau_m': 25,      # membrane time constant (in ms)
        'tau_refrac': 0.1,  # duration of refractory period (in ms)
        'v_reset': -65.0,   # reset potential after a spike (in mV)
        'v_rest': -65.0,    # resting membrane potential (in mV) !
        'v_thresh': -15,    # spike threshold (in mV)
    }

Reaction = sim.Population(
    x_reaction * y_reaction,
    sim.IF_cond_exp(**Reaction_parameters),
    label="Reaction",
    structure=Grid2D(dx=reaction_downscale_factor_1D, dy=reaction_downscale_factor_1D, z=50, aspect_ratio=x_event/y_event)
)

subRegionsInput = []

if options.inter_layer :
    Events = sim.Population(
        x_event*y_event,
        sim.IF_cond_exp(**Events_parameters),
        label="Events",
        structure=Grid2D(dx=1, dy=1, z=0, aspect_ratio=x_event/y_event)
    )

    for X in range(x_event):
        for Y in range(y_event):
            subRegionsInput.append(
                sim.PopulationView(Input, np.array(
                    [np.ravel_multi_index( (x,y) , (x_input , y_input) )  for x in range(events_downscale_factor_1D*X, events_downscale_factor_1D*(X+1)) for y in range(events_downscale_factor_1D*Y, events_downscale_factor_1D*(Y+1)) ]
                ))
            )
    subRegionsEvents = []

subRegionsROI = []
if options.classif:
    subRegionsClassifiers = [[]]*nb_classifiers
for X in range(x_reaction):
    for Y in range(y_reaction):

        if options.inter_layer :
            region_coordonates = np.array( [np.ravel_multi_index( (x,y) , (x_event, y_event) )  for x in range(reaction_downscale_factor_1D*X, reaction_downscale_factor_1D*(X+1)) for y in range(reaction_downscale_factor_1D*Y, reaction_downscale_factor_1D*(Y+1)) ] )
            subRegionsEvents.append(
                sim.PopulationView(Events, region_coordonates)
            )

        else :
            region_coordonates = np.array([np.ravel_multi_index( (x,y) , (x_input,y_input) ) for x in range(events_downscale_factor_1D*reaction_downscale_factor_1D*X, events_downscale_factor_1D*reaction_downscale_factor_1D*(X+1)) for y in range(events_downscale_factor_1D*reaction_downscale_factor_1D*Y, events_downscale_factor_1D*reaction_downscale_factor_1D*(Y+1))])
            subRegionsInput.append(
                sim.PopulationView(Input, region_coordonates)
            )

        subRegionsROI.append(
            sim.PopulationView(ROI, region_coordonates)
        )

        if options.classif :
            for n in range(nb_classifiers):
                subRegionsClassifiers[n].append(
                    sim.PopulationView(classifiers[n], region_coordonates)
                )


Reaction.record(("spikes","v"))
if options.inter_layer :
    Events.record(("spikes","v"))

if options.inter_layer :
    print("\nSize of populations :\n> Input", Input.size, "with shape",(x_input, y_input), "\n> Events", Events.size, "with shape", (x_event,y_event),"\n> Reaction", Reaction.size, "with shape", (x_reaction, y_reaction), "\n> ROI",ROI.size, "with shape", (x_input, y_input),end="\n\n")
else :
    print("\nSize of populations :\n> Input", Input.size, "with shape",(x_input, y_input), "\n> Reaction", Reaction.size, "with shape", (x_reaction, y_reaction), "\n> ROI",ROI.size, "with shape", (x_input, y_input),end="\n\n")


######################## CONNECTIONS ########################
print("Initiating connections...")

# connection between input and ROI visualisation
input2ROI = sim.Projection(
    Input, ROI,
    connector=sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=0.8),
    receptor_type="excitatory",
    label="Connection input to ROI"
)

# connection between input and intermediate events layer
if options.inter_layer :
    input2events = []
    for n in tqdm(range(Events.size)):
        input2events.append( sim.Projection(
            subRegionsInput[n], sim.PopulationView(Events, [n]),
            connector=sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=1),
            receptor_type="excitatory",
            label="Connection input region to events neuron "+str(n)
        ))
    events2reaction = []

input2reaction = []
reaction2ROI = []

w_Reaction2ROI=GaussianConnectionWeight(x_reaction, y_reaction, w_max=0.1)
for n in tqdm(range(Reaction.size)):
    reaction_neuron = sim.PopulationView(Reaction, [n])

    # connection between intermediate events layer and ROI detection
    if options.inter_layer :
        events2reaction.append( sim.Projection(
            subRegionsEvents[n], reaction_neuron,
            connector=sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=0.4),
            receptor_type="excitatory",
            label="Connection events region to reaction neuron "+str(n)
        ))

    # connection between input and ROI detection
    input2reaction.append( sim.Projection(
        subRegionsInput[n], reaction_neuron,
        connector = sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=1),
        receptor_type="excitatory",
        label="Connection events region to reaction neuron "+str(n)
    ))
    # connection between ROI detection and ROI selection
    # reaction2ROI.append( sim.Projection(
    #     reaction_neuron, subRegionsROI[n],
    #     connector=sim.AllToAllConnector(),
    #     synapse_type=sim.StaticSynapse(weight=0.1),
    #     receptor_type="excitatory",
    #     label="One to one excitatory connection reaction neuron "+str(n)+" to corresponding region in ROI layer"
    # ))
    for e in filter(lambda x: x != n, range(Reaction.size)):
        reaction2ROI.append( sim.Projection(
            reaction_neuron, subRegionsROI[e],
            connector=sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=w_Reaction2ROI[n,e]),  #0.05),
            receptor_type="inhibitory",
            label="Inhibitory connection reaction neuron "+str(n)+" to corresponding region in ROI layer"
        ))


    ### feedback
    if options.gaussian_feedback:

        feedback = {
            "big excitation": [],
            "small excitation": [],
            "inhibition" : []
        }
        f_neurons=[n]

        feedback['big excitation'].append(
            sim.Projection(
                reaction_neuron, subRegionsEvents[n],
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=0.1),
                receptor_type="excitatory",
                label="Feedback connection reaction to input - strongly excitatory"
            )
        )

        X,Y = np.unravel_index(n, (x_reaction, y_reaction))
        for x in range(X-1, X+2):
            for y in range(Y-1, Y+2):
                if x!=X or y!=Y:

                    try :
                        n_small_excitation = np.ravel_multi_index((x,y),(x_reaction,y_reaction))
                        f_neurons.append(n_small_excitation)
                        feedback['small excitation'].append(
                            sim.Projection(
                                reaction_neuron, subRegionsEvents[n_small_excitation],
                                connector=sim.AllToAllConnector(),
                                synapse_type=sim.StaticSynapse(weight=0.05),
                                receptor_type="excitatory",
                                label="Feedback connection reaction to input - lightly excitatory"
                            )
                        )
                    except ValueError:
                        pass

        for e in filter(lambda x: x not in f_neurons, range(Reaction.size)):
            feedback["inhibition"].append(
                sim.Projection(
                    reaction_neuron, subRegionsEvents[e],
                    connector=sim.AllToAllConnector(),
                    synapse_type=sim.StaticSynapse(weight=0.01),
                    receptor_type="inhibitory",
                    label="Feedback connection reaction to input - strongly inhibitory"
                )
            )

"""
else :
    input2events = sim.Projection(
        Input, Events,
        connector=sim.OneToOneConnector(),
        synapse_type=sim.StaticSynapse(weight=1),
        receptor_type="excitatory",
        label="Connection input to events"
    )

    events2reaction = sim.Projection(
        Events, Reaction,
        connector=sim.OneToOneConnector(),
        synapse_type=sim.StaticSynapse(weight=1),
        receptor_type="excitatory",
        label="Connection events to reaction layer"
    )
"""

# lateral inhibition on ROI detection
WTA = sim.Projection(
    Reaction, Reaction,
    connector=sim.AllToAllConnector(allow_self_connections=False),
    synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_reaction, y_reaction, w_max=0.05)),
    receptor_type="inhibitory",
    label="Winner-Takes-All"
)

if options.classif:
    roi2classifiers = []
    roi2inhibitors = []

    inhibitors2classifiers = []
    classifiers2inhibitors = []

    lateral_inhibition_classifiers = []
    WTA_classifier = []

    for n in tqdm(range(nb_classifiers)) :
        layer = classifiers[n]

        # connection between ROI visualisation and classifier
        roi2classifiers.append( sim.Projection(
            ROI, layer,
            connector=sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=0.7),
            receptor_type="excitatory",
            label="Connection from ROI visualisation layer to classifier "+str(n+1)
        ))

        # lateral inhibition on classifier
        WTA_classifier.append( sim.Projection(
            layer, layer,
            connector=sim.AllToAllConnector(allow_self_connections=False),
            synapse_type=sim.StaticSynapse(weight=GaussianConnectionWeight(x_input, y_input)),
            receptor_type="inhibitory",
            label="Self inhibition with Winner-Takes-All"
        ))

        if n > 0:
            inhib_neuron = inhibitors_classifier[n-1]

            # connection between ROI visualisation and inhibitor neuron
            roi2inhibitors.append( sim.Projection(
                ROI, inhib_neuron,
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=0.7),
                receptor_type="excitatory",
                label="Connection from ROI visualisation layer to inhibitor neuron "+str(n)
            ))

            # connections between classifier and inhibitor neuron
            classifiers2inhibitors.append (sim.Projection(
                classifiers[n-1], inhib_neuron,
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=0.7),
                receptor_type="inhibitory",
                label="Inhibitory connection from classifier "+str(n)+" to inhibitor "+str(n)
            ))

            inhibitors2classifiers.append (sim.Projection(
                inhib_neuron, layer,
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=0.7),
                receptor_type="inhibitory",
                label="Inhibitory connection from inhibitor "+str(n)+" to classifier "+str(n+1)
            ))

            # connection between classifier and previous classifiers
            for i in range(n):
                lateral_inhibition_classifiers.append( sim.Projection(
                    classifiers[i], layer,
                    connector=sim.OneToOneConnector(),
                    synapse_type=sim.StaticSynapse(weight=10),
                    receptor_type="inhibitory",
                    label="Inhibitory connection from classifier "+str(i+1)+" to classifier "+str(n+1)
                ))


print("Network initialisation OK")


######################## RUN SIMULATION ########################

class LastSpikeRecorder(object):
    
    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self._spikes = np.zeros(self.population.size)
    
    def __call__(self, t):
        try : 
            self._spikes = np.array([t[-1] if t != [] else 0 for t in self.population.get_data().segments[0].spiketrains])
        except IndexError:
            pass
        return t+self.interval


class dynamicWeightAdaptation(object):
    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection
        self.source = projection[0].pre
        self.target = projection[0].post
        self.increase = 0.01
        self.default_w = 0.7
        self._weights = []
    
    def __call__(self, t):
        firing_t = spikesReaction._spikes
        if type(self.projection) != list:
            w = self.projection.get('weight', format='list', with_address=False)
            w = np.where(
                np.logical_and(
                    firing_t > t-1,
                    firing_t != 0
                ),
                w + self.increase,
                np.where(
                    firing_t < t-100,
                    max(
                        w - self.increase,
                        self.default_w
                    ),
                    w
                )
            )
            self.projection.set(weight=w)
            self._weights.append(w)

        elif type(self.projection) == list:
            global_w = []
            for n in range(len(self.projection)):
                w = self.projection[n].get('weight', format='array', with_address=False)
                if firing_t[n] > t-1 and firing_t[n] != 0:
                    w = w + self.increase
                elif firing_t[n] < t-100: 
                    w=max(w - self.increase , self.default_w)
                self.projection[n].set(weight=w)
                global_w.append(np.unique(w))
            self._weights.append(global_w)

        return t+self.interval
    
    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal


class dynamicThresholdAdaptation(object):
    def __init__(self, sampling_interval, activated_pop, modified_pop):
        self.interval = sampling_interval
        self.increase = 0.1
        self.default_th = -40
        self._thresholds = []
        self.AP = activated_pop
        self.MP = modified_pop    # list of sub regions in modified population (list of PopulationView)
        
    def __call__(self,t):
        firing_t = spikesReaction._spikes
        i=0
        for subr in self.MP:
            th = subr.get('v_thresh', gather=True)
            if firing_t[i] > t-1:
                th = th - self.increase
            else:
                th = th + self.increase
            subr.set(v_thresh=th)
            self._thresholds.append(th)
            i+=1
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

visualise_time = visualiseTime(sampling_interval=10.0)
spikesReaction = LastSpikeRecorder(sampling_interval=1.0, pop=Reaction)
weightRuleInput2reaction = dynamicWeightAdaptation(sampling_interval=1.0, projection=input2reaction)
callbacks = [visualise_time, spikesReaction, weightRuleInput2reaction]
if options.classif:
    callbacks.append( dynamicThresholdAdaptation(sampling_interval=1.0, activated_pop=Reaction, modified_pop=subRegionsClassifiers[0]) )

print("\nStart simulation ...")
sim.run(time_data, callbacks=callbacks)
print("Simulation done\n")


######################## SIMULATION RESULTS ########################

Input_data = Input.get_data().segments[0]
Reaction_data = Reaction.get_data().segments[0]
ROI_data = ROI.get_data().segments[0]
weights = weightRuleInput2reaction.get_weights()
if options.classif :
    classifiers_data = [c.get_data().segments[0] for c in classifiers]
    thresholds = callbacks[-1].get_thresholds()

figure_filename = normalized_filename("Results", "Saliency detection", "png", options.simulator)

if options.plot_figure and not options.inter_layer and not options.classif:
    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, xlabel="ROI detector spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),
        # raster plot of the Reaction neurons spike times
        Panel(ROI_data.spiketrains, xlabel="ROI visualisation spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, ROI.size)),

        # # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI detector layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # # membrane potential of the Reaction neurons
        Panel(ROI_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI visualisation layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # evolution of the synaptic weights with time
        Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Intermediate Output Weight",
                legend=False, xlim=(0, time_data)),
            
        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ) #.save(figure_filename)
    # print("Figure correctly saved as", figure_filename)
    plt.show()

elif options.plot_figure and not options.inter_layer and options.classif:
    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, xlabel="ROI detector spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),
        # raster plot of the Reaction neurons spike times
        Panel(ROI_data.spiketrains, xlabel="ROI visualisation spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, ROI.size)),
        # membrane potential of the Classifier 1
        Panel(classifiers_data[0].spiketrains, xlabel="Classification layer 1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, classifiers[0].size)),
        # membrane potential of the Classifier 2
        Panel(classifiers_data[1].spiketrains, xlabel="Classification layer 2 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, classifiers[1].size)),

        # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI detector layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # membrane potential of the Reaction neurons
        Panel(ROI_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI visualisation layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False, xlabel="Time (ms)", xticks=True),
        # evolution of the synaptic weights with time
        Panel(weights, yticks=True, ylabel="Reaction2ROI Weight",
                legend=False, xlim=(0, time_data)),
        # evolution of the thresholds with time
        Panel(thresholds, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Classifier 1 threshold",
                legend=False, xlim=(0, time_data)),
        
        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ) #.save(figure_filename)
    # print("Figure correctly saved as", figure_filename)
    plt.show()

elif options.plot_figure and options.inter_layer :
    Events_data = Events.get_data().segments[0]
    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the event inputs spike times
        Panel(Events_data.spiketrains, xlabel="Events spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Events.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, xlabel="ROI detector spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Reaction.size)),
        # raster plot of the Reaction neurons spike times
        Panel(ROI_data.spiketrains, xlabel="ROI visualisation spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, ROI.size)),

        # membrane potential of the Events neurons
        Panel(Events_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nEvents layer", xlim=(0, time_data), linewidth=0.2, legend=False, yticks=True),
        # # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI detector layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # # membrane potential of the Reaction neurons
        Panel(ROI_data.filter(name='v')[0], ylabel="Membrane potential (mV)\nROI visualisation layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False, xlabel="Time (ms)", xticks=True),

        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ).save(figure_filename)
    print("Figure correctly saved as", figure_filename)



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

    events = events[events[:,coord_t].argsort()]
    return events

save = input("Save ROI ? (y/n) ")

if options.save_all:
    if options.inter_layer:
        Events_filename = normalized_filename("Results", "Saliency detection - Events", "npy", options.simulator)
        np.save(
            Events_filename,
            spikes2ev(Events_data.spiketrains, x_event, y_event, coord_ts)
        )
        # Events.write_data(Events_filename)
        print("Output data from Events layer correctly saved as", Events_filename)

    Reaction_filename = normalized_filename("Results", "Saliency detection - ROI detector", "npy", options.simulator)
    np.save(
        Reaction_filename,
        spikes2ev(Reaction_data.spiketrains, x_reaction, y_reaction, coord_ts)
    )
    print("Output data from ROI detector layer correctly saved as", Reaction_filename)

if options.save_all or options.save_ROI or save=="y":

    ROI_filename = normalized_filename("Results", "Saliency detection - ROI visualisation", "npy", options.simulator)
    np.save(
        ROI_filename,
        spikes2ev(ROI_data.spiketrains, x_input, y_input, coord_ts)
    )
    print("Output data from ROI selection layer correctly saved as", ROI_filename)

    if options.classif:
        for c in range(nb_classifiers):
            classif_filename = normalized_filename("Results", "Saliency detection - classifier "+str(c+1), "npy", options.simulator)
            np.save(
                classif_filename,
                spikes2ev(classifiers_data[c], x_input, y_input, coord_ts)
            )
            # classifiers[c].write_data(classif_filename)
            print("Output data from classifier layer "+str(c+1)+" correctly saved as", classif_filename)

sim.end()
