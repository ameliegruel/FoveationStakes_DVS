#!/bin/python3

"""
Spiking neural network for saliency detection in event-camera data
Author : Amélie Gruel - Université Côte d'Azur, CNRS/i3S, France - amelie.gruel@i3s.unice.fr
Run as : $ python3 getSaliency.py nest
Use of DDD17 (DVS Driving Dataset 2017) annotated by [Alonso, 2017]
"""

from pyNN.nest import connectors
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
from pyNN.space import Grid2D
from quantities.units.radiation import R
from events2spikes import ev2spikes
import matplotlib.pyplot as plt
import numpy as np 
from numpy.random import randint
import sys
import neo
from quantities import ms
from tqdm import tqdm


### Configure simulator
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--events","Get input file with inputs"),
                             ("--downscale", "Downscale the connection between events and reaction", {"action":"store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}))

if options.debug:
    init_logging(None, debug=True)

if sim == "nest":
    from pyNN.nest import *
elif sim == "spinnaker":
    import pyNN.spiNNaker as sim

dt = 1/80000
sim.setup(timestep=0.01)

downscale_factor_1D = 10
downscale_factor_2D = downscale_factor_1D*downscale_factor_1D

# ### Get Data
try : 
    ev = np.load(options.events)
    ev[:,2] *= 1000
except (ValueError, FileNotFoundError, TypeError):
    print("Error: The input file has to be of format .npy")
    sys.exit()
time_data = int(np.max(ev[:2000,2]))
print(time_data)
ev, x_input, y_input = ev2spikes(ev[:5000], coord_t=2)
print(x_input, y_input)


# ev = [sorted(list(randint(0, time_data/2, randint(0,2*time_data/3)))),
#       [4,200,400],
#       sorted(list(randint(time_data/2, time_data, randint(0,time_data/3)))),
#       [150,200,450,470]]

### Network

# populations
print("Initiating populations...")

Input = sim.Population(
    x_input*y_input,
    sim.SpikeSourceArray(spike_times=ev),
    label="Input",
    structure=Grid2D(dx=1, dy=1, z=-50, aspect_ratio=x_input/y_input)
)
Input.record("spikes")

Events_parameters = {
    'tau_m': 25,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV) 
    'v_rest': -65.0,    # resting membrane potential (in mV) !
    'v_thresh': -25,    # spike threshold (in mV) 
}
Events = sim.Population(
    x_input*y_input,
    sim.IF_cond_exp(**Events_parameters),
    label="Events",
    structure=Grid2D(dx=1, dy=1, z=0, aspect_ratio=x_input/y_input)
)
Events.record(("spikes","v"))

Reaction_parameters = {
    'tau_m': 25,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV) 
    'v_rest': -65.0,    # resting membrane potential (in mV) !
    'v_thresh': -25,    # spike threshold (in mV) 
}

if options.downscale:
    Reaction = sim.Population(
        int(x_input/downscale_factor_1D) * int(y_input/downscale_factor_1D),
        sim.IF_cond_exp(**Reaction_parameters),
        label="Reaction",
        structure=Grid2D(dx=downscale_factor_1D, dy=downscale_factor_1D, z=50, aspect_ratio=x_input/y_input)
    )

    subRegions = []
    for v in range(Reaction.size):
        subRegions.append(
            sim.PopulationView(Events, np.array(range(downscale_factor_2D*v, downscale_factor_2D*(v+1))))
        )

else : 
    Reaction = sim.Population(
        x_input*y_input,
        sim.IF_cond_exp(**Reaction_parameters),
        label="Reaction",
        structure=Grid2D(dx=1, dy=1, z=50, aspect_ratio=x_input/y_input)
    )
Reaction.record(("spikes","v"))

print("Size of populations :",Events.size, Reaction.size)

# connections
print("Initiating connections...")

input2events = sim.Projection(
    Input, Events,
    connector=sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=1),
    receptor_type="excitatory",
    label="Connection input to events"
)

if options.downscale :
    for n in tqdm(range(Reaction.size)):
        events2reaction = sim.Projection(
            subRegions[n], sim.PopulationView(Reaction, [n]),
            connector=sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=1),
            receptor_type="excitatory",
            label="Connection events to reaction layer"
        )

else :
    events2reaction = sim.Projection(
        Events, Reaction,
        connector=sim.OneToOneConnector(),
        synapse_type=sim.StaticSynapse(weight=1),
        receptor_type="excitatory",
        label="Connection events to reaction layer"
    )

feedback = sim.Projection(
    Reaction, Events,
    connector=sim.DistanceDependentProbabilityConnector("exp(-d+50)"),
    synapse_type=sim.StaticSynapse(weight=1),
    receptor_type="excitatory",
    label="Feedback connection reaction to input"
)


"""
WTA = sim.Projection(
    Events, Events,
    connector=sim.AllToAllConnector(allow_self_connections=True),
    synapse_type=sim.StaticSynapse(weight=0.5),
    receptor_type="inhibitory",
    label="Winner-Takes-All"
)
"""

print("Network initilisation OK")

### Run simulation
"""
*************************************************************************************************
From example "simple_STDP.py" on : http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html
"""

class WeightRecorder(object):
    """
    Recording of weights is not yet built in to PyNN, so therefore we need
    to construct a callback object, which reads the current weights from
    the projection at regular intervals.
    """

    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection
        self._weights = []

    def __call__(self, t):
        print(t)
        if type(self.projection) != list:
            self._weights.append(self.projection.get('weight', format='list', with_address=False))
        elif type(self.projection) == list:
            for proj in self.projection:
                self._weights.append(proj.get('weight', format='list', with_address=False))
        return t + self.interval

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal

weight_recorder = WeightRecorder(sampling_interval=1.0, projection=events2reaction)
"""
**************************************************************************************************
"""

print("Start simulation")

sim.run(time_data, callbacks=[weight_recorder])
print("Simulation done")

### Simulation results
Events_data = Events.get_data().segments[0]
Reaction_data = Reaction.get_data().segments[0]
weights = weight_recorder.get_weights()

filename = normalized_filename("Results", "Saliency detection", "pkl", options.simulator)

if options.plot_figure:
    figure_filename = filename.replace("pkl", "png")
    Figure(
        # raster plot of the event inputs spike times
        Panel(Events_data.spiketrains, ylabel="Events spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        # raster plot of the Reaction neurons spike times
        Panel(Reaction_data.spiketrains, ylabel="Reaction spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        # membrane potential of the Reaction neurons
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV) event2reaction", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False, xlabel="Time (ms)", xticks=True),
        
        # evolution of the synaptic weights with time
        # Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights event2reaction",
        #         legend=False, xlim=(0, time_data)),
        title="Saliency detection",
        annotations="Simulated with "+ options.simulator.upper() + "\nInput events from "+options.events
    ).save(figure_filename)

sim.end()