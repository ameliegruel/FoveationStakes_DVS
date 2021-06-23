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
from numpy.random import randint
import sys
import neo
from quantities import ms


### Configure simulator
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--events","Get input file with inputs"),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

if sim == "nest":
    from pyNN.nest import *
elif sim == "spinnaker":
    import pyNN.spiNNaker as sim

dt = 1/80000
sim.setup(timestep=0.01)


# ### Get Data
# try : 
#     ev = np.load(options.events)
# except (ValueError, FileNotFoundError, TypeError):
#     print("Error: The input file has to be of format .npy")
#     sys.exit()
# x_input, y_input = ev.shape[:-1]
# print(x_input,y_input)
time_data = 500
ev = [sorted(list(randint(0, time_data/2, randint(0,2*time_data/3)))),
      [4,200,400],
      sorted(list(randint(time_data/2, time_data, randint(0,time_data/3)))),
      [150,200,450,470]]

### Network

# populations
Events = sim.Population(
    # x_input*y_input,
    4,
    sim.SpikeSourceArray(spike_times=ev),
    label="Events"
)
Events.record("spikes")

Reaction_parameters = {
    'tau_m': 20.0,      # membrane time constant (in ms)
    'tau_refrac': 0.1,  # duration of refractory period (in ms)
    'v_reset': -65.0,   # reset potential after a spike (in mV) 
    'v_rest': -200.0,    # resting membrane potential (in mV) !
    'v_thresh': -5,    # spike threshold (in mV) 
}
Reaction = sim.Population(
    # x_input*y_input,
    4,
    sim.IF_cond_exp(**Reaction_parameters)
)
Reaction.record(("spikes","v"))
print(Reaction.get("v_rest"))

# connections
events2reaction = sim.Projection(
    Events, Reaction,
    connector=sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=np.random.uniform(0.5, 1, (Events.size, Reaction.size))),
    receptor_type="excitatory",
    label="Connection input events to reaction layer"
)

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

class IntrinsicPlasticity(object):
    def __init__(self, sampling_interval):
        self.interval = sampling_interval,
        self.projection = events2reaction,
        self.source = Events,
        self.target = Reaction,
        self.rest = self.target.get("v_rest")

    def __call__(self,t):
        # si pas d'events dans une timewindow partout 
        self.rest = np.where(t_post_target > t - 1, np.max(self.th*self.delta_th_post, self.th_min), self.th)
        increase = np.zeros(self.target.size)
        increase = np.sum(np.where(np.logical_and(np.greater_equal(self.tau_ip_plus, delta_t), np.greater(delta_t, self.tau_ip_minus)), np.min(increase+self.delta_th_pair*self.w[synapse], self.th_max), increase), axis=0)
        self.th = self.th + increase
        # second si :events à 2 neurones en même temps sur même pas de temps
        


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
        Panel(Reaction_data.filter(name='v')[0], ylabel="Membrane potential (mV) event2reaction", yticks=True, xlim=(0, time_data), linewidth=0.2),
        
        # evolution of the synaptic weights with time
        # Panel(weights, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights event2reaction",
        #         legend=False, xlim=(0, time_data)),
        title="Saliency detection",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)

sim.end()