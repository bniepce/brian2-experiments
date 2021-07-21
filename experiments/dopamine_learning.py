from topology.net import SNN
import numpy as np
import brian2 as b2
import random, torch
import matplotlib.pyplot as plt
from utils import *
from topology.connections import ForwardConnection, Diehl_and_Cook_STDP, STDPConnection
from topology.layers import ManualSpikeInput, LIF, Diehl_and_Cook_LIF, Izhikevich
import h5py

if __name__ == "__main__":
    
    # INIT PARAMETERS
    weights = np.random.random((1, 100)) + 0.01
    weights *= 0.3
    
    # GET DATASET
    dataset_path = '../data/bsa_encoded_spike_trains.h5'
    spike_train = np.random.choice([0, 1], size=(300, ), p=[1./3, 2./3])

    print('\nLoading dataset from : {}\n'.format(dataset_path))
    f = h5py.File(dataset_path, 'r')
    x, y = torch.Tensor(f['x'][:]), torch.Tensor(f['y'][:])
    dataset = torch.utils.data.TensorDataset(x, y)
    
    input_layer = ManualSpikeInput(spike_train, dt=1, n_neurons=1, name='input_spikes')
    hidden_layer = LIF(n_neurons = 12, name = 'hidden_lif_layer')
    net = SNN(name='simple_simulation')
    
    net.add_layer(input_layer)
    net.add_layer(hidden_layer)
    net.add_connection(Diehl_and_Cook_STDP(input_layer, hidden_layer, weights, name = 'ee_input'))
    net.train(dataset, sim_duration=300*b2.ms)
    
    #plot_all(net.monitors['hidden_lif_layer'][0])
    #plot_state_variables(net.monitors['hidden_lif_layer'][0], layer_name = 'hidden_lif_layer')
    #plot_spiking_activity(net.monitors['hidden_lif_layer'][1], layer_name = 'hidden_lif_layer')
