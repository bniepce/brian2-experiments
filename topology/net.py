import brian2 as b2
import numpy as np
from tqdm import tqdm
from utils import spike_train_to_times, plot_spiking_activity

class SNN(b2.Network):
    '''
    Network is simulated in a feed forward way, first added layer is the first
    to get simulated.
    '''
    def __init__(self, name : str):
        super().__init__()
        self.layers = {}
        self.connections = {}
        self.monitors = {}
    
    def add_layer(self, layer : object) -> None:
        '''
        Add a neuron layer to the Network.
        This method also provides monitor for the NeuronGroup created.
        '''
        neurons = layer.neurons
        var_list = layer.var_list
        self.layers[layer.name] = layer
        if isinstance(neurons, b2.SpikeGeneratorGroup):
            self.monitors[layer.name] = b2.SpikeMonitor(neurons)
            self.add([neurons, self.monitors[layer.name]])
            
        else:
            self.monitors[layer.name] = [b2.StateMonitor(neurons, var_list, record=True),
                                b2.SpikeMonitor(neurons)]
            self.add([neurons, self.monitors[layer.name]])
        self.store()

    def add_connection(self, conn : object) -> None:
        self.connections[conn.name] = conn
        if conn.var_list:
            self.monitors[conn.name] = b2.StateMonitor(conn.synapses, conn.var_list, record = [0])
            self.add(self.monitors[conn.name])
        self.add(conn.synapses)
        self.store()
        
    def describe(self):
        
        print('\x1b[0;33;40m'+'\nNetwork architecture : '+'\x1b[0m')
        print('\033[1m'+ '\nNeuron groups :'+'\x1b[0m')
        for idx, (key, value) in enumerate(self.layers.items()):
            print('    {}. {}'.format(idx + 1, value.__class__.__name__))
            
        print('\033[1m'+'\nConnections : '+'\x1b[0m')
        for idx, (key, value) in enumerate(self.connections.items()):
            print('    {}. {}'.format(idx + 1,value.__class__.__name__))
        
        monitors = [i for i in self.sorted_objects if isinstance(i, b2.StateMonitor) \
            or isinstance(i, b2.SpikeMonitor)]
        print('\033[1m'+ '\nMonitors :'+'\x1b[0m')
        for idx, i in enumerate(monitors):
            print('    {}. {}'.format(idx + 1, i.name))
        print('\n')
        return
    
    def run_simulation(self, duration):
        '''Run neuron dynamics.
    
        Parameters
        ----------
        input : brian2.TimedArray
            A brian TimedArray containing a list of current value in amp.
        duration : brian2.Quantity
            A duration time for simulation, forced to be a b2.Quantity instance.
        '''
        print('\n##### Launching SNN Simulation #####\n')
        self.describe()
        self.run(duration, report='text')
        # plot_connection_activity(self.connections[0], self.monitors['conn'])

    def train(self, dataset, sim_duration):
        '''Train the SNN
    
        Parameters
        ----------
        dataset : torch Dataset
            Dataset iterator
        sim_duration : brian2.Quantity
            A duration time for simulation, forced to be a b2.Quantity instance.
        '''
        print('\n##### Launching SNN Training #####\n')
        self.describe()

        n_class                 = len(set(dataset.tensors[1].flatten().numpy()))
        input_neurons           = list(self.layers.items())[0][1].neurons
        hidden_neurons          = list(self.layers.items())[1][1].neurons
        n_neurons               = hidden_neurons.N
        spike_monitor           = self.monitors[list(self.layers.items())[1][0]][1]
        assignment              = np.zeros(n_neurons)
        label_wise_spike_record = np.zeros((n_class, n_neurons))
        
        with tqdm(total=len(dataset), desc='Training progress : ') as pbar:
            for i, (x, y) in enumerate(dataset):
                if i > 2:
                    break
                data, target = spike_train_to_times(x.numpy()), y.numpy()
                indices = [input_neurons.N - 1 for i in range(len(data))]
                input_neurons.set_spikes(indices=indices, times=data)
                
                self.run(sim_duration)
                label_wise_spike_record[int(y)] = spike_monitor.count[:]
            
                # ADD LABEL RECORDING            
                pbar.update()
                self.restore()
                            
                            
        print(label_wise_spike_record)