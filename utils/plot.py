import brian2 as b2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')

def plot_all(S):
    t = S.t / b2.ms
    for i in range(len(S.v)):
        v = S.v[i] / b2.mV
        plt.ion()
        plt.figure(figsize=(15,7))
        plt.suptitle('State variables for neuron : {}'.format(i))
        plt.subplot(211)
        plt.plot(t, v, color = u'#d84a37', label='Membrane potential')
        plt.legend(loc=1)
        plt.xlim(0, t[-1])
        plt.xlabel("t [ms]")
        plt.ylabel("v [mV]")
        plt.ioff()
    plt.show()
    
def plot_state_variables(S, layer_name, neuron_id = 0):
    
    t = S.t / b2.ms
    v = S.v[neuron_id] / b2.mV
    
    plt.ion()
    plt.figure(figsize=(15,7))
    plt.suptitle('State variables for neuron layer : {}'.format(layer_name))
    plt.subplot(211)
    plt.plot(t, v, color = u'#d84a37', label='Membrane potential')
    plt.legend(loc=1)
    plt.xlim(0, t[-1])
    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.ioff()
    plt.show()
    
    
def plot_spiking_activity(S, layer_name):
    t = S.t / b2.ms
    N = len(list(set(S.all_values()['t'])))
    plt.ion()
    plt.figure(figsize=(15,7))
    plt.suptitle('Spiking activity on layer : {}'.format(layer_name))
    plt.plot(t, S.i, '.', label='Spike')
    plt.legend(loc=1)
    plt.xlim(0, t[-1])
    plt.ylim(-1, N + 1)
    plt.xlabel('t [ms]')
    plt.ylabel('Neuron index')
    plt.grid()
    plt.ioff()
    plt.show()

def plot_stdp_activity(connection, monitor):
    plt.subplot(311)
    gmax = connection.parameters['gmax']
    S = connection.synapses
    plt.ion()
    plt.plot(S.w / gmax, '.k')
    plt.ylabel('Weight / gmax')
    plt.xlabel('Synapse index')
    
    plt.subplot(312)
    plt.hist(S.w / gmax, 20)
    plt.xlabel('Weight / gmax')
    
    plt.subplot(313)
    plt.plot(monitor.t/ b2.second, monitor.w.T/gmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight / gmax')
    plt.tight_layout()
    plt.ioff()
    plt.show()