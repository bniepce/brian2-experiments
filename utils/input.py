import brian2 as b2
import numpy as np
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import instantaneous_rate
from quantities import Hz, s, ms

def get_step_current(t_start, t_end, unit_time, current_unit, amplitude, append_zero=False):
    tmp_size = 1 + t_end
    if append_zero:
        tmp_size += 1
    tmp = np.zeros((tmp_size)) * current_unit
    tmp[t_start: t_end + 1] = amplitude * current_unit
    curr = b2.TimedArray(tmp, dt=1 * unit_time)
    return curr


def spike_train_to_times(spike_trains, unit = b2.ms, dt = 1):
    '''Take a spike train as a list and returns its equivalent spike times.
    
    Parameters
    ----------
    x : np.array
        One sampled spike train
    '''
    times = [idx * dt for idx, i in enumerate(spike_trains) if i == 1]
    return times * unit


def random_spike_train(rate, start, end):
    return homogeneous_poisson_process(rate=rate, t_start=start, t_stop=end)

def get_average_spike_rate(spike_train, unit):
    return np.mean(instantaneous_rate(spike_train, unit))
    