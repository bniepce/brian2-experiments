import numpy as np

def smooth(x,window_len=11, window='hanning'):
    '''
    Moving average window smoothing function
    '''
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


# def compute_bsa_decoded_series(series, mean, std, step, amp, threshold):
#     bsa = BSAEncoder(filter_response=signal.gaussian(M=mean, std=std), 
#                  step=step, filter_amp=amp, threshold=threshold)
#     spikes = bsa.encode(series)
#     enc = bsa._last_spike_times
#     dec = bsa.decode(plot=False)
#     return series, enc, dec


# def bsa_analysis(data, win_size, mean, std, amp, step, threshold):
#     x_smooth = smooth(data, win_size)
#     x_smooth = np.squeeze(cv2.normalize(x_smooth, None, 0.0, 1.0, cv2.NORM_MINMAX))
#     series, enc, dec = compute_bsa_decoded_series(x_smooth, mean, std, step, amp, threshold)

#     plt.subplot(311)
#     plt.plot(series)

#     plt.subplot(312)
#     plt.eventplot(enc)

#     plt.subplot(313)
#     plt.plot(dec)