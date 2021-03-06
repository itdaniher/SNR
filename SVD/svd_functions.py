import numpy as np
from numpy import pi, sin
import matplotlib.pyplot as plt
import random


LENGTH_MS = 1000;
SAMPLING_FREQ = 2000;

WINDOW_SIZE = 10
NUM_TERMS = 1


def signal_function(t):
    """ Returns values to be used as a signal.
    """
    # return square_signal(t)
    return sin_signal(t)

def sin_signal(t):
    """ Returns values of many sin waves.
    """
    # return sin(10*t)
    return np.sin(50*t) + np.sin(20*t) + np.sin(12*t) + np.sin(60*t)

def square_signal(t, clock_ms=100, sampling_freq=SAMPLING_FREQ):
    """ Returns a square signal.
    """
    data = [1,0,1,1,0,1,1,1,0,1]

    num_samples_per_bit = int((clock_ms/1000) * sampling_freq)
    data_stream = np.repeat(data, (num_samples_per_bit,))
    return np.resize(data_stream, (len(t),))


def signal_list(length_ms=LENGTH_MS, sampling_freq=SAMPLING_FREQ, signal_func=None):
    """ Returns a list of values of signal_func. (By default signal_function)
    """

    if signal_func is None:
        signal_func = signal_function

    num_samples = int((length_ms / 1000) * sampling_freq)
    omega = pi*2 / sampling_freq
    xvalues = np.arange(num_samples) * omega
    signal =  signal_func(xvalues)
    signal_norm = signal/max(signal)
    return signal_norm


def noise(length):
    """ Returns a list of random values from -1 to 1.
    """
    return (np.random.random(length) * 2) - 1


def signal_noise(sn_ratio = 1, length_ms=LENGTH_MS, sampling_freq=SAMPLING_FREQ):
    """ Returns two lists, clean signal and signal with noise.
    """
    signal = signal_list(length_ms, sampling_freq)
    noisy_signal = signal + (noise(len(signal)) * sn_ratio)
    return signal, noisy_signal


def plot_vector(wave, downsample=1, title="", ax_labels=["",""], trace_labels = [""]):
    """ Makes a plot of a vector or list of vectors.
    """
    fig, ax = plt.subplots()
    if type(wave) is list:
        if len(trace_labels) < len(wave):
            trace_labels *= len(wave)
        print("Plotting", len(wave), "vectors")
        for (values, label) in zip(wave, trace_labels):
            plt.plot(values[::downsample], label=label)
    else:
        plt.plot(wave[::downsample], label=trace_labels[0])
    ax.set_title(title)
    if trace_labels[0] != "":
        plt.legend()
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    plt.show()


def mean_squared_error(a, b):
    """ Compares two lists with MSE.
    """
    len_to_compare = min(len(a), len(b))
    return ((a[:len_to_compare] - b[:len_to_compare])**2).mean(axis=None)


def partition_vector_sliding(input_vector, window_size, step_size=1):
    """ Partitions a vector into a matrix by sliding a window of window size over it, with a step of step_size.
        Pads input_vector with last value of input vector.
    """
    if(len(input_vector) < window_size):
        raise ValueError("Input vecor is smaller than window size", len(input_vector, window_size))

    input_vector = np.append(input_vector, [input_vector[-1]] * (window_size))
    x_size = int((len(input_vector) - window_size) / step_size) + 1
    # print(len(input_vector)/step_size, x_size)

    out_matrix = np.zeros([x_size, window_size])
    for i in range(x_size):
        out_matrix[i] = input_vector[i*step_size:(i*step_size)+window_size]

    # print(out_matrix)
    return out_matrix


def unpartition_matrix_sliding(input_matrix, step_size=1):
    """ Returns a vector from a matrix made with 'sliding'
    """
    out_vector = np.append(input_matrix[:-1,:step_size], input_matrix[-1, :])
    print(max(out_vector))
    return out_vector


def partition_vector_cut(input_vector, window_size):
    """ Cuts a vector up into a matrix without overlapping. Discards undivisible data.
    """
    if(len(input_vector) < window_size):
        raise ValueError("Input vecor is smaller than window size", len(input_vector, window_size))

    x_size = int(len(input_vector) / window_size)
    out_matrix = np.zeros([x_size, window_size])
    for i in range(x_size):
        out_matrix[i] = input_vector[i*window_size : (i*window_size)+window_size]

    return out_matrix

def unpartition_matrix_cut(input_matrix):
    """ Returns a vector from a matrix made with 'cut'
    """
    out_vector = input_matrix.flatten()
    return out_vector


def svd_filter(input_vector, window_size=WINDOW_SIZE, num_terms=NUM_TERMS, step_size=1, partition_type="cut"):
    """ Returns a vector filtered by svd. It partitions the input, takes num_terms largest svd values,
        then unpartitions the resulting matrix into a filtered vector.
    """
    if partition_type == "cut":
        svd_input = partition_vector_cut(input_vector, window_size)
    elif partition_type ==  "sliding":
        svd_input = partition_vector_sliding(input_vector, window_size, step_size)

    # print(svd_input, "\n")
    U, s, V = np.linalg.svd(svd_input)
    # print(U,"\n")
    # print(s,"\n")
    # print(V,"\n")
    # print(U.shape, s.shape, V.shape)

    if num_terms > len(s):
        num_terms = len(s)

    S = np.zeros([U.shape[0], V.shape[0]])
    S[:num_terms, :num_terms] = np.diag(s[:num_terms])

    # print(S[:num_terms,:])
    # print()
    #
    # print(np.dot(U, np.dot(S, V)))

    output_matrix = np.dot(U, np.dot(S, V))

    # print(output_matrix)

    if partition_type == "cut":
        output_vector = unpartition_matrix_cut(output_matrix)
    if partition_type == "sliding":
        output_vector = unpartition_matrix_sliding(output_matrix, step_size)[:len(input_vector)]
        # output_vector = output_vector / max(output_vector)

    return output_vector
