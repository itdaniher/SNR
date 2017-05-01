from svd_functions import *


# Create Signal
signal, noisy_signal = signal_noise(sn_ratio = 1)

# Add sin wave to signal
# noisy_signal = noisy_signal + (signal_list(signal_func=sin_signal) - .5)

# Filter Signal
svd_filtered_cut = svd_filter(noisy_signal, partition_type="cut")
svd_filtered_sliding = svd_filter(noisy_signal, partition_type="sliding")

# Print mean squared errors
print("Noisy Signal:\t", mean_squared_error(signal, noisy_signal))
print("SVD Cut:\t", mean_squared_error(signal, svd_filtered_cut))
print("SVD Sliding:\t",mean_squared_error(signal, svd_filtered_sliding))

# Plot all vectors
plot_vector([noisy_signal, svd_filtered_cut, svd_filtered_sliding, signal], trace_labels=['noisy', 'filtered_cut', 'filtered_sliding', 'original'])
