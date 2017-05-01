from svd_functions import *


# Create Signal
signal, noisy_signal = signal_noise(sn_ratio = 0.5)

# Add sin wave to signal
# noisy_signal = noisy_signal + (signal_list(signal_func=sin_signal) - .5)

# Filter Signal
svd_filtered = svd_filter(noisy_signal, window_size=100, num_terms=1)

# Print mean squared errors
print(mean_squared_error(signal, noisy_signal))
print(mean_squared_error(signal, svd_filtered))

# Plot all vectors
plot_vector([noisy_signal, svd_filtered, signal])
