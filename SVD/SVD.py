from svd_functions import *


signal, noisy_signal = signal_noise(sn_ratio = 0)
# noisy_signal = noisy_signal + (signal_list(signal_func=sin_signal) - .5)
svd_filtered = svd_filter(partition_vector(noisy_signal, 100), num_terms=1)

print(mean_squared_error(signal, noisy_signal))
print(mean_squared_error(signal, svd_filtered))

plot_vector([noisy_signal, svd_filtered, signal])
