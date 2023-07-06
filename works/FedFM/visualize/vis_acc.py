import matplotlib.pyplot as plt
import numpy as np

fedfm_acc = np.array([0.0158, 0.0881, 0.1606, 0.2118, 0.2536, 0.2919, 0.3128, 0.3293, 0.3445, 0.3564, 0.2692, 0.2723, 0.2884, 0.2972, 0.3022, 0.3071, 0.3099, 0.3166, 0.3236, 0.321, 0.3194, 0.3275, 0.3264, 0.3335, 0.3279, 0.3294, 0.3308, 0.3298, 0.3336, 0.332, 0.3311, 0.3322, 0.3319, 0.3293, 0.3238, 0.3235, 0.3125, 0.3143, 0.3097, 0.3198])


plt.figure()
plt.title('Average Accuracy vs Communication rounds')
# plt.plot(range(len(fedavg_acc)), fedavg_acc, color='k', label='FedAvg')
plt.plot(range(len(fedfm_acc)), fedfm_acc, color='b', label='FedFM')

plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('acc_curves/fedfm_acc_list.png')