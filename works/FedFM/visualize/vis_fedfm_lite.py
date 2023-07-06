from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

fontsize=20
fig, ax = plt.subplots()
fig.set_size_inches(7.5, 5.7)
ax.set_xscale('log', base=2)
ax.set_axisbelow(True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Background color
ax.set_facecolor('#eaeaf1')
ax.grid(True, color='white', linestyle='-', linewidth=1, which='both')



num_params = 11689512
num_anchors = 5120
total_round = 100
anchor_round = 80

fedfm_lite_x = np.array([num_params*total_round/10+num_anchors*anchor_round, num_params*total_round/5+num_anchors*anchor_round, num_params*total_round/2+num_anchors*anchor_round, num_params*total_round+num_anchors*anchor_round])
fedfm_lite_y = np.array([65.97, 70.66, 71.62, 72.71])
baseline_x = np.array([num_params*total_round/10, num_params*total_round/5, num_params*total_round/2, num_params*total_round])
fedavg_y = np.array([62.53, 66.11, 67.22, 66.69])
feddyn_y = np.array([58.17, 64.17, 69.10, 68.32])
moon_y = np.array([64.15, 66.22, 68.16, 67.74])
scaffold_y = np.array([59.01, 65.72, 67.73 ,69.45, 69.91])

plt.plot(fedfm_lite_x, fedfm_lite_y, c='red', marker='s', linewidth=3, markersize=15, label='FedFM-Lite')
# plt.plot(baseline_x, fedavg_y, c='g', marker='s', linewidth=3, markersize=15, label='FedAvg')
# plt.plot(baseline_x[filter], feddyn_y[filter], c='k', marker='s', linewidth=3, markersize=15, label='FedDyn')
# plt.plot(baseline_x[filter], moon_y[filter], c='burlywood', marker='s', linewidth=3, markersize=15, label='MOON')
plt.plot(baseline_x[:4], scaffold_y[:4], c='blue', marker='s', linewidth=3, markersize=15, label='SCAFFOLD')

# plt.scatter(num_params*total_round+num_anchors*anchor_round, 72.89, c='slateblue', marker='s', s=200, label='FedAvg')
# plt.scatter(num_params*total_round, 66.69, c='g', marker='s', s=200, label='FedAvg')
# plt.scatter(num_params*total_round, 68.32, c='k', marker='s', s=200, label='FedDyn')
# plt.scatter(num_params*total_round, 67.74, c='burlywood', marker='s', s=200, label='MOON')
# plt.scatter(num_params*total_round*2, 69.91, c='blue', marker='s', s=200, label='SCAFFOLD')

plt.legend(fontsize=15, loc='lower right')
plt.xticks(fontsize=15, fontweight='medium')
plt.yticks(fontsize=15, fontweight='medium')
plt.xlabel('Communication Cost', fontsize=fontsize, fontweight='medium')
plt.title('Accuracy (%)', fontsize=fontsize, fontweight='medium')

# plt.plot(fedavg_disco-fedavg, label='FedAvg', color='r', marker='s')
# plt.plot(fedprox_disco-fedprox, label='FedProx', color='b', marker='s')
# plt.plot(feddyn_disco-feddyn, label='FedDyn', color='k', marker='s')
# plt.plot(moon_disco-moon, label='MOON', color='deeppink', marker='s')



# plt.plot(fedavg, label='FedAvg', color='r')
# plt.plot(fedavg_disco, label='FedAvg+Disco', color='r', linestyle='dashed')

# plt.plot(fedprox, label='FedProx', color='b')
# plt.plot(fedprox_disco, label='FedProx+Disco', color='b', linestyle='dashed')

# plt.plot(feddyn, label='FedDyn', color='deeppink')
# plt.plot(feddyn_disco, label='FedDyn+Disco', color='deeppink', linestyle='dashed')

# plt.plot(moon, label='MOON', color='k')
# plt.plot(moon_disco, label='MOON+Disco', color='k', linestyle='dashed')

plt.savefig('fedfm_lite_v2.png', dpi=300, bbox_inches='tight')