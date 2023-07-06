import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fed_acc_matrix = np.array((
    [46.55, 47.83, 49.79, 50.36, 50.44],
    [44.94, 47.72, 48.74, 50.69, 51.13],
    [44.95, 49.03, 50.74, 50.23, 51.98],
    [44.44, 45.83, 49.58, 51.67, 50.81],
    [44.13, 47.55, 46.93, 50.22, 52.65]
))

only_acc_matrix = np.array((
    [44.95, 46.16, 50.39, 48.31, 48.46],
    [0,0,0,0,0],
    [41.73, 46.91, 50.13, 48.86, 50.00],
    [44.87, 44.44, 48.82, 50.39, 49.94],
    [41.83, 47.26, 46.25, 50.54, 51.76]
))

values = np.array((range(fed_acc_matrix.shape[0])))

ax = sns.heatmap(fed_acc_matrix, cmap="Blues")
ax.set_xticks(values+0.5)
ax.set_yticks(values+0.5)
plt.xticks(ticks=values+0.5,labels=5*(values+1))
plt.yticks(ticks=values+0.5,labels=np.flip(5*(values+1), 0))
plt.xlabel('No. uniform clients')
plt.ylabel('No. 2-class clients')
plt.title('Performance vs client number (1000 samples per client)')
ax.figure.savefig("output.png")