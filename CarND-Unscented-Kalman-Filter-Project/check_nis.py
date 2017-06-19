import numpy as np
import matplotlib.pyplot as plt

lidar_nis = np.loadtxt("./data/lidar_nis.txt")
radar_nis = np.loadtxt("./data/radar_nis.txt")

# lidar has 2 degree of freedom so we would expect the 95% reference to be 5.991
lidar_ref = np.ones(len(lidar_nis))*5.991;
# radar has 3 degree of freedom so the reference is 7.815
radar_ref = np.ones(len(radar_nis))*7.815;

plt.figure()
plt.plot(range(len(lidar_nis)), lidar_nis, range(len(lidar_nis)), lidar_ref)

plt.figure()
plt.plot(range(len(radar_nis)), radar_nis, range(len(radar_nis)), radar_ref)

plt.show()

