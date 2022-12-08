import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import time

correctFile = False
while(not correctFile):
    filename = input("Please enter filename: \n")
    try:
        with open(filename) as f:
            boidDataList = [line.rstrip() for line in f]
        correctFile = True
    except:
        print("The filename you entered does not exist")



nBoids = int(boidDataList[0])
nFrames = int(boidDataList[1])
xSize = int(boidDataList[2])
ySize = int(boidDataList[3])
startLine = 5
listLength = len(boidDataList)

boidData = np.zeros(((nFrames, nBoids, 4)))
boidPositionsData = np.zeros(((nFrames, nBoids, 2)))


boidDataList = boidDataList[startLine:]
boidData = np.reshape(boidDataList, (nFrames,nBoids,4)).astype(float)

del boidDataList




fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, xSize), ylim=(0, ySize))

boids, = ax.plot([], [], 'bo', ms=6)


def init():
    """initialize animation"""
    global boidData
    boids.set_data([], [])
    return boids, 


def animate(i):
    """perform animation step"""

    global boidData, fig, ax
    xList = boidData[i].T[0]
    yList = boidData[i].T[1]
    boids.set_data(xList, yList)
    boids.set_markersize(3)
    return boids, 


ani = animation.FuncAnimation(fig, animate, frames=nFrames,
                            interval = 10, blit=True, init_func=init)

plt.show()







