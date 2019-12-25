import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from drawnow import drawnow



def plot_3d(teritry):
    print('test')
    x = []
    y = []
    z = []
    fig = plt.figure()
    m = '.'
    ax = plt.axes(projection='3d')
    for i in range(0,len(teritry[0])):
        col1 = [val[i] for val in teritry]

        if col1[0] == 0 and col1[1] == 0 and col1[2] == 0 :
            continue

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        xs = col1[0]
        ys = col1[1]
        zs = col1[2]
        ax.scatter(xs, ys, zs , marker=m)
        # Data for a three-dimensional line
        x.append(col1[0])
        y.append(col1[1])
        z.append(col1[2])


     
        # ax.plot3D(int(col1[0]), int(col1[1]), int(col1[2]), 'gray')
        # Data for three-dimensional scattered points
        # zdata = 15 * np.random.random(100)
        # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
        # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
        # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    

    ax.plot3D(x, y, z, 'blue')
    # zdata = 15 * np.random.random(100)
    # ax.scatter3D(x, y, z, c=np.squeeze(zdata), cmap='gray')

    plt.show()


