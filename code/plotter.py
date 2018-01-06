from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plot_function(X, Y, function_to_plot):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    #X = np.arange(-20, 20, 0.25)
    #Y = np.arange(-20, 20, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = function_to_plot(np.array([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_errors(train_errors, val_errors, train_label, val_label, title, fname):
    train_loss_line, = plt.plot(np.arange(1, len(train_errors) + 1), train_errors, label=train_label)
    val_loss_line, = plt.plot(np.arange(1, len(val_errors) + 1), val_errors, label=val_label)
    plt.legend(handles=[train_loss_line, val_loss_line])
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True)
    plt.savefig("{}".format(fname))
    plt.close()