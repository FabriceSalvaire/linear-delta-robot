####################################################################################################

import math

import numpy as np

from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid.anchored_artists import AnchoredDrawingArea
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab

####################################################################################################

def compute_zi(x, y, z):

    q1 = (rho - x)**2 + y**2
    q2 = (rho/2. + x)**2 + (math.sqrt(3)/2.*rho - y)**2
    q3 = (rho/2. + x)**2 + (math.sqrt(3)/2.*rho + y)**2

    # Compute valid region on the mesh
    valid1 = np.where(q1 < .9*L**2, True, False)
    valid2 = np.where(q2 < .9*L**2, True, False)
    valid3 = np.where(q3 < .9*L**2, True, False)
    valid = valid1 * valid2 * valid3 # Fixme: valid1 and ...

    # Force to z out of the valid region
    z1 = z + np.sqrt(L**2 - np.where(valid, q1, L**2))
    z2 = z + np.sqrt(L**2 - np.where(valid, q2, L**2))
    z3 = z + np.sqrt(L**2 - np.where(valid, q3, L**2))

    # Force to z out of the valid region
    dz1dx = (-x+rho)/np.sqrt(L**2 - np.where(valid, q1, -1e6))

    return z1, z2, z3, dz1dx, valid

####################################################################################################

print "@origin:", compute_zi(0, 0, 0)

number_of_points = 2*R + 1

x = np.linspace(-R, R, number_of_points)
y = np.linspace(-R, R, number_of_points)
#y = np.ones(number_of_points) * 10
#y = np.zeros(number_of_points)
z = np.zeros(number_of_points)

x, y = np.meshgrid(x, y)
z1, z2, z3, dz1dx, valid = compute_zi(x, y, 0)

####################################################################################################

# figure2 = plt.figure()
# axe2 = plt.subplot(111)
# pylab.grid()
# pylab.plot(x, y, z1, 'r')
# pylab.plot(x, z2, 'g')
# pylab.plot(x, z3, 'b')
# pylab.show()

# figure3 = plt.figure()
# axe3 = figure3.gca(projection='3d')
# surface_z1 = axe3.plot_surface(x, y, dz1dx,
#                                rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# figure4 = plt.figure()
# axe4 = figure4.gca(projection='3d')
# surface_z2 = axe4.plot_surface(x, y, z2,
#                                rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# figure5 = plt.figure()
# axe5 = figure5.gca(projection='3d')
# surface_z3 = axe5.plot_surface(x, y, z3,
#                                rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# figure6 = plt.figure()
# axe6 = figure6.gca(projection='3d')
# surface_valid = axe6.plot_surface(x, y, valid,
#                                rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

figure = plt.figure()
axe = figure.gca(projection='3d')
surface_dz1dx = axe.plot_surface(x, y, dz1dx,
                                 rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

####################################################################################################

plt.show()

####################################################################################################
#
# End
#
####################################################################################################