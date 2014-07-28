#################################################################################################

import math

import numpy as np

from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

####################################################################################################
#
# Mesh - -> +
#
####################################################################################################

####################################################################################################
#
# Linear Delta Robot Parameters
#
####################################################################################################

# unit is mm
R = 300 # tower radius
r = 50 # nacelle radius
L = 1.5*R # arm length
number_of_steps = 400
screw_step = 10

####################################################################################################

def vector_to_list(x):
    return float(x[0]), float(x[1])

####################################################################################################
#
# Computed parameters
#
####################################################################################################

z_step = screw_step / float(number_of_steps)
rho = R - r
z0 = math.sqrt(L**2 - rho**2)

print """
Tower radius: {R} mm
Nacelle radius: {r} mm
Arm length: {L} mm
rho: {rho} mm
Stepper number of steps: {number_of_steps}
Screw step: {screw_step}
z step: {z_step} um
""".format(
    R=R,
    r=r,
    L=L,
    rho=rho,
    number_of_steps=number_of_steps,
    screw_step=screw_step,
    z_step=z_step*1000,
)

R1 = np.ones((2,2))
R2 = np.array([[-1/2., -math.sqrt(3)/2.],
               [math.sqrt(3)/2., -1/2.]])
R3 = np.array([[-1/2., math.sqrt(3)/2.],
               [-math.sqrt(3)/2., -1/2.]])

# Position of the axes
Pxy1 = np.array([[R], [0]])
Pxy2 = R2.dot(Pxy1)
Pxy3 = R3.dot(Pxy1)

Px1, Py1 = vector_to_list(Pxy1)
Px2, Py2 = vector_to_list(Pxy2)
Px3, Py3 = vector_to_list(Pxy3)

Axy1 = np.array([[rho], [0]])
Axy2 = R2.dot(Axy1)
Axy3 = R3.dot(Axy1)

Ax1, Ay1 = vector_to_list(Axy1)
Ax2, Ay2 = vector_to_list(Axy2)
Ax3, Ay3 = vector_to_list(Axy3)

####################################################################################################

def vdot(a, b):
    """ vectorized dot """
    if len(a.shape) == 2:
        c = np.sum(a*b, axis=1)
        c.shape = c.shape[0], 1 # Fixme
        return c
        # return np.einsum('ij,ij->i', a, b)
    else:
        return np.dot(a, b)

####################################################################################################

def vcross(a, b):
    """ vectorized cross product """
    if len(a.shape) == 2:
        return np.cross(a, b, axisa=-1, axisb=-1, axisc=-1)
    else:
        return np.cross(a, b)

####################################################################################################

def norm(x):
    return np.sqrt(vdot(x, x))

####################################################################################################

def unit_vector(x):
    return x / norm(x)

####################################################################################################

def zi_to_pi(xi, yi, zi):
    if np.isscalar(zi):
        pi = np.array((xi, yi, zi))
    else:
        pi = np.zeros((zi.shape[0], 3))
        pi[:,0] = xi
        pi[:,1] = yi
        pi[:,2] = zi
    return pi

####################################################################################################

def direct_cinematic(zi, z_offset=0):

    if isinstance(zi, (list, tuple)) or len(zi.shape) == 1:
        z1, z2, z3 = tuple(zi)
    else:
        z1, z2, z3 = zi[:,0], zi[:,1], zi[:,2]

    p1 = zi_to_pi(Ax1, Ay1, z1 + z_offset)
    p2 = zi_to_pi(Ax2, Ay2, z2 + z_offset)
    p3 = zi_to_pi(Ax3, Ay3, z3 + z_offset)

    v21 = p2 - p1
    v31 = p3 - p1

    l21 = norm(v21)
    # l31 = norm(v31)

    ex = v21 / l21 # Fixme: vectorize
    d = l21
    i = vdot(v31, ex)

    ey = unit_vector(v31 - i * ex)
    j = vdot(v31, ey)

    ez = vcross(ex, ey)

    pex = d/2
    pey = (i**2 + j**2 - i*d)/(2*j)
    pez = np.sqrt(L**2 - pex**2 - pey**2)

    p = p1 + pex*ex + pey*ey - pez*ez

    return p

####################################################################################################

def indirect_cinematic(p, stack_zi=True):

    if isinstance(p, (list, tuple)) or len(p.shape) == 1:
        x, y, z = tuple(p)
    else:
        x, y, z = p[:,0], p[:,1], p[:,2]

    q1 = (x - Ax1)**2 + (y - Ay1)**2
    q2 = (x - Ax2)**2 + (y - Ay2)**2
    q3 = (x - Ax3)**2 + (y - Ay3)**2

    z1 = z + np.sqrt(L**2 - q1)
    z2 = z + np.sqrt(L**2 - q2)
    z3 = z + np.sqrt(L**2 - q3)

    if stack_zi:
        if not np.isscalar(z1):
            z1.shape = z1.shape[0], 1
            z2.shape = z1.shape[0], 1
            z3.shape = z1.shape[0], 1
        return np.hstack((z1, z2, z3))
    else:
        return z1, z2, z3

####################################################################################################

def compute_valid_region(x, y, radius=R):

    """ Compute the valid region
    
    x, y can be scalar or 2D mesh.
    """

    q1 = (x - Ax1)**2 + (y - Ay1)**2
    q2 = (x - Ax2)**2 + (y - Ay2)**2
    q3 = (x - Ax3)**2 + (y - Ay3)**2
    q4 = x**2 + y**2

    valid1 = np.where(q1 < L**2, True, False)
    valid2 = np.where(q2 < L**2, True, False)
    valid3 = np.where(q3 < L**2, True, False)
    valid4 = np.where(q4 < radius**2, True, False)
    valid = valid1 * valid2 * valid3 * valid4 # Fixme: valid1 and ...

    return valid

####################################################################################################

def compute_jacobian_determinant(x, y, z, z1, z2, z3):

    dx1 = x - Ax1
    dx2 = x - Ax2
    dx3 = x - Ax3

    dy1 = y - Ay1
    dy2 = y - Ay2
    dy3 = y - Ay3

    dz1 = z - z1
    dz2 = z - z2
    dz3 = z - z3

    D = ((dx1*dy2 - dx2*dy1) / (dz1*dz2) +
         (dx3*dy1 - dx1*dy3) / (dz1*dz3) +
         (dx2*dy3 - dx3*dy2) / (dz2*dz3))

    return D

####################################################################################################

def compute_jacobian(x, y, z, z1, z2, z3, dx, dy, dz):

    Dx1 = x - Ax1
    Dx2 = x - Ax2
    Dx3 = x - Ax3

    Dy1 = y - Ay1
    Dy2 = y - Ay2
    Dy3 = y - Ay3

    Dz1 = z - z1
    Dz2 = z - z2
    Dz3 = z - z3

    dz1 = Dx1/Dz1*dx + Dy1/Dz1*dy + dz
    dz2 = Dx2/Dz2*dx + Dy2/Dz2*dy + dz
    dz3 = Dx3/Dz3*dx + Dy3/Dz3*dy + dz

    return dz1, dz2, dz3

####################################################################################################

def compute_jacobian_inverse(x, y, z, z1, z2, z3, dz1, dz2, dz3):

    Dx1 = x - Ax1
    Dx2 = x - Ax2
    Dx3 = x - Ax3

    Dy1 = y - Ay1
    Dy2 = y - Ay2
    Dy3 = y - Ay3

    Dz1 = z - z1
    Dz2 = z - z2
    Dz3 = z - z3

    denominator = (Dx1*Dy2 - Dx2*Dy1)*Dz3 + (Dx3*Dy1 - Dx1*Dy3)*Dz2 + (Dx2*Dy3 - Dx3*Dy2)*Dz1

    dx =  (Dy2*Dz1*Dz3 - Dy3*Dz1*Dz2)*dz1 - (Dy1*Dz2*Dz3 - Dy3*Dz1*Dz2)*dz2 + (Dy1*Dz2*Dz3 - Dy2*Dz1*Dz3)*dz3
    dy = -(Dx2*Dz1*Dz3 - Dx3*Dz1*Dz2)*dz1 + (Dx1*Dz2*Dz3 - Dx3*Dz1*Dz2)*dz2 - (Dx1*Dz2*Dz3 - Dx2*Dz1*Dz3)*dz3
    dz = (Dx2*Dy3 - Dx3*Dy2)*Dz1*dz1 - (Dx1*Dy3 - Dx3*Dy1)*Dz2*dz2 + (Dx1*Dy2 - Dx2*Dy1)*Dz3*dz3

    dx /= denominator
    dy /= denominator
    dz /= denominator

    return dx, dy, dz

####################################################################################################

def test():

    z = 1
    zi = float(indirect_cinematic((0, 0, z), stack_zi=False)[0])
    print "p = [0, 0, 1] :", direct_cinematic((zi, zi, zi))
    
    print
    p = direct_cinematic((0, 0, 0), z_offset=L)
    print "p:", p
    print "zi = 0 :", indirect_cinematic(p) - L
    
    print
    p = direct_cinematic((1, 2, 3), z_offset=L)
    print "p:", p
    print "zi = [1, 2, 3] :", indirect_cinematic(p) - L
    
    print
    zi = []
    X = 5
    for z1 in xrange(-X, X):
        for z2 in xrange(-X, X):
            for z3 in xrange(-X, X):
                zi.append((z1, z2, z3))
    zi = np.array(zi)
    print "zi :", zi
    p = direct_cinematic(zi, z_offset=L)
    print "p:", p
    print "delta zi = 0 :", np.all((indirect_cinematic(p) - L - zi) < 1e-6)

####################################################################################################

def plot_workspace_circles():

    figure = plt.figure()
    axe = plt.subplot(111, aspect='equal')
    axe.set_title("Workspace")
    axe.set_xlim(-1.25*R, 1.25*R)
    axe.set_ylim(-1.25*R, 1.25*R)
    axe.grid()
    patch1 = Circle((Ax1, Ay1), L, fc="r", alpha=.5)
    patch2 = Circle((Ax2, Ay2), L, fc="g", alpha=.5)
    patch3 = Circle((Ax3, Ay3), L, fc="b", alpha=.5)
    patch0 = Circle((0, 0), R, fc="white", alpha=.5)
    for patch in patch1, patch2, patch3, patch0:
        axe.add_artist(patch)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')

####################################################################################################

def plot_workspace():

    X = int(1.1*R)
    number_of_points = 2*X + 1
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)
    valid = compute_valid_region(x, y)

    figure = plt.figure()
    axe = plt.subplot(111, aspect='equal')
    axe.set_title("Workspace")
    axe.grid()
    # Fixme: colormap
    axe.imshow(valid, extent=(-X, X, -X, X), aspect='equal', cmap=cm.Blues)
    patch0 = Circle((0, 0), R, fc="white", alpha=.5)
    for patch in patch0,:
        axe.add_artist(patch)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')

####################################################################################################

def plot_jacobian_determinant():

    X = int(1.1*R)
    number_of_points = (2*X + 1)*2/10
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)
    valid = compute_valid_region(x, y)
    xv = x * valid
    yv = y * valid

    z = 1
    z1, z2, z3 = indirect_cinematic((xv, yv, z), stack_zi=False)
    z1 = np.where(valid, z1, 0)
    z2 = np.where(valid, z2, 0)
    z3 = np.where(valid, z3, 0)

    determinant = compute_jacobian_determinant(xv, yv, z, z1, z2, z3)
    determinant = np.where(valid, determinant, 0)
    print determinant

    figure = plt.figure()
    axe = plt.subplot(111, aspect='equal')
    axe.set_title("Jacobian Determinant")
    axe.grid()
    # Fixme: colormap
    image = axe.imshow(determinant, extent=(-X, X, -X, X), aspect='equal')
    plt.colorbar(image)
    patch0 = Circle((0, 0), R, fc="white", alpha=.1)
    for patch in patch0,:
        axe.add_artist(patch)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')

####################################################################################################

def plot_jacobian():

    X = int(1.1*R)
    # number_of_points = (2*X + 1)*2/10
    number_of_points = 100
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)
    valid = compute_valid_region(x, y)
    xv = x * valid
    yv = y * valid

    z = 1
    z1, z2, z3 = indirect_cinematic((xv, yv, z), stack_zi=False)
    z1 = np.where(valid, z1, 0)
    z2 = np.where(valid, z2, 0)
    z3 = np.where(valid, z3, 0)

    dz1, dz2, dz3 = compute_jacobian(xv, yv, z, z1, z2, z3, z_step, 0, 0)
    dz1 = np.where(valid, dz1, 0)
    dz2 = np.where(valid, dz2, 0)
    dz3 = np.where(valid, dz3, 0)

    # dz1 /= z_step
    # dz2 /= z_step
    # dz3 /= z_step

    # Plot dzi
    for dzi, axis in (dz1, 'dz1'), (dz2, 'dz2'), (dz3, 'dz3'):
        figure = plt.figure()
        axe = plt.subplot(111, aspect='equal')
        axe.set_title("{}/dt".format(axis))
        axe.grid()
        # Fixme: colormap
        image = axe.imshow(dzi, extent=(-X, X, -X, X), aspect='equal')
        plt.colorbar(image)
        patch0 = Circle((0, 0), R, fc="white", alpha=.1)
        for patch in patch0,:
            axe.add_artist(patch)
        axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
                 (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
                 'o')

####################################################################################################

def plot_jacobian_inverse():

    X = int(1.1*R)
    # number_of_points = (2*X + 1)*2/10
    number_of_points = 300
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)
    valid = compute_valid_region(x, y)
    xv = x * valid
    yv = y * valid

    z = 1
    z1, z2, z3 = indirect_cinematic((xv, yv, z), stack_zi=False)
    z1 = np.where(valid, z1, 0)
    z2 = np.where(valid, z2, 0)
    z3 = np.where(valid, z3, 0)

    dx, dy, dz = compute_jacobian_inverse(xv, yv, z, z1, z2, z3, z_step, 0, 0)
    dx = np.where(valid, dx, 0)
    dy = np.where(valid, dy, 0)
    dz = np.where(valid, dz, 0)

    # Plot dxi
    for dxi, axis in (dx, 'x'), (dy, 'y'), (dz, 'z'):
        figure = plt.figure()
        axe = plt.subplot(111, aspect='equal')
        axe.set_title("{}/dt".format(axis))
        axe.grid()
        # Fixme: colormap
        image = axe.imshow(dxi, extent=(-X, X, -X, X), aspect='equal')
        plt.colorbar(image)
        patch0 = Circle((0, 0), R, fc="white", alpha=.1)
        for patch in patch0,:
            axe.add_artist(patch)
        axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
                 (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
                 'o')

####################################################################################################

def plot_zi():

    X = int(1.1*R)
    number_of_points = (2*X + 1)*2/10
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)
    valid = compute_valid_region(x, y)
    xv = x * valid
    yv = y * valid
    z1, z2, z3 = indirect_cinematic((xv, yv, 0), stack_zi=False)
    z1 = np.where(valid, z1, 0)
    z2 = np.where(valid, z2, 0)
    z3 = np.where(valid, z3, 0)

    # Plot z1 contour
    figure = plt.figure()
    axe = plt.subplot(111, aspect='equal')
    axe.grid()
    axe.set_title('z1')
    number_of_lines = 50
    CS = axe.contour(x, y, z1, number_of_lines)
    axe.clabel(CS, fontsize=9, inline=1)
    patches = []
    patches.append(Circle((0, 0), R, fc="white", alpha=.5))
    for p in Axy1, Axy2, Axy3:
        for s in -1, 1:
            patches.append(Circle(tuple(p*s), r, fc="red", alpha=.5))
    for patch in patches:
        axe.add_artist(patch)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')

    # Plot zi contours
    figure = plt.figure()
    axe = plt.subplot(111, aspect='equal')
    axe.grid()
    axe.set_title('z1')
    number_of_lines = 25
    CS = axe.contour(x, y, z1, number_of_lines)
    CS = axe.contour(x, y, z2, number_of_lines)
    CS = axe.contour(x, y, z3, number_of_lines)
    patch0 = Circle((0, 0), R, fc="white", alpha=.5)
    patch1 = Circle((rho, 0), r, fc="red", alpha=.5)
    patch2 = Circle((-rho, 0), r, fc="red", alpha=.5)
    for patch in patch0, patch1, patch2:
        axe.add_artist(patch)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')

####################################################################################################

def plot_delta():

    # Fixme: y -> -y ???

    z = 0

    # Define the XY mesh
    # X = R
    X = int(.8*R)
    # number_of_points = (2*X + 1)*2/10
    number_of_points = 300
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)

    # Compute the valid region
    valid = compute_valid_region(x, y, radius=R)
    xv = x * valid
    yv = y * valid

    # Plot the valid region
    if False:
        figure = plt.figure()
        axe = plt.subplot(111, aspect='equal')
        axe.grid()
        axe.imshow(valid, extent=(-X, X, -X, X), aspect='equal', cmap=cm.Blues)

    # Compute zi
    z1, z2, z3 = indirect_cinematic((xv, yv, z), stack_zi=False)

    # Apply a step
    z1 += z_step
    # z2 += z_step
    #z3 += z_step

    # Correct for valid region
    z1 = np.where(valid, z1, z0)
    z2 = np.where(valid, z2, z0)
    z3 = np.where(valid, z3, z0)

    # Stack zi
    z1.shape = number_of_points**2, 1
    z2.shape = number_of_points**2, 1
    z3.shape = number_of_points**2, 1
    zi = np.hstack((z1, z2, z3))

    # Compute the direct cinematic for zi
    p = direct_cinematic(zi)
    p.shape = number_of_points, number_of_points, 3
    px = p[:,:,0]
    py = p[:,:,1]
    pz = p[:,:,2]

    # Compute dxi
    dx = px - xv
    dy = py - yv
    dz = pz - z
    print 'dx:', np.min(dx), np.max(dx)
    print 'dy:', np.min(dy), np.max(dy)
    print 'dz:', np.min(dz), np.max(dz)
    if False:
        # Check dxi = 0 when any step is applied
        assert(np.all(dx < 1e-6))
        assert(np.all(dy < 1e-6))
        assert(np.all(dz < 1e-6))

    # Correct for valid region
    dx *= valid
    dy *= valid
    dz *= valid

    # dy *= -1

    # Plot dxi
    for di, axis in (dx, 'x'), (dy, 'y'), (dz, 'z'):
        di *= 1000 # um
        figure = plt.figure()
        # axe = plt.subplot(111)
        axe = plt.subplot(111, aspect='equal')
        axe.grid()
        axe.set_title('delta ' + axis)
        # CS = axe.contourf(xv, yv, di, 50)
        # axe.clabel(CS, fontsize=9, inline=1)
        # plt.colorbar(CS)
        image = axe.imshow(di, extent=(-X, X, -X, X), aspect='equal')
        plt.colorbar(image)
        patch0 = Circle((0, 0), R, fc="white", alpha=0)
        axe.add_artist(patch0)
        axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
                 (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
                 'o')

####################################################################################################

def compute_angles(x, y, z):

    N = x.shape[0]
    x.shape = N**2
    y.shape = N**2

    z1, z2, z3 = indirect_cinematic((x, y, z), stack_zi=False)

    # p1 = zi_to_pi(Ax1, Ay1, z1 - z)
    # p = zi_to_pi(x, y, 0)

    p1 = np.zeros((N**2, 3))
    p1[:,0] = Ax1
    p1[:,1] = Ay1
    # p1[:,2] = 0
    p1[:,2] = z1 - z
    p = np.zeros((N**2, 3))
    p[:,0] = x
    p[:,1] = y
    p[:,2] = 0
    o = np.zeros((N**2, 3))
    o[:,0] = x
    o[:,1] = 0
    o[:,2] = 0
    v1 = o - p1
    v2 = p - p1

    # |v1 x v2| = |v1| |v2| |sin n|
    sin_alpha1 = norm(vcross(v1, v2)) / (norm(v1)*norm(v2))
    # sin_alpha1 = np.abs(y) / L
    # sin_alpha1.shape = N, N

    # alpha1 = np.acos(np.sqrt(Vx1**2 + Vy1**2) / L)
    # sin_alpha1 = norm(vcross(v1, v2)) / L**2 # (norm(-p1)*norm(p-p1))
    # sin_alpha1 = norm(vcross(p, p1)) / L**2
    alpha1 = np.degrees(np.arcsin(sin_alpha1))

    # beta1 = np.degrees(np.arcsin(z1/L))
    # beta1.shape = N, N

    output = alpha1
    output.shape = N, N
    print output.min(), output.max()

    return output

####################################################################################################

def plot_angles():

    z = 0

    # Define the XY mesh
    # X = R
    X = int(.8*R)
    # number_of_points = (2*X + 1)*2/10
    number_of_points = 300
    x = np.linspace(-X, X, number_of_points)
    x, y = np.meshgrid(x, x)

    # Compute the valid region
    valid = compute_valid_region(x, y, radius=R)
    xv = x * valid
    yv = y * valid

    alpha1 = compute_angles(xv, yv, z)

    # Correct for valid region
    alpha1 = np.where(valid, alpha1, 0)

    figure = plt.figure()
    # axe = plt.subplot(111)
    axe = plt.subplot(111, aspect='equal')
    axe.grid()
    axe.set_title('alpha 1')
    image = axe.imshow(alpha1, extent=(-X, X, -X, X), aspect='equal')
    plt.colorbar(image)
    patch0 = Circle((0, 0), R, fc="white", alpha=0)
    axe.add_artist(patch0)
    axe.plot((0, Ax1, Ax2, Ax3, Px1, Px2, Px3),
             (0, Ay1, Ay2, Ay3, Py1, Py2, Py3),
             'o')
    
####################################################################################################

#plot_workspace_circles()
#plot_workspace()
#plot_jacobian_determinant()
#plot_jacobian()
#plot_jacobian_inverse()
#plot_zi()
#plot_delta()
plot_angles()
plt.show()

####################################################################################################
#
# End
#
####################################################################################################
