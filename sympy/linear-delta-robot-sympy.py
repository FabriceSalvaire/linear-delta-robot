####################################################################################################
#
# ipython qtconsole linear-delta-robot-sympy.py
#
# %run linear-delta-robot-sympy.py
#
####################################################################################################

####################################################################################################

from sympy import (
    init_printing,
    symbols,
    pi,
    cos, sin,
    Eq, solve,
    diff,
    Matrix,
)
# from sympy.geometry import *

####################################################################################################

init_printing() # use_unicode=True

####################################################################################################

# Define some functions
def magnitude_square(x):
    return x.dot(x) 
 
####################################################################################################

x, y = symbols('x y')
z = symbols('z', positive=True)
z1, z2, z3 = symbols('z1 z2 z3', positive=True)
R, r, rho = symbols('R r rho', positive=True)
rho = symbols('rho', positive=True) # = R - r
L = symbols('L', positive=True)

# Nacelle coordinate
N = Matrix([x, y, z])
# N = Point(x, y, z) # Only two dimensional points currently supported

# Axis angle
theta = 2*pi/3

# Rotation matrix for each axis
R1 = Matrix([[1,0,0],
             [0,1,0],
             [0,0,1]])
 
R2 = Matrix([[cos(theta),-sin(theta),0],
             [sin(theta),cos(theta),0],
             [0,0,1]])
 
R3 = Matrix([[cos(theta),sin(theta),0],
             [-sin(theta),cos(theta),0],
             [0,0,1]])
 
# Ball joint Point
P1 = R1 * Matrix([rho, 0, z1])
P2 = R2 * Matrix([rho, 0, z2])
P3 = R3 * Matrix([rho, 0, z3])
 
# System of equations
Eq1 = Eq(L**2, magnitude_square(N-P1))
Eq2 = Eq(L**2, magnitude_square(N-P2))
Eq3 = Eq(L**2, magnitude_square(N-P3))
 
# Indirect Cinematic Zi functions
Z1 = solve(Eq1, z1)[1] # select zi > z part
Z2 = solve(Eq2, z2)[1]
Z3 = solve(Eq3, z3)[1]
 
# d z_i / d x_i
diff(Z1, x)
diff(Z1, y)
diff(Z2, x)
diff(Z2, y)
diff(Z3, x)
diff(Z3, y)
 
# z_i @origin
Z1.subs(x, 0).subs(y, 0)
Z2.subs(x, 0).subs(y, 0)
Z3.subs(x, 0).subs(y, 0)
 
# Direct Cinematic

# Resolve y
# Eq2 - Eq3 is not supported
Sy = solve(Eq2.args[1] - Eq3.args[1], y)[0].factor()
# why expand is required ?
A = Sy.expand().coeff(z, 1).simplify()
B = Sy.expand().coeff(z, 0).simplify()

# Resolve x
Sx = solve(Eq2.args[1] - Eq1.args[1], x)[0].subs(y, A*z+B)
C = Sx.expand().coeff(z, 1).simplify()
D = Sx.expand().coeff(z, 0).simplify()

# Resolve z
EqZ = Eq1.args[1].subs(y, A*z+B).subs(x, C*z+D) - L**2
E = EqZ.expand().coeff(z, 2)
F = EqZ.expand().coeff(z, 1)
G = EqZ.expand().coeff(z, 0)

e, f, g = symbols('e f g')
solve(Eq(L**2, e*z**2 + f*z + g), z)
 
# 
# /* Check z_i = Z */
# subst(z2, z3, A);
# subst(z2, z3, B);
# subst(z1, z2, subst(z2, z3, B));
# subst(z1, z2, subst(z2, z3, C));
# /* -> 0 */
# /* -> [z=z1-sqrt(L^2-rho^2),z=sqrt(L^2-rho^2)+z1] */
# 
# /* (-y^2-x^2+2*rho*x-rho^2) - expand(-(x-rho)^2-y^2); */
# /* (-y^2+sqrt(3)*rho*y-x^2-rho*x-rho^2) - expand(-(x+rho/2)^2-(y-sqrt(3)/2*rho)^2); */
# /* (-y^2-sqrt(3)*rho*y-x^2-rho*x-rho^2) - expand(-(x+rho/2)^2-(y+sqrt(3)/2*rho)^2); */

####################################################################################################
# 
# End
# 
####################################################################################################
