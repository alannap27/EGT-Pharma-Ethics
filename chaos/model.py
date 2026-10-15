import numpy as np

#k = 0.7975000000000001
k = 0.6
B_r = 1.0
C_r = 0.1
B_p = 0.8977256300359548
C_p = 0.10167761086872712
C_base = 0.1
S_p = 0.32090249260668396 
alpha = 0.20778719376726207
gamma = 0.1

A = -C_base
B = S_p + alpha - k**3
C = B_r - C_r
D = -B_r
E = B_p - C_p
F = -B_p
W = gamma

def fun(x_vec,t):
    x,y,z = x_vec
    
    xdot = x*(1-x)*(A+B*y+W*z)
    ydot = y*(1-y)*(C+D*x)
    zdot = z*(1-z)*(E+F*x)
    return np.array([xdot,ydot,zdot])

def jac(x_vec,t):
    x,y,z = x_vec
    jac = np.array(
            [[(1-2*x)*(A+B*y+W*z),x*(1-x)*B,x*(1-x)*W],
             [y*(1-y)*D,(1-2*y)*(C+D*x),0],
             [z*(1-z)*F,0,(1-2*z)*(E+F*x)]])
    return jac
