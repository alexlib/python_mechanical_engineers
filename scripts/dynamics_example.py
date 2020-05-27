
import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

# ------ Parameters ------
m1 = 1
m2 = 1
I1 = I2 = 0.5
l1 = 1
lc1 = l1/2
l2 = 1
lc2 = l2/2
g = -9.81
n = 2

# ------ Dynamic matrices ------

def M_matrix(q):
    M = np.zeros((2,2))
    M[0,0] = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2*l1*lc2*np.cos(q[1])) + I1 + I2
    M[0,1] = M[1,0] = m2 * (lc2**2 + l1*lc2*np.cos(q[1])) + I2
    M[1,1] = m2 * lc2**2 + I2
    return M

def C_matrix(q, dq):
    h = -m2 * l1 * lc2 * np.sin(q[1])
    C = np.array([[h*dq[1], h*dq[1] + h*dq[0]], [-h*dq[0], 0]])
    return C

def G_vector(q):
    G = -np.array([(m1*lc1 + m2*l1)*g*np.cos(q[0]), m2*lc2*g*np.cos(q[0]+q[1])])
    return G

# ------ Direct kinematics ------

def direct_kinematics(q):
    return [l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]), l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1])]

# ------ ODE ------

def model(x, t, u):
    x1 = x[:2] # q
    x2 = x[2:] # dq

    invM = np.linalg.inv(M_matrix(x1))
    C = C_matrix(x1, x2)
    G = G_vector(x1)

    dx1 = x2
    dx2 = invM.dot(u - C.dot(x2) - G)

    dxdt = np.concatenate((dx1, dx2), axis = 0)

    return dxdt

# ------ Solve ------

t = np.linspace(0, 10, 1000)
x0 = np.array([0,0,0,0]).reshape((2*n,))
u = np.array([1, 1])

Q = odeint(model, x0, t, args=(u,))
X = np.array([direct_kinematics(q) for q in Q])

# ------ Plot ------

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.plot(t, np.rad2deg(Q[:,:2]))
ax1.set_title('Angles')
ax1.legend(('q1','q2'))
ax1.set_xlabel('t (sec)')
ax1.set_ylabel('q (deg)')
ax1.set_xlim([0, np.max(t)])

ax2.plot(t, Q[:,2:])
ax2.set_title('Angular velocity')
ax2.legend(('w1','w2'))
ax2.set_xlabel('t (sec)')
ax2.set_ylabel('w (rad/sec)')
ax2.set_xlim([0, np.max(t)])
plt.show()

# ------ Animate ------

fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
lines, = ax.plot([], [], lw=2)
points, = ax.plot([], [], 'ok')
path, = ax.plot([], [], ':', lw=1)

def init():
    lines.set_data([], [])
    points.set_data([], [])
    path.set_data([], [])
    return lines, points, path,

# animation function.  This is called sequentially
def animate(i):
    q = Q[i,:2]
    p = np.array([[0, 0], [l1*np.cos(q[0]), l1*np.sin(q[0])], direct_kinematics(q)])

    lines.set_data(p[:,0], p[:,1])
    points.set_data(p[:,0], p[:,1])
    path.set_data(X[:i,0], X[:i,1])
    return lines, points, path,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Q.shape[0], interval=20, blit=False)
# anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
