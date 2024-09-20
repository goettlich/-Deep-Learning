import numpy as np

def rk4_step(f, y, t, dt=0.01):
    """
    f can be either the RHS of a diffEq, or e.g. a torch Network().
    Why dependent on time? To cope with systems that are not time-invariant.
    - Neural Network: Rain prediction could change based on season.
    - Flow field: Acceleration field could be variable in time (e.g. oscillating)
    """
    k1 = f(y, t)
    k2 = f(y + dt*k1/2, t + dt/2)
    k3 = f(y + dt*k2/2, t + dt/2)
    k4 = f(y + dt*k3, t + dt)
    
    return y + dt/6* (k1 + 2*k2 + 2*k3 + k4)

def rk4_solve(f, y0, t=[0,1], dt=0.01):
    
    idx=1
    t_now = t[0]
    y = np.zeros(shape=(len(t),len(y0)))
    y[0], y_now = y0, y0
    
    while idx < len(t):
        
        dt_step = min(dt, t[idx]-t_now)
        y_now = rk4_step(f, y_now, t_now, dt_step)
        t_now += dt_step

        if abs(t_now - t[idx]) < 1e-8:
            y[idx] = y_now
            idx += 1
            
    return y
    
def ODESolve(f, y0, t=[0,1], method='rk4', dt=0.01):

    if method=='rk4':
        return rk4_solve(f, y0, t, dt)
    



