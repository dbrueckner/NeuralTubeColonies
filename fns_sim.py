import numpy as np
from numba import jit

@jit(nopython=True)
def LogisticSigmoid(r,rad):
    x = 100*(rad-r)
    return 1/(1+np.exp(-x))

@jit(nopython=True)
def gaussianfn(r,rad,sigma):
    return np.exp(-(r-rad)**2/(2*sigma**2))

@jit(nopython=True)
def activation(x,K,h):
    return x**h/(K**h+x**h)

@jit(nopython=True)
def inhibition(x,K,h):
    return K**h/(K**h+x**h)


@jit(nopython=True)
def func_sim(params,modes,mode_perturb,t_nogKO=0):
    
    #Neumann BC:
    def func_diff(c,j,r,N_cells,dx):
        if j==0:
            c_2nd = ( 2*c[j+1]-2*c[j] )/dx**2
        elif j==N_cells-1:
            c_2nd = ( -2*c[j]+2*c[j-1] )/dx**2
        else:
            c_2nd = ( c[j+1]-2*c[j]+c[j-1] )/dx**2 + ( c[j+1]-c[j-1] )/(2*dx*r)
        return c_2nd
    
    (D_b,D_n,D_w,actBbyS,degB,degN,degS,degW,actBbyL,degL,b0,A,sigma,rad,actLbyS,KhillS,HillS,KhillL,HillL,actLbyW,KHillN) = params
    (N_cells,N_t,delta_t,oversampling,dx) = modes
    
    dt = delta_t/oversampling
    N_t_oversampling = int(N_t*oversampling)
    N_var = 5
    X_all = np.zeros((N_cells,N_t,N_var))
    
    xx = np.arange(N_cells)*dx
    
    b_prev = b0*np.ones(N_cells)
    n_prev = np.zeros(N_cells)
    w_prev = np.zeros(N_cells)
    s_prev = np.zeros(N_cells)
    l_prev = np.zeros(N_cells)
    
    b_next = np.zeros(N_cells)
    n_next = np.zeros(N_cells)
    w_next = np.zeros(N_cells)
    s_next = np.zeros(N_cells)
    l_next = np.zeros(N_cells)
    
    count_t = 0
    for t in range(0,N_t_oversampling):
        
        for j in range(0,N_cells):
        
            r = xx[j]
            b_2nd = func_diff(b_prev,j,r,N_cells,dx)
            n_2nd = func_diff(n_prev,j,r,N_cells,dx)
            w_2nd = func_diff(w_prev,j,r,N_cells,dx)

            b_next[j] = np.abs(b_prev[j] + ( D_b*b_2nd + (actBbyS*s_prev[j] + actBbyL*l_prev[j] )*LogisticSigmoid(r,rad) - degB*b_prev[j])*dt)

            n_next[j] = np.abs(n_prev[j] + ( D_n*n_2nd + s_prev[j]*LogisticSigmoid(r,rad) - degN*n_prev[j])*dt)
            
            if A>0:
                radialfac = (1+A*gaussianfn(r,rad,sigma))/A
            elif A==0:
                radialfac = 1
            s_next[j] = np.abs(s_prev[j] + ( ( b_prev[j]**HillS/((n_prev[j]/KHillN)**HillS+b_prev[j]**HillS) )*LogisticSigmoid(r,rad)*( radialfac ) - degS*s_prev[j])*dt)

            w_next[j] = np.abs(w_prev[j] + ( D_w*w_2nd + ( l_prev[j] ) *LogisticSigmoid(r,rad) - degW*w_prev[j])*dt)
  
            l_next[j] = np.abs(l_prev[j] + ( ( actLbyS*s_prev[j] + actLbyW*activation(s_prev[j],KhillS,HillS)*activation(w_prev[j],KhillL,HillL) )*LogisticSigmoid(r,rad) - degL*l_prev[j])*dt)
            
            
        b_prev = b_next
        n_prev = n_next
        w_prev = w_next
        s_prev = s_next
        l_prev = l_next
    
        if(np.mod(t,oversampling)==0):
            X_all[:,count_t,0] = b_prev
            X_all[:,count_t,1] = n_prev
            X_all[:,count_t,2] = s_prev
            X_all[:,count_t,3] = l_prev
            X_all[:,count_t,4] = w_prev
            count_t += 1
            
    return X_all
