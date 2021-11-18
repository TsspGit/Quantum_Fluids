__author__ = "@Tssp"
__date__ = "18/10/21"
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, ifft, ifft2, fftfreq, fftshift

#----------------------------------------- 1D -----------------------------------------#

class GPE_Solver:
    """
    Solver of the 1D non-linear Schrödinger equation (Gross-Pitaevskii, GPE) 
    using the Split Step Fourier method for a given potential V(x)
    """
    def __init__(self, x, V, psi0, k, hbar=1, m=1):
        """
        Parameters
        ----------
        x : array_like, float
            length-N array of evenly spaced spatial coordinates
        psi_x0 : array_like, complex
            length-N array of the initial wave function at time t0
        V_x : array_like, float
             length-N array giving the potential at each x
        k    : array_like, float,
            wavenumber
        hbar : float
            value of planck's constant (default = 1)
        m : float
            particle mass (default = 1)
        """
        self.x, self.psi0, self.k, self.V = map(np.asarray, (x, psi0, k, V))
        N = self.x.size
        self.N = N
        # Validation of array inputs
        assert self.x.shape    == (self.N,)
        assert self.psi0.shape == (self.N,)
        assert self.V.shape   == (self.N,)
        
        # Set internal parameters
        self.hbar = hbar
        self.m    = m
        self.dX   = 2*self.x[-1]/N # L/N
        print(f'''GPE Solver initialized for a GRID from {np.real(self.x[0])} to {np.real(self.x[-1])} 
hbar={self.hbar} 
m={self.m}
In order to see the potential and the initial wavefunction we highly recommend to run:
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, GPE.abs_square(psi0))
ax.plot(x, V, 'k')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$|\psi_0(x)|^2$')
        ''')
        
    def abs_square(self, psi):
        """
        Returns the absolute square of a complex array.

        Parameters
        ----------
        arr (np.array): Input array

        Returns
        -------
        np.array: Absolute square of input
        """
        return np.conjugate(psi)*psi

    def V_NL(self, psi, g):
        """Returns the nonlinear potential of the GPE equation

        Parameters
        ----------
        psi (np.array): State
        g  (np.float): coupling term

        Returns
        -------
        np.array: Nonlinear part of the potential
        """
        return g*self.abs_square(psi)

    
    def time_evolution(self, num_steps, dt, V, g):
        """
        Calculates the GPE time evolution of the probability density given an
        initial state using the split-step fourier method.

        Parameters
        ----------
        psi0 (np.array): Initial state
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
        V (array float): Interaction potential
        g  (np.float): coupling term

        Returns
        -------
        Probability density for all iteration times; states are sorted in
        columns in temporal order.
        """
        prob_densities = np.zeros((num_steps+1, self.N))
        psi_k          = np.zeros((num_steps+1, self.N), dtype=complex)
        psi_x          = np.zeros((num_steps+1, self.N), dtype=complex)
        psi            = self.psi0
        psi_x[0]       = psi
        psi_k[0]       = fft(psi)
        prob_densities[0] = np.real(self.abs_square(psi))

        for i in range(num_steps):
            psi = fft(psi)
            psi = np.exp(-1j*dt*self.k**2/2)*psi
            psi_k[i+1] = psi
            psi = ifft(psi)
            psi = np.exp(-1j*dt*(self.V_NL(psi, g) + self.V))*psi
            psi_x[i+1] = psi
            prob_densities[i+1] = np.real(self.abs_square(psi))

        out = {'psi_x': psi_x,
              'psi_k': psi_k,
              'prob_densities': prob_densities}

        return out
        
    def run_and_plot(self, num_steps, dt, g, file_name):
        """
        Runs the model given an initial state and plots the temporal
        development of the probability density.

        Parameters
        ----------
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
        g  (np.float): coupling term
        file_name (str): Figure title
        """
        dic = self.time_evolution(num_steps, dt, self.V, g)
        self.prob_densities = dic['prob_densities']
        self.psi_x = dic['psi_x']
        self.psi_k = dic['psi_k']

        plt.imshow(self.prob_densities, cmap=plt.get_cmap("BuPu"), origin='lower', 
                   extent=[self.x[0], self.x[-1]-self.dX, dt/2, (num_steps+1/2)*dt], aspect='auto')
        plt.xlabel('$x$ $[\\xi]$')
        plt.ylabel('$t$ $[\\xi/c_s]$')
        cbar = plt.colorbar()
        cbar.set_label(r'$|\psi_0(x)|^2$', labelpad=20)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()

#----------------------------------------- 2D -----------------------------------------#

class GPE_Solver_2D:
    """
    Solver of the 1D non-linear Schrödinger equation (Gross-Pitaevskii, GPE) 
    using the Split Step Fourier method for a given potential V(x)
    """
    def __init__(self, x, y, V, psi0, kx, ky, hbar=1, m=1):
        """
        Parameters
        ----------
        x : array_like, float
            length-N array of evenly spaced spatial coordinates
        psi_x0 : array_like, complex
            length-N array of the initial wave function at time t0
        V_x : array_like, float
             length-N array giving the potential at each x
        k    : array_like, float,
            wavenumber
        hbar : float
            value of planck's constant (default = 1)
        m : float
            particle mass (default = 1)
        """
        self.x, self.y, self.psi0, self.kx, self.ky, self.V = map(np.asarray, (x, y, psi0, kx, ky, V))
        Nx = self.x.shape[0]
        Ny = self.y.shape[0]
        self.Nx = Nx
        self.Ny = Ny
        # Validation of array inputs
        assert self.x.shape    == (self.Nx,self.Ny)
        assert self.psi0.shape == (self.Nx,self.Ny)
        assert self.V.shape   == (self.Nx,self.Ny)
        
        # Set internal parameters
        self.hbar = hbar
        self.m    = m
        self.dX   = 2*self.x[-1]/Nx # L/N
        self.dY   = 2*self.y[-1]/Ny
        print(f'''
        GPE Solver initialized for a GRID from x={np.real(self.x[0][0])} to x={np.real(self.x[0][-1])} and 
        y={np.real(self.y[0][0])} to y={np.real(self.y[0][-1])}
        hbar={self.hbar} 
        m={self.m}
        ''')
        
    def abs_square(self, psi):
        """
        Returns the absolute square of a complex array.

        Parameters
        ----------
        arr (np.array): Input array

        Returns
        -------
        np.array: Absolute square of input
        """
        return np.conjugate(psi)*psi

    def V_NL(self, psi, g):
        """Returns the nonlinear potential of the GPE equation

        Parameters
        ----------
        psi (np.array): State
        g  (np.float): coupling term

        Returns
        -------
        np.array: Nonlinear part of the potential
        """
        return g*self.abs_square(psi)

    
    def time_evolution(self, num_steps, dt, V, g):
        """
        Calculates the GPE time evolution of the probability density given an
        initial state using the split-step fourier method.

        Parameters
        ----------
        psi0 (np.array): Initial state
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
        V (array float): Interaction potential
        g  (np.float): coupling term

        Returns
        -------
        Probability density for all iteration times; states are sorted in
        columns in temporal order.
        """
        prob_densities        = np.zeros((num_steps+1, self.Nx, self.Ny))
        psi_x                 = np.zeros((num_steps+1, self.Nx, self.Ny), dtype=complex)
        psi                   = self.psi0
        psi_x[0,:,:]          = psi
        prob_densities[0,:,:] = np.real(self.abs_square(psi))

        for i in range(num_steps):
            psi                     = fft2(psi)
            psi                     = np.exp(-1j*dt*(self.kx**2 + self.ky**2)/2)*psi
            psi                     = ifft2(psi)
            psi                     = np.exp(-1j*dt*(self.V_NL(psi, g) + self.V))*psi
            psi_x[i+1,:,:]          = psi
            prob_densities[i+1,:,:] = np.real(self.abs_square(psi_x[i+1,:,:]))

        out = {'prob_densities': prob_densities,
              'psi_x': psi_x}

        return out
        
    def run_and_plot(self, num_steps, dt, g):
        """
        Runs the model given an initial state and plots the temporal
        development of the probability density.

        Parameters
        ----------
        num_steps (int): Number of iteration steps
        dt (float, optional): Time step size
        g  (np.float): coupling term
        """
        dic = self.time_evolution(num_steps, dt, self.V, g)
        self.prob_densities = dic['prob_densities']
        self.psi_x = dic['psi_x']