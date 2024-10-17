import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot2(y, x, K, epsilon):
    return [y[1], (K*(x**2) - epsilon) * y[0]]

tol = 1e-4  # Define a tolerance level 
colors = ['r', 'b', 'g', 'c', 'm']  # Colors for plotting eigenfunctions
L = 4  # Boundary for y
yp = [-L, L] 
K=1
yshoot = np.arange(yp[0], yp[1] + 0.1, 0.1)

eigenvalues = []  # List to store eigenvalues
eigenfunctions = []  # List to store eigenfunctions

epsilon_start = 0.2  # Starting value for epsilon
for mode in range(1, 6):  # Begin mode loop for the first five eigenfunctions
    epsilon = epsilon_start  # Initial guess for epsilon
    y0 = [1, np.sqrt(L**2 - epsilon)]
    depsilon = 0.1  # Step size for epsilon
    
    for _ in range(1000):  # Convergence loop for epsilon
        y = odeint(shoot2, y0, yshoot, args=(K,epsilon)) 
        
        # Check convergence condition
        if abs((y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) - 0) < tol:
            print(f"Found eigenvalue for mode {mode}: {epsilon}")
            eigenvalues.append(epsilon)  # Save the eigenvalue
            break

        # Update epsilon based on boundary condition behavior
        if (-1) ** (mode + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    # Adjust epsilon start for the next eigenvalue
    epsilon_start = epsilon + 2  
    
    # Normalize eigenfunction
    norm = np.trapz(y[:, 0]**2, yshoot)
    eigenfunc_norm = y[:, 0] / np.sqrt(norm)
    eigenfunctions.append(eigenfunc_norm)  # Save the normalized eigenfunction

    # Plot the eigenfunction
    plt.plot(yshoot, eigenfunc_norm, colors[mode - 1], label=f"Mode {mode}")


# Output eigenvalues and eigenfunctions
eigenvalues = np.array(eigenvalues)
eigenfunctions = np.array(eigenfunctions)
abs_eigenfunctions = np.column_stack([np.abs(ef) for ef in eigenfunctions])

print("Eigenvalues:")
print(eigenvalues)
print(len(eigenvalues))
print("\nEigenfunctions:")
for col in range(abs_eigenfunctions.shape[1]):
    print(f"Mode {col+1} eigenfunction:")
    print(abs_eigenfunctions[:,col])
    #print(abs_eigenfunctions[:,col].shape)
    #print(abs_eigenfunctions.shape)
    #print(eigenvalues.shape)

A2 = eigenvalues
A1 = abs_eigenfunctions

print(A1.shape)
print(A2.shape)
# Show the plot
plt.title("Eigenfunctions of Quantum Harmonic Oscillator")
plt.xlabel("x")
plt.ylabel("Normalized |ψ(x)|")
plt.legend()
plt.grid(True)
plt.show()


