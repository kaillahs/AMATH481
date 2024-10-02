import numpy as np 

# Define the function and its derivative
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

# Newton-Raphson method
def newton_raphson(x0, tolerance=1e-6, max_iterations=1000):
    xn = x0
    for n in range(max_iterations):
        x_vals.append(float(xn))
        
        if abs(f(xn)) < tolerance:
            return xn, n+1  # Return root and number of iterations
        xn = xn - f(xn) / df(xn)
    return None, max_iterations  # Did not converge

x0 = -1.6
x_vals = []
root, iterations = newton_raphson(x0)
if root is not None:
    print(f"Newton-Raphson method: Root found at x = {root:.6f} after {iterations} iterations")
else:
    print("Newton-Raphson method did not converge")

A1 = x_vals

#Bisection Method

def bisection(xl,xr, tolerance = 1e-6, max_iterations=1000):
   for j in range (1000):
    xc = (xl + xr)/2
    fc = f(xc)
    
    bisection_x.append(xc)
    
    if (fc > 0):
        xl = xc
    else:
        xr = xc
    
    if (abs(fc) < 1e-6):
        return xc,j+1

xl = -0.7
xr = -0.4
bisection_x = []

rootB, iterationsB = bisection(xl,xr)

print(f"Bisection method: Root found at x = {rootB:.6f} after {iterationsB} iterations")
A2 = bisection_x

A3 = [iterations,iterationsB]

import numpy as np

# Define the matrices and vectors
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B

A5 = (3 * x) - (4 * y)

A6 = A @ x

A7 = B @ (x - y)

A8 = D @ x

A9 = (D @ y) + z

A10 = A @ B

A11 = B @ C

A12 = C @ D
