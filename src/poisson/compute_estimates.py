from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os

# Remove logging
set_log_active(False)

def solver(N, degree):
	"""
	Solver for Poisson problem to evaluate a posteriori error estimators. 

		- div(grad(u)) = f 

	on unit square with homogenous Dirichlet boundary conditions. 

	The a posteriori error estimates are evaluated in the H1 norm.

	"""

	# Mesh
	mesh = UnitSquareMesh(N,N) 	# square

	# Function spaces
	V = FunctionSpace(mesh, "CG", degree)
	W = FunctionSpace(mesh, "CG", degree+2)

	# Test and trial functions
	u = TrialFunction(V)
	v = TestFunction(V)

	# Exact solution
	u_e = Expression("sin(2*pi*x[1])") 

	# Source term
	f = Expression("4*pi*pi*sin(2*pi*x[1])")

	# Boundary condition
	bc = DirichletBC(V, u_e, "on_boundary")

	# Numerical solution
	u_h = Function(V)

	# Variational form
	a = inner(grad(u), grad(v))*dx
	L = f*v*dx

	# Solve variational form
	solve(a==L, u_h, bc)

	# Exact solution needs to interpolated into higher order function space
	u_e = interpolate(u_e, W)

	return u_h, u_e, f, mesh

def compute_estimator(u_h, u_e, f, mesh):

	""" Compute a posteriori error estimator (residual-based) for 
		the Poisson problem. """

	# Quantities needed for error estimator
	h = CellSize(mesh)
	h_E = MaxCellEdgeLength(mesh)
	n = FacetNormal(mesh)
	n_E = MaxFacetEdgeLength(mesh)

	# Create elementwise distributed function w 
	DG0 = FunctionSpace(mesh, "DG", 0)
	w = TestFunction(DG0)

	# Find average of f on each element 
	CR = FunctionSpace(mesh, "CR", 1) 	# Crouzeix-Raviart elements
	f_K = interpolate(f, CR)

	# Element and edge esiduals
	R_K = f + div(grad(u_h)) 		# Element residual
	R_E = -jump(grad(u_h),n)		# Edge residual

	# Residual norms
	R = w*h**2*dot(R_K,R_K)*dx
	J = avg(h)*avg(w)*dot(R_E,R_E)*dS
	R_S = f_K + div(grad(u_h)) 		
	RS = w*h**2*dot(R_S,R_S)*dx
	S = w*h**2*dot(f-f_K, f-f_K)*dx 
	E = R + J

	# Piecewise constant functions
	eta_R = Function(DG0)
	eta_J = Function(DG0)
	eta_RS = Function(DG0)
	eta_S = Function(DG0)
	err_ind = Function(DG0)

	# Assemble integrals
	assemble(R, tensor=eta_R.vector())
	assemble(J, tensor=eta_J.vector())
	assemble(RS, tensor=eta_RS.vector())
	assemble(S, tensor=eta_S.vector())
	assemble(E, tensor=err_ind.vector())
	
	# Extract vectors of residual
	eta_R = eta_R.vector()
	eta_J = eta_J.vector() 
	eta_S = eta_S.vector()
	eta_RS = eta_RS.vector()

	# A posteriori error estimator no.1
	est1 = np.sqrt(eta_R.sum() + eta_J.sum())

	# A posteriori error estimator no.2 (with source term)
	eta_RK = np.sqrt(eta_RS.sum() + eta_J.sum())
	est2 = np.sqrt(eta_RK**2 + eta_S.sum())

	return est1, est2, err_ind

def convergence_rates(e, h):
	""" Function that computes the convergence rate given the error 
		and discretization parameter """

	for i in range(len(e)-1):
		r = np.log(e[i+1]/e[i])/np.log(h[i+1]/h[i])
		print "r = %.3f" % r

def plot_error_magnitude(est, degree, N):
	""" Plot error magnitude for chosen N """
	plotname = "%d_est" % N
	cwd = os.getcwd()
	newpath = r"%s/fig/%s" % (cwd, degree)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)
	filename = "%s/poisson_%s.pvd" % (newpath, plotname)
	file = File(filename)
	file << est
		

def run():
	""" Numerical experiments:
	Evaluate norm of error and the a posteriori estimate 
	for [P1, P2, P3] elements in H1 norm.  """

	for degree in [1,2,3]:
		
		# Empty error lists
		e1 = []; e2 = []; e_u = []; h = []

		print "======== Degree: %d ========" %degree

		for N in [4,8,16,32,64]:

			# Solve for each N
			u_h, u_e, f, mesh = solver(N, degree)
			
			# Compare numerical and exact solution with 
			error = errornorm(u_e, u_h, "H1")

			# Compute error estimator
			est1, est2, err_ind = compute_estimator(u_h, u_e, f, mesh)

			# Print error to terminal
			print "N = %d, error norm: %.3e, est1: %.3e, est2: %.3e" %(N, error, est1, est2)

			# Save info to lists
			e1.append(est1); e2.append(est2)
			e_u.append(error); h.append(1./N)

			# Plot error magnitude 
			plot_error_magnitude(err_ind, degree, N)


		# Convergence rate for error estimate
		print "Convergence rate: u_error"
		convergence_rates(e_u, h)
		print "Convergence rate: est1"
		convergence_rates(e1, h)
		print "Convergence rate: est2"
		convergence_rates(e2, h)

		

if __name__ == "__main__":
	run()
		
