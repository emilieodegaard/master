from dolfin import *
import numpy as np
import os
import matplotlib.pyplot as plt
import get_source_terms as st

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Remove logging
set_log_active(False)

def elastic_stress(u, mu, lamb):
	""" Elastic stress tensor """
	return 2.0*mu*sym(grad(u)) + lamb*tr(sym(grad(u)))*Identity(len(u))

def solver(N, T, dt, u_s, p_s, fs, gs, params):
	"""
	Solver for a 2D Barrenblatt-Biot problem test stability of a posteriori error estimators.

		-div(sigma(u)) + alpha1*grad(p1) + alpha2*grap(p2) = f
	c1*p1_t + alpha1*div(u_t) - K1*div(grad(p1)) + xi1*(p1-p2) = g1
	c2*p2_t + alpha2*div(u_t) - K2*div(grad(p2)) + xi2*(p2-p1) = g2

	on the unit square.
 
	"""

	# Mesh
	mesh = UnitSquareMesh(N,N)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)		# displacement
	V1 = VectorFunctionSpace(mesh, "CG", 3)
	Q = FunctionSpace(mesh, "CG", 1)			# pressure network 1
	Q1 = FunctionSpace(mesh, "CG", 2)
	W = MixedFunctionSpace([V,Q,Q]) 			# mixed function space

	# Parameters
	alpha = params["alpha"]						# Biot-Willis coefficient for each network
	lamb = Constant(params["lamb"])				# Lame parameter
	mu = Constant(params["mu"])					# Lame parameter
	c = params["c"]								# storage coeffient for each network
	K = params["K"]								# permeability for each network
	xi = params["xi"]							# transfer coefficient for each network
	dt_ = Constant(dt)							# avoid recompiling each time we solve
	time = Constant(0.0) 						# time-loop updater

	# Functions
	(u, p1, p2) = TrialFunctions(W)
	(v, q1, q2) = TestFunctions(W)
	
	# Initial functions
	up0 = Function(W)
	(u0, p1_0, p2_0) = split(up0)

	# Source terms
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha1=alpha[0], alpha2=alpha[1], degree=3, t=time)
	g1 = Expression(gs[0], alpha1=alpha[0], K1=K[0], c1=c[0], xi1=xi[0], degree=3, t=time)
	g2 = Expression(gs[1], alpha2=alpha[1], K2=K[1], c2=c[1], xi2=xi[1], degree=3, t=time)
	g = [g1, g2]

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), degree=3, t=time)
	p1_e = Expression(p_s[0], degree=3, t=time)
	p2_e = Expression(p_s[1], degree=3, t=time)

	# Stress tensor
	sigma = lambda u: elastic_stress(u, mu, lamb)
	
	# Boundary function
	def boundary(x):
		return x[0] < DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0-DOLFIN_EPS 
		
	# Boundary conditions
	u_bc = DirichletBC(W.sub(0), u_e, boundary)
	p1_bc = DirichletBC(W.sub(1), p1_e, boundary)
	p2_bc = DirichletBC(W.sub(2), p2_e, boundary)
	bc = [u_bc, p1_bc, p2_bc]

	# Variational problem
	a1 = (inner(sigma(u),grad(v)) - alpha[0]*inner(p1,div(v)) - alpha[1]*inner(p2,div(v)))*dx
	L1 = inner(f,v)*dx
	a2 = (c[0]*inner(p1,q1) + alpha[0]*inner(div(u),q1) + K[0]*dt_*inner(grad(p1),grad(q1)) + dt_*xi[0]*inner((p1-p2),q1))*dx
	L2 = (dt_*inner(g1,q1) + c[0]*inner(p1_0,q1) + alpha[0]*inner(div(u0),q1))*dx
	a3 = (c[1]*inner(p2,q2) + alpha[1]*inner(div(u),q2) + K[1]*dt_*inner(grad(p2),grad(q2)) + dt_*xi[1]*inner((p2-p1),q2))*dx
	L3 = (dt_*inner(g2,q2) + c[1]*inner(p2_0,q2) + alpha[1]*inner(div(u0),q2))*dx
	a = a1 + a2 + a3
	L = L1 + L2 + L3

	# Numerical solution
	up = Function(W)
	
	# Solve for each time step t
	t = dt
	while float(t) <= (T - 1.e-10):

		# Assign current time step to all functions
		time.assign(t)

		# Solve variational problem
		solve(a==L, up, bc)

		# Update previous solution 
		up0.assign(up)

		# Update time step
		t += dt

	# Numerical solutions
	(u_h, p1_h, p2_h) = up.split()
	p_h = [p1_h, p2_h]

	# Exact solution needs to match time level
	u_e = interpolate(Expression((u_s[0], u_s[1]), t=t-dt, degree=3), V)
	p1_e = interpolate(Expression(p_s[0], t=t-dt, degree=3), Q)
	p2_e = interpolate(Expression(p_s[1], t=t-dt, degree=3), Q)
	p_e = [p1_e, p2_e]
	
	return u_h, p_h, u_e, p_e, mesh


def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = error.keys()
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print "rate %s = %.3f" % (key, rate)


def run_solver():
	""" Run default parameters, all set to 1. """

	# Parameters
	T = 0.1
	alpha = (1.0, 1.0)
	nu = None
	E = None
	mu = 0.5
	lamb = 1.0
	c = (1.0, 1.0)
	K = (1.0, 1.0)
	xi = (1.0, 1.0)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, nu=nu, E=E, c=c, K=K, xi=xi)
		
	# Print parameter-info to terminal
	print "alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu
	print "lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))" 		# string expression
	p_str = "(sin(pi*x)*cos(pi*y)*sin(2*pi*t), cos(pi*x)*sin(pi*y)*sin(2*pi*t))" 	# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 	# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 		# get source terms

	# Error lists
	E_u  = []; E_p1 = []; E_p2 = []; h = []
	
	# Solve for each N
	for N in [4,8,16,32,64]:
		dt = (1./N)**2
		u_h, p_h, u_e, p_e, mesh = solver(N, T, dt, u_s, p_s, fs, gs, params)

		# Compute error norms
		u_error = errornorm(u_h, u_e, "H1")
		p1_error = errornorm(p_h[0], p_e[0], "L2")
		p2_error = errornorm(p_h[1], p_e[1], "L2")

		# Append error to list
		E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error)
		h.append(mesh.hmin())

		# Print info to terminal
		print "N = %.d, dt = %.1e, u: %.3e, p1: %.3e, p2: %.3e" % (N, dt, u_error, p1_error, p2_error)


	# Convergence rate
	E = dict(u=E_u, p1=E_p1, p2=E_p2)
	convergence_rate(E, h)


if __name__ == "__main__":
	run_solver()
