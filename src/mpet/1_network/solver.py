from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import get_source_terms as st

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Remove logging
set_log_active(False)

"""
Solver for a Biot problem using a manufactured solution with Taylor-Hood elements, 

		 	- div(sigma(u)) + alpha*grad(p) = f
	   c*p_t + alpha*div(u_t) - K*div(grad(p)) = g

on the unit square.

Boundary conditions:

			u(., t) = u_e on Dirichlet boundary
			p(., t) = p_e on Dirichlet boundary

"""

def solver(N, T, dt, u_s, p_s, fs, gs, alpha, mu, lamb, c, K, mms_test=1):

	# Mesh
	mesh = UnitSquareMesh(N,N)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)		# displacement
	Q = FunctionSpace(mesh, "CG", 1)			# pressure
	V1 = VectorFunctionSpace(mesh, "CG", 3)		# displacement
	Q1 = FunctionSpace(mesh, "CG", 2)			# pressure
	W = MixedFunctionSpace([V,Q]) 				# mixed function space

	# Parameters
	alpha = Constant(alpha)			# Biot-Willis coefficient
	mu = Constant(mu)				# Lame parameter
	lamb = Constant(lamb) 			# Lame parameter
	c = Constant(c)					# compressibility
	K = Constant(K)					# mobility
	dt_ = Constant(dt)				# avoid recompiling each time we solve
	time = Constant(0.0) 			# time-loop updater

	# Functions
	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)

	# Initial functions
	up0 = Function(W)
	
	# Elements for exact solution
	deg_boost = 4 				
	P2 = VectorElement("CG", mesh.ufl_cell(), deg_boost, 2)
	P1 = FiniteElement("CG", mesh.ufl_cell(), deg_boost, 1)

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), element=P2, degree=deg_boost, t=time)
	p_e = Expression(p_s, element=P1, degree=deg_boost, t=time)

	# Source terms
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha=alpha, element=P2, degree=deg_boost, t=time)
	g = Expression(gs, alpha=alpha, K=K, c=c, element=P1, degree=deg_boost, t=time)

	# Initial conditions
	u_init = interpolate(u_e, W.sub(0).collapse())
	p_init = interpolate(p_e, W.sub(1).collapse())
	assign(up0.sub(0), u_init)
	assign(up0.sub(1), p_init)

	# Split initial function, and use as the previous solution
	u0 = split(up0)[0]
	p0 = split(up0)[1]

	# Stress tensor
	def sigma(u):
		return 2.0*mu*eps(u) + lamb*div(u)*Identity(len(u))

	# Symmetric gradient
	def eps(u):
		return sym(grad(u))

	# Boundary function
	def boundary(x):
		return x[0] < DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0-DOLFIN_EPS 
		
	# Boundary conditions
	u_bc = DirichletBC(W.sub(0), u_e, boundary)
	p_bc = DirichletBC(W.sub(1), p_e, boundary)
	bc = [u_bc, p_bc]

	# Variational problem
	a1 = (inner(sigma(u),eps(v)) - alpha*inner(p,div(v)))*dx
	L1 = inner(f,v)*dx
	a2 = (c*inner(p,q) + alpha*inner(div(u),q) + K*dt_*inner(grad(p),grad(q)))*dx
	L2 = (dt_*inner(g,q) + c*inner(p0,q) + alpha*inner(div(u0),q))*dx 
	a = a1 + a2
	L = L1 + L2

	# Numerical solution
	up = Function(W)

	# Solve for each time step t
	t = 0.0
	while t < T:

		# Update time step
		time.assign(t)

		# Solve variational problem
		solve(a==L, up, bc)

		# Update previous solution 
		up0.assign(up)

		# Update time step
		t += dt

	# Numerical solutions
	(u_h, p_h) = up.split()

	# Interpolate exact solutions to higher order function spaces?
	u_e = interpolate(u_e, V1)
	p_e = interpolate(p_e, Q1)
	
	return u_h, p_h, u_e, p_e, mesh


def run(mms_test=1):
	
	# Input parameters
	T = 1.0
	alpha = 1.0
	nu = 1./3
	E = 4./3
	mu = E/(2.0*((1.0 + nu)))
	lamb = nu*E/((1.0-2.0*nu)*(1.0+nu))
	c = 1.0
	K = 1.0
	
	# Get source terms using sympy
	if mms_test == 1:
		u_str = "(x*(x-1)*sin(pi*x)*cos(pi*y)*sin(pi*t), y*(y-1)*sin(pi*x)*cos(pi*y)*sin(pi*t))"	# string expression
		p_str = "sin(pi*x)*sin(pi*y)*sin(pi*t)" 

	if mms_test == 2:
		u_str = "(t*t*sin(pi*x)*sin(2*pi*y), t*sin(3*pi*x)*sin(4*pi*y))"
		p_str = "t*exp(1-t)*x*y*(1-x)*(1-y)"

	if mms_test == 3:
		u_str = "(0.0,0.0)"
		p_str = "x*y*(1-x)*(1-y)"

	u_s = st.str2expr(u_str); p_s = st.str2expr(p_str)	# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str)		# get source terms

	# Print parameter-info to terminal
	print "mu = %.2f, lamb = %.2f, c = %.e, K = %.e" % (mu, lamb, c, K)

	# Error lists
	E_u  = []; E_p = []; h = []

	# Solve for each N
	for N in [4,8,16,32]:
		dt = (1./N)**2
		u_h, p_h, u_e, p_e, mesh = solver(N, T, dt, u_s, p_s, fs, gs, alpha, mu, lamb, c, K, mms_test)

		# Error norms
		u_error = errornorm(u_e, u_h, "H1")
		p_error = errornorm(p_e, p_h, "L2")
		E_u.append(u_error); E_p.append(p_error)
		h.append(mesh.hmin())
		print "N = %.d, dt = %.1e, u_error: %.3e, p_error: %.3e" % (N, dt, u_error, p_error)

	# Convergence rate
	convergence_rate(E_u, E_p, h)

	# Solution plots
	viz_solution(u_h, "u_num")
	viz_solution(u_e, "u_exact")
	viz_solution(p_h, "p_num")
	viz_solution(p_e, "p_exact")

	# Error plots
	error_plots(h, E_u, E_p)

def convergence_rate(E_u, E_p, h):
	""" Compute convergence rates """
	for i in range(len(E_u)-1):
		r_u = np.log(E_u[i+1]/E_u[i])/np.log(h[i+1]/h[i])
		r_p = np.log(E_p[i+1]/E_p[i])/np.log(h[i+1]/h[i])
		print "r_u = %.3f, r_p = %.3f" % (r_u, r_p)
	
def viz_solution(u, plotname):
	""" Visualize solutions """
	filename = "fig/Biot_%s.pvd" %plotname
	file = File(filename)
	file << u

def error_plots(h, E_u, E_p):
	""" Error plots for u and p"""
	E_u = np.array(E_u); E_p = np.array(E_p)
	plt.loglog(h, E_u, label="u")
	plt.loglog(h, E_p, label="p")
	plt.legend(loc=2)
	plt.savefig("fig/Biot_error.png")

def run_test(mms_test=1):

	import time as tm
	print "================ Test %d ================" %mms_test
	t0 = tm.clock()
	run(mms_test=mms_test)
	t1 = tm.clock() - t0
	print "runtime: %.2f seconds" %t1


if __name__ == "__main__":
	run_test(mms_test=1)
	run_test(mms_test=2)
	run_test(mms_test=3)