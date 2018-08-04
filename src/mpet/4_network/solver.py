from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import get_source_terms as st
import os

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Refinement methods
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

# Remove logging
set_log_level(30)

def solver(N, T, dt, u_s, p_s, fs, gs, params):
	"""
	Solver for a 2D MPET 4-network 2-field formulation using a manufactured 
	solution with Taylor-Hood elements. 

		-div(sigma(u)) + alpha1*grad(p1) +.. + alpha4*grad(p4) = f
			 c1*p1_t + alpha1*div(u_t) - K1*div(grad(p1)) + S1 = g1
			 c2*p2_t + alpha2*div(u_t) - K2*div(grad(p2)) + S2 = g2
			 c3*p3_t + alpha3*div(u_t) - K3*div(grad(p3)) + S3 = g3
			 c4*p4_t + alpha4*div(u_t) - K4*div(grad(p4)) + S4 = g4

	on the unit square with Dirichlet BC's, where 

			S1 = xi1*(p1-p2) + xi2*(p1-p3) + xi3*(p1-p4)
			S2 = xi1*(p2-p1) + xi4*(p2-p3) + xi5*(p2-p4)
			S3 = xi2*(p3-p1) + xi4*(p3-p2) + xi6*(p3-p4)
			S4 = xi3*(p4-p1) + xi5*(p4-p2) + xi6*(p4-p3)

	and xi = (xi_{1->2},  xi_{1->3}, xi_{1->4}, xi_{2->3}, xi_{2->4}, xi_{4->3})
	
	"""

	# Unit square mesh
	mesh = UnitSquareMesh(N,N)

	P2 = VectorElement("CG", mesh.ufl_cell(), 2)	# P2 element
	P1 = FiniteElement("CG", mesh.ufl_cell(), 1)	# P1 element
	TH = MixedElement([P2,P1,P1,P1,P1])				# Taylor-Hood mixed element
	W = FunctionSpace(mesh, TH) 					# mixed function space for all 4 networks
	V = W.sub(0).collapse()
	Q = W.sub(1).collapse()

	# Parameters
	alpha = params["alpha"]				# Biot-Willis coefficient
	lamb = Constant(params["lamb"])		# Lame parameter
	mu = Constant(params["mu"])			# Lame parameter
	c = params["c"]						# storage coeffient for each network
	K = params["K"]						# permeability for each network
	xi = params["xi"]					# transfer coefficient for each network
	dt_ = Constant(dt)					# avoid recompiling each time we solve
	time = Constant(0.0) 				# time-loop updater

	# Functions
	(u, p1, p2, p3, p4) = TrialFunctions(W)
	(v, q1, q2, q3, q4) = TestFunctions(W)
	
	# Initial functions
	up0 = Function(W)
	(u0, p1_0, p2_0, p3_0, p4_0) = split(up0)

	# Source terms
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha1=alpha[0], alpha2=alpha[1], alpha3=alpha[2], alpha4=alpha[3], t=time, degree=5)
	g1 = Expression(gs[0], alpha1=alpha[0], K1=K[0], c1=c[0], xi1=xi[0], xi2=xi[1], xi3=xi[2], t=time, degree=5)
	g2 = Expression(gs[1], alpha2=alpha[1], K2=K[1], c2=c[1], xi1=xi[0], xi4=xi[3], xi5=xi[4], t=time, degree=5)
	g3 = Expression(gs[2], alpha3=alpha[2], K3=K[2], c3=c[2], xi2=xi[1], xi4=xi[3], xi6=xi[5], t=time, degree=5)
	g4 = Expression(gs[3], alpha4=alpha[3], K4=K[3], c4=c[3], xi3=xi[3], xi5=xi[4], xi6=xi[5], t=time, degree=5)

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), t=time, degree=5)
	p1_e = Expression(p_s[0], t=time, degree=5)
	p2_e = Expression(p_s[1], t=time, degree=5)
	p3_e = Expression(p_s[2], t=time, degree=5)
	p4_e = Expression(p_s[3], t=time, degree=5)

	# Stress tensor
	def sigma(u):
		return 2.0*mu*sym(grad(u)) + lamb*tr(sym(grad(u)))*Identity(len(u))

	# What happens on the boundary?
	def boundary(x):
		return x[0] < DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0-DOLFIN_EPS 
		
	# Boundary conditions
	u_bc = DirichletBC(W.sub(0), u_e, boundary)
	p1_bc = DirichletBC(W.sub(1), p1_e, boundary)
	p2_bc = DirichletBC(W.sub(2), p2_e, boundary)
	p3_bc = DirichletBC(W.sub(3), p3_e, boundary)
	p4_bc = DirichletBC(W.sub(4), p4_e, boundary)
	bc = [u_bc, p1_bc, p2_bc, p3_bc, p4_bc]

	# Transfer parameters
	S1 = xi[0]*(p1-p2) + xi[1]*(p1-p3) + xi[2]*(p1-p4)
	S2 = xi[0]*(p2-p1) + xi[3]*(p2-p3) + xi[4]*(p2-p4)
	S3 = xi[1]*(p3-p1) + xi[3]*(p3-p2) + xi[5]*(p3-p4)
	S4 = xi[2]*(p4-p1) + xi[4]*(p4-p2) + xi[5]*(p4-p3)

	# Variational problem
	a1 = (inner(sigma(u),grad(v)) - inner(alpha[0]*p1 + alpha[1]*p2 + alpha[2]*p3 + alpha[3]*p4,div(v)))*dx
	L1 = inner(f,v)*dx
	a2 = (c[0]*inner(p1,q1) + alpha[0]*inner(div(u),q1) + K[0]*dt_*inner(grad(p1),grad(q1)) + dt_*inner(S1,q1))*dx
	L2 = (dt_*inner(g1,q1) + c[0]*inner(p1_0,q1) + alpha[0]*inner(div(u0),q1))*dx 
	a3 = (c[1]*inner(p2,q2) + alpha[1]*inner(div(u),q2) + K[1]*dt_*inner(grad(p2),grad(q2)) + dt_*inner(S2,q2))*dx
	L3 = (dt_*inner(g2,q2) + c[1]*inner(p2_0,q2) + alpha[1]*inner(div(u0),q2))*dx 
	a4 = (c[2]*inner(p3,q3) + alpha[2]*inner(div(u),q3) + K[2]*dt_*inner(grad(p3),grad(q3)) + dt_*inner(S3,q3))*dx
	L4 = (dt_*inner(g3,q3) + c[2]*inner(p3_0,q3) + alpha[2]*inner(div(u0),q3))*dx 
	a5 = (c[3]*inner(p4,q4) + alpha[3]*inner(div(u),q4) + K[3]*dt_*inner(grad(p4),grad(q4)) + dt_*inner(S4,q4))*dx
	L5 = (dt_*inner(g4,q4) + c[3]*inner(p4_0,q4) + alpha[3]*inner(div(u0),q4))*dx 
	a = a1 + a2 + a3 + a4 + a5
	L = L1 + L2 + L3 + L4 + L5

	# Numerical solution
	up = Function(W)
	
	# Solve for each time step t
	t = dt
	while float(t) <= (T - 1.e-10):

		time.assign(t)

		# Solve variational problem
		solve(a==L, up, bc)

		# Update previous solution 
		up0.assign(up)

		t += dt

	# Numerical solutions
	(u_h, p1_h, p2_h, p3_h, p4_h) = up.split()

	p_h = [p1_h, p2_h, p3_h, p4_h]

	# Exact solution needs to match time level
	# u_e = interpolate(Expression((u_s[0], u_s[1]), t=t-dt, degree=5), V)
	# p1_e = interpolate(Expression(p_s[0], t=t-dt, degree=5), Q)
	# p2_e = interpolate(Expression(p_s[1], t=t-dt, degree=5), Q)
	# p3_e = interpolate(Expression(p_s[2], t=t-dt, degree=5), Q)
	# p4_e = interpolate(Expression(p_s[3], t=t-dt, degree=5), Q)

	p_e = [p1_e, p2_e, p3_e, p4_e]
	
	return u_h, p_h, u_e, p_e, mesh


def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = list(error.keys())
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print("rate %s = %.3f" % (key, rate))


def run_solver():
	
	# Input parameters
	T = 1.0
	alpha = (1.0, 1.0, 1.0, 1.0)
	mu = 0.5
	lamb = 1.0
	c = (1.0, 1.0, 1.0, 1.0)
	K = (1.0, 1.0, 1.0, 1.0)
	xi = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

	E = None
	nu = None
	#mu = E/(2.0*((1.0 + nu)))
	#lamb = nu*E/((1.0-2.0*nu)*(1.0+nu))

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*cos(pi*y)*sin(2*pi*t))" 	# string expression
	p_str = "(sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t), \
			sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t))" 		# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 								# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 									# get source terms

	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, c=c, K=K, xi=xi)

	# Error lists
	E_u  = []; E_p1 = []; E_p2 = []; E_p3 = []; E_p4 = []; h = []

	# Solve for each N
	for N in [4,8,16,32,64]:
		dt = (1./N)**2
		u_h, p_h, u_e, p_e, mesh = solver(N, T, dt, u_s, p_s, fs, gs, params)

		# Error norms
		u_error = errornorm(u_h, u_e, "H1", mesh=mesh)
		p1_error = errornorm(p_h[0], p_e[0], "L2", mesh=mesh)
		p2_error = errornorm(p_h[1], p_e[1], "L2", mesh=mesh)
		p3_error = errornorm(p_h[2], p_e[2], "L2", mesh=mesh)
		p4_error = errornorm(p_h[3], p_e[3], "L2", mesh=mesh)
		E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error); E_p3.append(p3_error); E_p4.append(p4_error)
		h.append(mesh.hmin())
		print("N = %.d, dt = %.3e, u_err: %.3e, p1_err: %.3e, p2_err: %.3e, p3_err: %.3e, p4_err: %.3e" % (N, dt, u_error, p1_error, p2_error, p3_error, p4_error))

	# Convergence rate
	error = dict(u=E_u, p1=E_p1, p2=E_p2, p3=E_p3, p4=E_p4)
	convergence_rate(error, h)

if __name__ == "__main__":
	run_solver()