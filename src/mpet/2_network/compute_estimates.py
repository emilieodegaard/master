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
set_log_level(30)

def elastic_stress(u, mu, lamb):
	return 2.0*mu*sym(grad(u)) + lamb*tr(sym(grad(u)))*Identity(len(u))

def compute_estimates(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement):

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
	P2 = VectorElement("CG", mesh.ufl_cell(), 2)	# P2 element
	P1 = FiniteElement("CG", mesh.ufl_cell(), 1)	# P1 element
	TH = MixedElement([P2,P1,P1])					# Taylor-Hood mixed element
	W = FunctionSpace(mesh, TH) 					# mixed function space for all 4 networks
	V = W.sub(0).collapse()
	Q = W.sub(1).collapse()

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
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha1=alpha[0], alpha2=alpha[1], degree=5, t=time)
	g1 = Expression(gs[0], alpha1=alpha[0], K1=K[0], c1=c[0], xi1=xi[0], degree=5, t=time)
	g2 = Expression(gs[1], alpha2=alpha[1], K2=K[1], c2=c[1], xi2=xi[1], degree=5, t=time)
	g = [g1, g2]

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), degree=5, t=time)
	p1_e = Expression(p_s[0], degree=5, t=time)
	p2_e = Expression(p_s[1], degree=5, t=time)

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

	# Lists to compute error estimators
	e_u = []; e_p1 = []; e_p2 = []
	e_u_dt = []; e_p1_dt = []; e_p2_dt = []
	e_time = []; e_time_p1 = []; e_time_p2 = []; ts = []

	# Facetnormal needed for jump-function
	n = FacetNormal(mesh)

	# Initial residuals (at time = 0)
	RK_u0 = f + div(sigma(u0)) - alpha[0]*grad(p1_0) - alpha[1]*grad(p2_0)
	RE_u0 = alpha[0]*jump(p1_0,n) + alpha[1]*jump(p2_0,n) - jump(sigma(u0),n)
	RK_p1_0 = dt_*g1 - c[0]*(p1_0) - alpha[0]*div(u0) + dt_*K[0]*div(grad(p1_0)) - dt_*xi[0]*(p1_0-p2_0)
	RK_p2_0 = dt_*g2 - c[1]*(p2_0) - alpha[1]*div(u0) + dt_*K[1]*div(grad(p2_0)) - dt_*xi[1]*(p2_0-p1_0)
	RE_p1_0 = -K[0]*jump(grad(p1_0),n)
	RE_p2_0 = -K[1]*jump(grad(p2_0),n)
	R0 = [RK_u0, RE_u0, RK_p1_0, RE_p1_0, RK_p2_0, RE_p2_0]

	# Numerical solution
	up = Function(W)
	
	# Solve for each time step t
	t = dt
	while float(t) <= (T - 1.e-10):

		time.assign(t)

		# Solve variational problem
		solve(a==L, up, bc)

		# Compute space error indicators
		R, err_ind, err_mag = error_indicators(up, up0, f, g, dt, mesh, R0, params)

		# Sum over error indicators
		e_u.append(err_ind[0].vector().sum())
		e_p1.append(err_ind[1].vector().sum())
		e_p2.append(err_ind[2].vector().sum())
		e_u_dt.append(err_ind[3].vector().sum())
		e_time.append(err_ind[4].sum())

		# Update previous solution 
		up0.assign(up)

		# Update residuals
		R0 = R

		# Update time step
		t += dt		

	# Compute final error estimators
	e_space = [e_u, e_p1, e_p2, e_u_dt]
	e_time = np.array(e_time)
	eta = error_estimators(e_space, e_time, dt)

	# Numerical solutions
	(u_h, p1_h, p2_h) = up.split()
	p_h = [p1_h, p2_h]

	# Exact solution needs to match time level
	u_e = interpolate(Expression((u_s[0], u_s[1]), t=t-dt, degree=5), V)
	p1_e = interpolate(Expression(p_s[0], t=t-dt, degree=5), Q)
	p2_e = interpolate(Expression(p_s[1], t=t-dt, degree=5), Q)
	p_e = [p1_e, p2_e]

	# Plot error magnitude:
	if refinement == "space":
		plot_error_magnitude(err_mag, N, T, experiment, refinement)
	if refinement == "time":
		plot_error_magnitude(err_mag, dt, T, experiment, refinement)
	
	return u_h, p_h, u_e, p_e, eta, mesh

def error_indicators(up, up0, f, g, dt, mesh, R0, params):
	
	# Quantities needed to compute error estimators
	h = CellDiameter(mesh)
	n = FacetNormal(mesh)
	DG0 = FunctionSpace(mesh, "DG", 0) 	# space of pieciewise constant functions
	w = TestFunction(DG0) 				# element indicator test function

	# Parameters
	alpha = params["alpha"]
	lamb = Constant(params["lamb"])
	mu = Constant(params["mu"])	
	c = params["c"]
	K = params["K"]
	xi = params["xi"]

	# Define elastic stress
	sigma = lambda u: elastic_stress(u, mu, lamb)

	# Split solutions
	u0 = split(up0)[0]; p1_0 = split(up0)[1]; p2_0 = split(up0)[2]
	u_h = split(up)[0]; p1_h = split(up)[1]; p2_h = split(up)[2]
	
	# Extract previous residuals
	RK_u0 = R0[0]; RE_u0 = R0[1]; RK_p1_0 = R0[2]; RE_p1_0 = R0[3]
	RK_p2_0 = R0[4]; RE_p2_0 = R0[5]

	# Facet normal needed for jump-function
	n = FacetNormal(mesh)

	# Element and face residuals for u
	RK_u = f + div(sigma(u_h)) - alpha[0]*grad(p1_h) - alpha[1]*grad(p2_h) 		# element residual
	RK_u_dt = (RK_u - RK_u0)/dt
	RE_u = alpha[0]*jump(p1_h,n) + alpha[1]*jump(p2_h,n) - jump(sigma(u_h),n) 	# face residual
	RE_u_dt = (RE_u - RE_u0)/dt
	
	# Element and face residuals for p1 and p2
	RK_p1 = dt*g[0] - c[0]*(p1_h-p1_0) - alpha[0]*div(u_h-u0) + dt*K[0]*div(grad(p1_h)) - dt*xi[0]*(p1_h-p2_h)
	RK_p2 = dt*g[1] - c[1]*(p2_h-p2_0) - alpha[1]*div(u_h-u0) + dt*K[1]*div(grad(p2_h)) - dt*xi[1]*(p2_h-p1_h)
	RK_p1_dt = (RK_p1 - RK_p1_0)/dt
	RK_p2_dt = (RK_p2 - RK_p2_0)/dt
	RE_p1 = -K[0]*jump(grad(p1_h),n)
	RE_p2 = -K[1]*jump(grad(p2_h),n)
	RE_p1_dt = (RE_p1 - RE_p1_0)/dt
	RE_p2_dt = (RE_p2 - RE_p2_0)/dt

	residuals = [RK_u, RE_u, RK_p1, RE_p1, RK_p2, RE_p2]

	# Residual spatial error estimators
	R_u = w*(h**2)*RK_u**2*dx + avg(h)*avg(w)*RE_u**2*dS
	R_p1 = w*(h**2)*(RK_p1**2)*dx + avg(h)*avg(w)*(RE_p1**2)*dS
	R_p2 = w*(h**2)*(RK_p2**2)*dx + avg(h)*avg(w)*(RE_p2**2)*dS

	# Residual temporal error estimators
	R_u_dt = (dt**2)*(w*(h**2)*(RK_u_dt**2)*dx + avg(h)*avg(w)*(RE_u_dt**2)*dS)									

	# Time estimator
	R_p1_dt = w*grad(p1_h-p1_0)**2*dx
	R_p2_dt = w*grad(p2_h-p2_0)**2*dx

	# For plotting
	R_p = R_p1 + R_p2
	R_p_dt = R_p1_dt + R_p2_dt

	# Piecewise constant functions
	err_u = Function(DG0)
	err_p1 = Function(DG0)
	err_p2 = Function(DG0)
	err_u_dt = Function(DG0)
	err_p1_dt = Function(DG0)
	err_p2_dt = Function(DG0)
	err_p = Function(DG0) 			# for plotting
	err_p_dt = Function(DG0)		# for plotting

	# Space error indicators
	assemble(R_u, tensor=err_u.vector())
	assemble(R_p1, tensor=err_p1.vector())
	assemble(R_p2, tensor=err_p2.vector())
	assemble(R_u_dt, tensor=err_u_dt.vector())

	# For plotting
	assemble(R_p, tensor=err_p.vector())
	assemble(R_p_dt, tensor=err_p_dt.vector())

	# Time error indicators
	assemble(R_p1_dt, tensor=err_p1_dt.vector())
	assemble(R_p2_dt, tensor=err_p2_dt.vector())
	err_time = (K[0]+xi[0])*err_p1_dt.vector() + (K[1]+xi[1])*err_p2_dt.vector() 

	# All error indicators
	error_indicators = [err_u, err_p1, err_p2, err_u_dt, err_time]

	# Error magnitude for plotting
	error_magnitude = [err_u, err_p, err_u_dt, err_p_dt]

	return residuals, error_indicators, error_magnitude

def error_estimators(e_space, e_time, dt):

	# Info for time estimator
	tau = np.ones(len(e_time))*dt
	p1_p2 = np.array(e_space[1]) + np.array(e_space[2])

	# Evaluation quantities
	eta1 = np.max(np.sqrt(e_space[0]))
	eta2 = np.sqrt(np.sum(p1_p2*tau))
	eta3 = np.sum(np.sqrt(e_space[3]))
	eta4 = np.sqrt(np.sum(e_time*tau))

	eta = [eta1, eta2, eta3, eta4]

	return [eta1, eta2, eta3, eta4]

def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = error.keys()
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print("rate %s = %.3f" % (key, rate))

def viz_est(est, plotname, experiment, refinement):
	""" Visualize estimators """
	cwd = os.getcwd()
	newpath = r"%s/fig/%s/%s/estimator/" % (cwd, experiment, refinement)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)
	filename = "%s/%s.pvd" % (newpath, plotname)
	file = File(filename)
	file << est


def plot_error_magnitude(err_mag, N, t, experiment, refinement):
	""" Plot error magnitude at chosen time step """
	viz_est(err_mag[0], "%f_eta1_%f" % (N, t), experiment, refinement)
	viz_est(err_mag[1], "%f_eta2_%f" % (N, t), experiment, refinement)
	viz_est(err_mag[2], "%f_eta3_%f" % (N, t), experiment, refinement)
	viz_est(err_mag[3], "%f_eta4_%f" % (N, t), experiment, refinement)



def run_estimates():
	""" Compute a posteriori error estimates with default parameters under simultaneous space/time refinement. """
	# Parameters
	T = 1.0
	alpha = (1.0, 1.0)
	nu = None
	E = None
	mu = 0.5
	lamb = 1.0
	c = (1.0, 1.0)
	K = (1.0, 1.0)
	xi = (1.0, 1.0)

	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, c=c, K=K, xi=xi)

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))" 		# string expression
	p_str = "(sin(pi*x)*cos(pi*y)*sin(2*pi*t), cos(pi*x)*sin(pi*y)*sin(2*pi*t))" 	# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 	# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 		# get source terms

	# Error lists
	E_u  = []; E_p1 = []; E_p2 = []; h = []; tau = []
	E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

	for N in [4,8,16,32,64]:
		dt = (1./N)**2
		u_h, p_h, u_e, p_e, eta, mesh = compute_estimates(N, T, dt, u_s, p_s, fs, gs, params, experiment="test", refinement=None)

		# Compute error norms
		u_error = errornorm(u_h, u_e, "H1")
		p1_error = errornorm(p_h[0], p_e[0], "L2")
		p2_error = errornorm(p_h[1], p_e[1], "L2")

		# Append error to list
		E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error)
		E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
		h.append(mesh.hmin())

		# Print info to terminal
		print("N = %.d, dt = %.1e, u: %.3e, p1: %.3e, p2: %.3e" % (N, dt, u_error, p1_error, p2_error))
		print("eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3]))

	# Convergence rate
	E = dict(u=E_u, p1=E_p1, p2=E_p2, eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
	convergence_rate(E, h)

if __name__ == "__main__":
	run_estimates()
	

	
