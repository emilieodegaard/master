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
"""

def elastic_stress(u, mu, lamb):
	return 2.0*mu*sym(grad(u)) + lamb*tr(sym(grad(u)))*Identity(len(u))


def solver(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement):

	# Mesh
	mesh = UnitSquareMesh(N,N)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)		# displacement
	Q = FunctionSpace(mesh, "CG", 1)			# pressure
	W = MixedFunctionSpace([V,Q]) 				# mixed function space

	# Parameters
	alpha = params["alpha"]
	mu = params["mu"]
	lamb = params["lamb"]
	nu = params["nu"]
	E = params["E"]
	c = params["c"]
	K = params["K"]
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
	(u0, p0) = split(up0)

	# Source terms
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha=alpha, t=time)
	g = Expression(gs, alpha=alpha, K=K, c=c, t=time)

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), t=time, degree=3)
	p_e = Expression(p_s, t=time, degree=3)

	# Stress tensor
	sigma = lambda u: elastic_stress(u, mu, lamb)
	
	# What happens on the boundary?
	def boundary(x):
		return x[0] < DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0-DOLFIN_EPS 
		
	# Boundary conditions
	u_bc = DirichletBC(W.sub(0), u_e, boundary)
	p_bc = DirichletBC(W.sub(1), p_e, boundary)
	bc = [u_bc, p_bc]

	# Variational problem
	a1 = (inner(sigma(u),grad(v)) - alpha*inner(p,div(v)))*dx
	L1 = inner(f,v)*dx
	a2 = (c*inner(p,q) + alpha*inner(div(u),q) + K*dt_*inner(grad(p),grad(q)))*dx
	L2 = (dt_*inner(g,q) + c*inner(p0,q) + alpha*inner(div(u0),q))*dx 
	a = a1 + a2
	L = L1 + L2

	# Lists to compute error estimators
	e_u = []; e_p = []; e_u_dt = []		# space error indicators
	e_time = []; ts = []				# time error indicators

	# Facetnormal needed for jump-function
	n = FacetNormal(mesh)

	# Initial residuals (at time = 0)
	RK_u0 = f + div(sigma(u0)) - alpha*grad(p0)
	RE_u0 = alpha*jump(p0,n) - jump(sigma(u0),n)
	RK_p0 = dt*g - c*(p0) - alpha*div(u0) + dt*K*div(grad(p0))
	RE_p0 = -K*jump(grad(p0),n)
	R0 = [RK_u0, RE_u0, RK_p0, RE_p0]

	# Numerical solution
	up = Function(W)

	# Time range to plot error magnitude
	time_range = range(0,int(T/dt), int(T*0.25/dt))
	
	# Solve for each time step t
	t = dt
	while t <= T:

		# Update time 
		time.assign(t)

		# Solve variational problem
		solve(a==L, up, bc)

		# Compute space error indicators
		R, err_ind = error_indicators(up, up0, f, g, dt, mesh, R0, mu, lamb, alpha, c, K)

		# Sum over error indicators
		e_u.append(err_ind[0].vector().sum())
		e_p.append(err_ind[1].vector().sum())
		e_u_dt.append(err_ind[2].vector().sum())
		e_time.append(err_ind[3].vector().sum())

		# Update previous solution 
		up0.assign(up)

		# Update residuals
		R0 = R

		t += dt

		# Plot error magnitude at 4 evenly spaced out time blocks
		if int(T/t) in time_range:
			plot_error_magnitude(err_ind, N, t, experiment, refinement)

	# Numerical solutions
	(u_h, p_h) = up.split()

	# Compute final error estimators
	e_space = [e_u, e_p, e_u_dt]
	e_time = np.array(e_time)
	eta = error_estimators(e_space, e_time, dt)

	# Exact solution needs to match time level
	u_e = interpolate(Expression((u_s[0], u_s[1]), t=t-dt, degree=3), V)
	p_e = interpolate(Expression(p_s, t=t-dt, degree=3), Q)
	
	return u_h, p_h, u_e, p_e, eta, mesh

def error_indicators(up, up0, f, g, dt, mesh, R0, mu, lamb, alpha, c, K):
	
	# Quantities needed to compute error estimators
	h = CellSize(mesh)
	n = FacetNormal(mesh)
	DG0 = FunctionSpace(mesh, "DG", 0) 	# space of pieciewise constant functions
	w = TestFunction(DG0) 				# element indicator test function

	# Define elastic stress
	sigma = lambda u: elastic_stress(u, mu, lamb)

	# Split solutions
	u_h = split(up)[0]
	p_h = split(up)[1]
	u0 = split(up0)[0]
	p0 = split(up0)[1]

	# Extract previous residuals
	RK_u0 = R0[0]; RE_u0 = R0[1]; RK_p0 = R0[2]; RE_p0 = R0[3]

	# Facet normal needed for jump-function
	n = FacetNormal(mesh)

	# Element and face residuals for u
	RK_u = f + div(sigma(u_h)) - alpha*grad(p_h) 	# element residual
	RK_u_dt = (RK_u - RK_u0)/dt
	RE_u = alpha*jump(p_h,n) - jump(sigma(u_h),n) 	# face residual
	RE_u_dt = (RE_u - RE_u0)/dt

	# Element and face residuals for p
	RK_p = dt*g - c*(p_h-p0) - alpha*div(u_h-u0) + dt*K*div(grad(p_h))
	RK_p_dt = (RK_p - RK_p0)/dt
	RE_p = -K*jump(grad(p_h),n)
	RE_p_dt = (RE_p - RE_p0)/dt

	# Residual spatial error estimators
	R_u = w*h**2*RK_u**2*dx + avg(h)*avg(w)*RE_u**2*dS
	R_p = w*h**2*RK_p**2*dx + avg(h)*avg(w)*RE_p**2*dS

	# Residual temporal error estimators
	R_u_dt = (dt**2)*(w*(h**2)*dot(RK_u_dt, RK_u_dt)*dx + avg(h)*avg(w)*dot(RE_u_dt, RE_u_dt)*dS)
	
	# Time estimator
	R_p_dt = K*w*grad(p_h-p0)**2*dx

	# Error indicators
	err_u = Function(DG0)
	err_p = Function(DG0)
	err_u_dt = Function(DG0)
	err_p_dt = Function(DG0)

	assemble(R_u, tensor=err_u.vector())
	assemble(R_p, tensor=err_p.vector())
	assemble(R_u_dt, tensor=err_u_dt.vector())
	assemble(R_p_dt, tensor=err_p_dt.vector())

	residuals = [RK_u, RE_u, RK_p, RE_p]

	error_indicators = [err_u, err_p, err_u_dt, err_p_dt]

	return residuals, error_indicators

def error_estimators(e_space, e_time, dt):

	# Info for time estimator
	tau = np.ones(len(e_time))*dt

	# Evaluation quantities
	eta1 = np.sqrt(np.sum(e_space[1]*tau))
	eta2 = np.sqrt(np.max(e_space[0]))
	eta3 = np.sum(np.sqrt(e_space[2]))
	eta4 = np.sqrt(np.sum(e_time*tau))

	return [eta1, eta2, eta3, eta4]


	
def run(params, experiment, mms_test, refinement):
	
	# Get source terms using sympy
	if mms_test == 1:
		u_str = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))"	# string expression
		p_str = "sin(pi*x)*cos(pi*y)*sin(2*pi*t)"

	if mms_test == 2:
		u_str = "(x*(x-1)*sin(pi*x)*cos(pi*y)*sin(pi*t), y*(y-1)*sin(pi*x)*cos(pi*y)*sin(pi*t))"	# string expression
		p_str = "sin(pi*x)*sin(pi*y)*sin(pi*t)" 

	if mms_test == 3:
		u_str = "(t*t*sin(pi*x)*sin(2*pi*y), t*sin(3*pi*x)*sin(4*pi*y))"
		p_str = "t*exp(1-t)*x*y*(1-x)*(1-y)"

	if mms_test == 4:
		A = 2*pi**2*K/(alpha + c)
		u_str = "(-exp(-%.5f*t)*(1/2*pi)*cos(pi*x)*sin(pi*y), -exp(-%.5f*t)*(1/2*pi)**sin(pi*x)*cos(pi*y))" % (A,A)
		p_str = "exp(-%.5f*t)*sin(pi*x)*sin(pi*y)*sin(pi*t)" % A

	u_s = st.str2expr(u_str); p_s = st.str2expr(p_str)	# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str)		# get source terms

	# Error lists
	E_u  = []; E_p = []; h = []; tau = []
	E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

	if refinement=="space":

		print "============= Space refinement ============="

		# Parameters
		T = 0.1
		dt = 5.0e-5
		
		# Solve for each N
		for N in [4,8,16,32,64]:
			u_h, p_h, u_e, p_e, eta, mesh = solver(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)

			# Error norms
			u_error = errornorm(u_e, u_h, "H1")
			p_error = errornorm(p_e, p_h, "L2")

			# Append error to list
			E_u.append(u_error); E_p.append(p_error)
			E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
			h.append(mesh.hmin())

			# Print info to terminal
			print "N = %.d, dt = %.1e, u: %.3e, p: %.3e" % (N, dt, u_error, p_error)
			print "eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3])

			# Solution plots for each N
			viz_sol(u_h, "u_num_%d" % N, experiment, refinement)
			viz_sol(u_e, "u_exact_%d" % N, experiment, refinement)
			viz_sol(p_h, "p_num_%d" % N, experiment, refinement)
			viz_sol(p_e, "p_exact_%d" % N, experiment, refinement)

		# Convergence rate
		E = dict(u=E_u, p=E_p, eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		convergence_rate(E, h)

		# Error plots
		error = dict(u=E_u, p=E_p)
		plot_loglog(h, error, experiment, refinement, plotname="solution")
		est = dict(eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		plot_loglog(h, est, experiment, refinement, plotname="estimator")

	if refinement == "time":

		print "============= Time refinement ============="

		# Parameters
		N = 128
		T = 1.0

		# Solve for each dt
		for dt in [0.02, 0.01, 0.005, 0.0025, 0.00125]:
			u_h, p_h, u_e, p_e, eta, mesh = solver(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)
			# Error norms
			u_error = errornorm(u_e, u_h, "H1")
			p_error = errornorm(p_e, p_h, "L2")

			# Append error to list
			E_u.append(u_error); E_p.append(p_error)
			E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
			h.append(mesh.hmin()); tau.append(dt)

			# Print info to terminal
			print "N = %.d, dt = %.1e, u: %.3e, p: %.3e" % (N, dt, u_error, p_error)
			print "eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3])

			# Solution plots for each dt
			viz_sol(u_h, "u_num_%.f" % dt, experiment, refinement)
			viz_sol(u_e, "u_exact%.f" % dt, experiment, refinement)
			viz_sol(p_h, "p_num%.f" % dt, experiment, refinement)
			viz_sol(p_e, "p_exact%.f" % dt, experiment, refinement)


		# Convergence rate
		E = dict(u=E_u, p=E_p, eta4=E_eta4)
		convergence_rate(E, tau)

		# Error plots
		error = dict(u=E_u, p=E_p)
		plot_loglog(tau, error, experiment, refinement, plotname="solution")
		est = dict(eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		plot_loglog(tau, est, experiment, refinement, plotname="estimator")

	if refinement == None:

		print "============= Time/space refinement ============="

		# Parameters
		T = 1.0

		for dt in [(1./4)**2,(1./8)**2,(1./16)**2,(1./32)**2]:
			
			# Error lists
			E_u  = []; E_p = []; h = []; tau = []
			E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []
		
			# Solve for each N
			for N in [4,8,16,32]:
				u_h, p_h, u_e, p_e, eta, mesh = solver(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)

				# Error norms
				u_error = errornorm(u_e, u_h, "H1")
				p_error = errornorm(p_e, p_h, "L2")

				# Append error to list
				E_u.append(u_error); E_p.append(p_error)
				E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
				h.append(mesh.hmin())

				# Print info to terminal
				print "N = %.d, dt = %.1e, u: %.3e, p: %.3e, eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (N, dt, u_error, p_error, eta[0], eta[1], eta[2], eta[3])

			# Convergence rate
			error = dict(u=E_u, p=E_p, eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
			convergence_rate(error, h)


def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = error.keys()
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print "rate %s = %.3f" % (key, rate)
	
def viz_sol(sol, plotname, experiment, refinement):
	""" Visualize solutions """
	filename = "fig/%s/%s/solution/bb_%s.pvd" % (experiment, refinement, plotname)
	file = File(filename)
	file << sol

def viz_est(est, plotname, experiment, refinement):
	""" Visualize solutions """
	filename = "fig/%s/%s/estimate/biot_%s.pvd" % (experiment, refinement, plotname)
	file = File(filename)
	file << est

def plot_loglog(h, error, experiment, refinement, plotname):
	""" Error plots """
	plt.figure()
	keys = error.keys()
	for i in range(len(keys)):
		key = keys[i]
		plt.loglog(h, error[key], label="%s" %key)
	plt.legend(loc=2)
	plt.savefig("fig/%s/%s/error/biot_%s_error.png" % (experiment, refinement, plotname))
	plt.hold(False)

def plot_error_magnitude(err_ind, N, t, experiment, refinement):
	""" Plot error magnitude at chosen time step """
	viz_est(err_ind[0], "%d_est_u_%f" % (N, t), experiment, refinement)
	viz_est(err_ind[1], "%d_est_p_%f" % (N, t), experiment, refinement)
	viz_est(err_ind[2], "%d_est_u_dt_%f" % (N, t), experiment, refinement)
	viz_est(err_ind[3], "%d_est_time_%f" % (N, t), experiment, refinement)


def run_default(mms_test=1):
	""" Run default parameters, all set to 1. """

	print "========= Biot, default parameters ========="

	experiment = "default"

	# Default parameters
	alpha = 1.0
	nu = None
	E = None
	mu = 0.5
	lamb = 1.0
	c = 1.0
	K = 1.0

	params = dict(alpha=alpha, mu=mu, lamb=lamb, nu=nu, E=E, c=c, K=K)
		
	# Print parameter-info to terminal
	print "alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu
	print "lamb = ", lamb, "c = ",  c, "K =", K

	#run(params, experiment, mms_test, refinement="space")
	run(params, experiment, mms_test, refinement="time")
	run(params, experiment, mms_test, refinement=None)

if __name__ == "__main__":
	run_default(mms_test=1)
	