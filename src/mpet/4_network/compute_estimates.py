from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import get_source_terms as st
import os

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Remove logging
set_log_level(30)

def elastic_stress(u, mu, lamb):
	return 2.0*mu*sym(grad(u)) + lamb*tr(sym(grad(u)))*Identity(len(u))

def compute_estimates(mesh, T, dt, u_s, p_s, fs, gs, params, refinement):

	"""
	Solver for a 2D 4-network MPET, 2-field formulation to test a posteriori error estimators.

		-div(sigma(u)) + alpha1*grad(p1) +.. + alpha4*grad(p4) = f
		 c1*p1_t + alpha1*div(u_t) - K1*div(grad(p1)) + S1 = g1
		 c2*p2_t + alpha2*div(u_t) - K2*div(grad(p2)) + S2 = g2
		 c3*p3_t + alpha3*div(u_t) - K3*div(grad(p3)) + S3 = g3
		 c4*p4_t + alpha4*div(u_t) - K4*div(grad(p4)) + S4 = g4

	on the unit square, where

			S1 = xi1*(p1-p2) + xi2*(p1-p3) + xi3*(p1-p4)
			S2 = xi1*(p2-p1) + xi4*(p2-p3) + xi5*(p2-p4)
			S3 = xi2*(p3-p1) + xi4*(p3-p2) + xi6*(p3-p4)
			S4 = xi3*(p4-p1) + xi5*(p4-p2) + xi6*(p4-p3)

	and xi = (xi_{1->2},  xi_{1->3}, xi_{1->4}, xi_{2->3}, xi_{2->4}, xi_{4->3})

	We assume Dirichlet BC's and use Taylor-Hood elements. 
	"""
	
	# Define function spaces
	P2 = VectorElement("CG", mesh.ufl_cell(), 2)	# P2 element
	P1 = FiniteElement("CG", mesh.ufl_cell(), 1)	# P1 element
	TH = MixedElement([P2,P1,P1,P1,P1])				# Taylor-Hood mixed element
	W = FunctionSpace(mesh, TH) 					# mixed function space for all 4 networks
	V = W.sub(0).collapse()
	Q = W.sub(1).collapse()

	# Parameters
	alpha = (params["alpha"])				# Biot-Willis coefficient
	lamb = (params["lamb"])					# Lame parameter
	mu = (params["mu"])						# Lame parameter
	c = (params["c"])						# storage coeffient for each network
	K = (params["K"])						# permeability for each network
	xi = (params["xi"])						# transfer coefficient for each network
	dt_ = Constant(dt)						# avoid recompiling each time we solve
	time = Constant(0.0) 					# time-loop updater

	# Functions
	(u, p1, p2, p3, p4) = TrialFunctions(W)
	(v, q1, q2, q3, q4) = TestFunctions(W)
	
	# Initial functions
	up0 = Function(W)
	(u0, p1_0, p2_0, p3_0, p4_0) = split(up0)

	# Source terms
	f = Expression((fs[0], fs[1]), mu=mu, lamb=lamb, alpha1=alpha[0], alpha2=alpha[1], alpha3=alpha[2], alpha4=alpha[3], t=time, degree=7)
	g1 = Expression(gs[0], alpha1=alpha[0], K1=K[0], c1=c[0], xi1=xi[0], xi2=xi[1], xi3=xi[2], t=time, degree=7)
	g2 = Expression(gs[1], alpha2=alpha[1], K2=K[1], c2=c[1], xi1=xi[0], xi4=xi[3], xi5=xi[4], t=time, degree=7)
	g3 = Expression(gs[2], alpha3=alpha[2], K3=K[2], c3=c[2], xi2=xi[1], xi4=xi[3], xi6=xi[5], t=time, degree=7)
	g4 = Expression(gs[3], alpha4=alpha[3], K4=K[3], c4=c[3], xi3=xi[3], xi5=xi[4], xi6=xi[5], t=time, degree=7)
	g = [g1, g2, g3, g4]

	# Exact solutions
	u_e = Expression((u_s[0], u_s[1]), t=time, degree=7)
	p1_e = Expression(p_s[0], t=time, degree=7)
	p2_e = Expression(p_s[1], t=time, degree=7)
	p3_e = Expression(p_s[2], t=time, degree=7)
	p4_e = Expression(p_s[3], t=time, degree=7)

	# Boundary function
	def boundary(x):
		return x[0] < DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0-DOLFIN_EPS 
		
	# Boundary conditions
	u_bc = DirichletBC(W.sub(0), u_e, boundary)
	p1_bc = DirichletBC(W.sub(1), p1_e, boundary)
	p2_bc = DirichletBC(W.sub(2), p2_e, boundary)
	p3_bc = DirichletBC(W.sub(3), p3_e, boundary)
	p4_bc = DirichletBC(W.sub(4), p4_e, boundary)
	bc = [u_bc, p1_bc, p2_bc, p3_bc, p4_bc]

	# Stress tensor
	sigma = lambda u: elastic_stress(u, mu, lamb)

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

	# Lists to compute error estimators
	e_u = []; e_p1 = []; e_p2 = []; e_p3 = []; e_p4 = []
	e_u_dt = []; e_p_dt = []; e_time = []

	# Facetnormal needed for jump-function
	n = FacetNormal(mesh)

	# Initial residuals (at time = 0)
	S1_0 = xi[0]*(p1_0-p2_0) + xi[1]*(p1_0-p3_0) + xi[2]*(p1_0-p4_0)
	S2_0 = xi[0]*(p2_0-p1_0) + xi[3]*(p2_0-p3_0) + xi[4]*(p2_0-p4_0)
	S3_0 = xi[1]*(p3_0-p1_0) + xi[3]*(p3_0-p2_0) + xi[5]*(p3_0-p4_0)
	S4_0 = xi[2]*(p4_0-p1_0) + xi[4]*(p4_0-p2_0) + xi[5]*(p4_0-p3_0)
	RK_u0 = f + div(sigma(u0)) - alpha[0]*grad(p1_0) - alpha[1]*grad(p2_0) - alpha[2]*grad(p3_0) - alpha[3]*grad(p4_0)
	RE_u0 = alpha[0]*jump(p1_0,n) + alpha[1]*jump(p2_0,n) + alpha[2]*jump(p3_0,n) + alpha[3]*jump(p4_0,n) - jump(sigma(u0),n)
	RK_p1_0 = dt_*g1 - c[0]*(p1_0) - alpha[0]*div(u0) + dt_*K[0]*div(grad(p1_0)) - dt_*S1_0
	RK_p2_0 = dt_*g2 - c[1]*(p2_0) - alpha[1]*div(u0) + dt_*K[1]*div(grad(p2_0)) - dt_*S2_0
	RK_p3_0 = dt_*g3 - c[2]*(p3_0) - alpha[2]*div(u0) + dt_*K[2]*div(grad(p3_0)) - dt_*S3_0
	RK_p4_0 = dt_*g4 - c[3]*(p4_0) - alpha[3]*div(u0) + dt_*K[3]*div(grad(p4_0)) - dt_*S4_0
	RE_p1_0 = -K[0]*jump(grad(p1_0),n)
	RE_p2_0 = -K[1]*jump(grad(p2_0),n)
	RE_p3_0 = -K[2]*jump(grad(p3_0),n)
	RE_p4_0 = -K[3]*jump(grad(p4_0),n)
	R0 = [RK_u0, RE_u0, RK_p1_0, RE_p1_0, RK_p2_0, RE_p2_0, RK_p3_0, RE_p3_0, RK_p4_0, RE_p4_0]

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
		e_p3.append(err_ind[3].vector().sum())
		e_p4.append(err_ind[4].vector().sum())
		e_u_dt.append(err_ind[5].vector().sum())
		e_time.append(err_ind[6].sum())

		# Update previous solution 
		up0.assign(up)

		# Update residuals
		R0 = R

		# Update time step
		t += dt


	# Compute final error estimators
	e_space = [e_u, e_p1, e_p2, e_p3, e_p4, e_u_dt]
	e_time = np.array(e_time)
	eta = error_estimators(e_space, e_time, dt)

	# Numerical solutions
	(u_h, p1_h, p2_h, p3_h, p4_h) = up.split()
	p_h = [p1_h, p2_h, p3_h, p4_h]

	# Exact solution needs to match time level
	# u_e = interpolate(Expression((u_s[0], u_s[1]), t=t-dt, degree=7), V)
	# p1_e = interpolate(Expression(p_s[0], t=t-dt, degree=7), Q)
	# p2_e = interpolate(Expression(p_s[1], t=t-dt, degree=7), Q)
	# p3_e = interpolate(Expression(p_s[2], t=t-dt, degree=7), Q)
	# p4_e = interpolate(Expression(p_s[3], t=t-dt, degree=7), Q)
	p_e = [p1_e, p2_e, p3_e, p4_e]

	if refinement == "space":
		plot_error_magnitude(err_mag, mesh.num_cells(), refinement)
	if refinement == "time":
		plot_error_magnitude(err_mag, dt, refinement)
	
	return u_h, p_h, u_e, p_e, eta, mesh

def error_indicators(up, up0, f, g, dt, mesh, R0, params):
	
	# Quantities needed to compute error estimators
	h = CellDiameter(mesh) 					# element diameter
	n = FacetNormal(mesh) 				# normal on element-facet
	DG0 = FunctionSpace(mesh, "DG", 0) 	# space of pieciewise constant functions
	w = TestFunction(DG0) 				# element indicator test function

	# Parameters
	alpha = (params["alpha"])		
	lamb = (params["lamb"])				
	mu = (params["mu"])				
	c = params["c"]							
	K = params["K"]			
	xi = params["xi"]	

	# Define elastic stress
	sigma = lambda u: elastic_stress(u, mu, lamb)

	# Split solutions
	u_h = split(up)[0]; p1_h = split(up)[1]; p2_h = split(up)[2]; p3_h = split(up)[3]; p4_h = split(up)[4]
	u0 = split(up0)[0]; p1_0 = split(up0)[1]; p2_0 = split(up0)[2]; p3_0 = split(up0)[3]; p4_0 = split(up0)[4]

	# Extract previous residuals
	RK_u0 = R0[0]; RE_u0 = R0[1]
	RK_p1_0 = R0[2]; RE_p1_0 = R0[3]; RK_p2_0 = R0[4]; RE_p2_0 = R0[5]
	RK_p3_0 = R0[6]; RE_p3_0 = R0[7]; RK_p4_0 = R0[8]; RE_p4_0 = R0[9]

	# Transfer coefficients
	S1 = xi[0]*(p1_h-p2_h) + xi[1]*(p1_h-p3_h) + xi[2]*(p1_h-p4_h)
	S2 = xi[0]*(p2_h-p1_h) + xi[3]*(p2_h-p3_h) + xi[4]*(p2_h-p4_h)
	S3 = xi[1]*(p3_h-p1_h) + xi[3]*(p3_h-p2_h) + xi[5]*(p3_h-p4_h)
	S4 = xi[2]*(p4_h-p1_h) + xi[4]*(p4_h-p2_h) + xi[5]*(p4_h-p3_h)

	# Element and face residuals for u
	RK_u = f + div(sigma(u_h)) - alpha[0]*grad(p1_h) - alpha[1]*grad(p2_h) - alpha[2]*grad(p3_h) - alpha[3]*grad(p4_h)
	RE_u = alpha[0]*jump(p1_h,n) + alpha[1]*jump(p2_h,n) + alpha[2]*jump(p3_h,n) + alpha[3]*jump(p4_h,n) - jump(sigma(u_h),n)
	RK_u_dt = (RK_u - RK_u0)/dt
	RE_u_dt = (RE_u - RE_u0)/dt

	# Element and face residuals for p1,..,p4
	RK_p1 = dt*g[0] - c[0]*(p1_h-p1_0) - alpha[0]*div(u_h-u0) + dt*K[0]*div(grad(p1_h)) - dt*S1
	RK_p2 = dt*g[1] - c[1]*(p2_h-p2_0) - alpha[1]*div(u_h-u0) + dt*K[1]*div(grad(p2_h)) - dt*S2
	RK_p3 = dt*g[2] - c[2]*(p3_h-p3_0) - alpha[2]*div(u_h-u0) + dt*K[2]*div(grad(p3_h)) - dt*S3
	RK_p4 = dt*g[3] - c[3]*(p4_h-p4_0) - alpha[3]*div(u_h-u0) + dt*K[3]*div(grad(p4_h)) - dt*S4
	RE_p1 = -K[0]*jump(grad(p1_0),n)
	RE_p2 = -K[1]*jump(grad(p2_0),n)
	RE_p3 = -K[2]*jump(grad(p3_0),n)
	RE_p4 = -K[3]*jump(grad(p4_0),n)
	RK_p1_dt = (RK_p1 - RK_p1_0)/dt; RE_p1_dt = (RE_p1 - RE_p1_0)/dt
	RK_p2_dt = (RK_p2 - RK_p2_0)/dt; RE_p2_dt = (RE_p2 - RE_p2_0)/dt
	RK_p3_dt = (RK_p3 - RK_p3_0)/dt; RE_p3_dt = (RE_p3 - RE_p3_0)/dt
	RK_p4_dt = (RK_p4 - RK_p4_0)/dt; RE_p4_dt = (RE_p4 - RE_p4_0)/dt
	
	residuals = [RK_u, RE_u, RK_p1, RE_p1, RK_p2, RE_p2, RK_p3, RE_p3, RK_p4, RE_p4]


	# Residual spatial error estimators
	R_u = w*(h**2)*RK_u**2*dx + avg(h)*avg(w)*RE_u**2*dS
	R_p1 = w*(h**2)*RK_p1**2*dx + avg(h)*avg(w)*RE_p1**2*dS
	R_p2 = w*(h**2)*RK_p2**2*dx + avg(h)*avg(w)*RE_p2**2*dS
	R_p3 = w*(h**2)*RK_p3**2*dx + avg(h)*avg(w)*RE_p3**2*dS
	R_p4 = w*(h**2)*RK_p4**2*dx + avg(h)*avg(w)*RE_p4**2*dS

	# Residual temporal error estimators
	R_u_dt = (dt**2)*(w*(h**2)*RK_u_dt**2*dx + avg(h)*avg(w)*RE_u_dt**2*dS)

	# Time estimator
	R_p1_dt = w*grad(p1_h-p1_0)**2*dx
	R_p2_dt = w*grad(p2_h-p2_0)**2*dx
	R_p3_dt = w*grad(p3_h-p3_0)**2*dx
	R_p4_dt = w*grad(p4_h-p4_0)**2*dx

	# Piecewise constant functions
	err_u = Function(DG0); err_u_dt = Function(DG0)
	err_p1 = Function(DG0); err_p1_dt = Function(DG0)
	err_p2 = Function(DG0); err_p2_dt = Function(DG0)
	err_p3 = Function(DG0); err_p3_dt = Function(DG0)
	err_p4 = Function(DG0); err_p4_dt = Function(DG0)
	err_p = Function(DG0); err_p_dt = Function(DG0)

	# Space error indicators
	assemble(R_u, tensor=err_u.vector())
	assemble(R_p1, tensor=err_p1.vector())
	assemble(R_p2, tensor=err_p2.vector())
	assemble(R_p3, tensor=err_p3.vector())
	assemble(R_p4, tensor=err_p4.vector())
	assemble(R_u_dt, tensor=err_u_dt.vector())

	# Time error indicators
	assemble(R_p1_dt, tensor=err_p1_dt.vector())
	assemble(R_p2_dt, tensor=err_p2_dt.vector())
	assemble(R_p3_dt, tensor=err_p3_dt.vector())
	assemble(R_p4_dt, tensor=err_p4_dt.vector())
	err_time = (K[0]+xi[0])*err_p1_dt.vector() + (K[1]+xi[1])*err_p2_dt.vector() + (K[2]+xi[2])*err_p3_dt.vector() + (K[3]+xi[3])*err_p4_dt.vector() 

	# All error indicators
	error_indicators = [err_u, err_p1, err_p2, err_p3, err_p4, err_u_dt, err_time]

	# Error magnitude for plotting
	error_magnitude = [err_u, err_p1, err_p2, err_p3, err_p4, err_u_dt, err_p1_dt, err_p2_dt, err_p3_dt, err_p4_dt]

	return residuals, error_indicators, error_magnitude

def error_estimators(e_space, e_time, dt):

	# Info for time estimator
	tau = np.ones(len(e_time))*dt
	p1_p4 = np.array(e_space[1]) + np.array(e_space[2] + np.array(e_space[3]) + np.array(e_space[4]))

	# Evaluation quantities
	eta1 = np.sqrt(np.sum(p1_p4*tau))
	eta2 = np.sqrt(np.max(e_space[0]))
	eta3 = np.sum(np.sqrt(e_space[3]))
	eta4 = np.sqrt(np.sum(e_time*tau))

	eta = [eta1, eta2, eta3, eta4]

	return [eta1, eta2, eta3, eta4]

def viz_est(est, plotname, refinement):
	""" Visualize solutions """
	filename = "fig/%s/estimate/mpet_%s.pvd" % (refinement, plotname)
	file = File(filename)
	file << est

def plot_error_magnitude(err_ind, discr, refinement):
	""" Plot error magnitude at chosen time step """
	if refinement == "space":

		viz_est(err_ind[0], "%f_eta1" % discr, refinement)
		viz_est(err_ind[1], "%f_eta2_p1" % discr, refinement)
		viz_est(err_ind[2], "%f_eta2_p2" % discr, refinement)
		viz_est(err_ind[3], "%f_eta2_p3" % discr, refinement)
		viz_est(err_ind[4], "%f_eta2_p4" % discr, refinement)
		viz_est(err_ind[5], "%f_eta3" % discr, refinement)
	if refinement == "time":
		viz_est(err_ind[6], "%f_eta4_p1" % discr, refinement)
		viz_est(err_ind[7], "%f_eta4_p2" % discr, refinement)
		viz_est(err_ind[8], "%f_eta4_p3" % discr, refinement)
		viz_est(err_ind[9], "%f_eta4_p4" % discr, refinement)


def run_estimates():

	# Input parameters
	T = 1.0
	alpha = (1.0, 1.0, 1.0, 1.0)
	mu = 0.5
	lamb = 1.0
	c = (1.0, 1.0, 1.0, 1.0)
	K = (1.0, 1.0, 1.0, 1.0)
	xi = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*cos(pi*y)*sin(2*pi*t))" 		# string expression
	p_str = "(sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t), \
			sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t))" 		# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 								# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 									# get source terms

	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, c=c, K=K, xi=xi)

	for N in [4,8,16,32,64]:
		dt = (1./N)**2
		mesh = UnitSquareMesh(N,N)
		u_h, p_h, u_e, p_e, eta, mesh = compute_estimates(mesh, T, dt, u_s, p_s, fs, gs, params)


if __name__ == "__main__":
	run_estimates()