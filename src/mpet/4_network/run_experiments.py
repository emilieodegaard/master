from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import get_source_terms as st
import compute_estimates as ce

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Remove logging
set_log_level(30)

def run(params, refinement):

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*cos(pi*y)*sin(2*pi*t))" 	# string expression
	p_str = "(sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t), \
			sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t))" 		# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 								# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 									# get source terms

	# Brain mesh
	path = r"%s/mesh" % os.getcwd()
	mesh = Mesh("%s/brain.xml" %path)
	#edge_numbers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, mesh.domains())
	
	if refinement == "space":

		# Parameters
		T = 0.1
		dt = 5.0e-3

		# Error lists
		E_u  = []; E_p1 = []; E_p2 = []; E_p3 = []; E_p4 = []; h = []
		E_eta1 = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

		# Solve for each unformal mesh refinement
		for i in range(0,2):
			u_h, p_h, u_e, p_e, eta, mesh_ = ce.compute_estimates(mesh, T, dt, u_s, p_s, fs, gs, params, refinement)

			# Error norms
			u_error = errornorm(u_e, u_h,  "H1", mesh=mesh)
			p1_error = errornorm(p_e[0], p_h[0], "L2", mesh=mesh)
			p2_error = errornorm(p_e[1], p_h[1], "L2", mesh=mesh)
			p3_error = errornorm(p_e[2], p_h[2], "L2", mesh=mesh)
			p4_error = errornorm(p_e[3], p_h[3], "L2", mesh=mesh)
			
			# Save error to lists
			E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error); E_p3.append(p3_error); E_p4.append(p4_error)
			h.append(mesh_.hmin())
			
			# Print error
			print("num_cells = %.d, dt = %.3e, u_err: %.3e, p1_err: %.3e, p2_err: %.3e, p3_err: %.3e, p4_err: %.3e" % (mesh.num_cells(), dt, u_error, p1_error, p2_error, p3_error, p4_error))
			print("eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3]))

			# Refine mesh uniformly
			mesh = adapt(mesh_)
			#refined_edge_numbers = adapt(edge_numbers, mesh)

		
		# Convergence rate
		error = dict(u=E_u, p1=E_p1, p2=E_p2, p3=E_p3, p4=E_p4)
		convergence_rate(error, h)
		eta = dict(eta1=E_eta1, pta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		convergence_rate(eta, h)

	if refinement == "time":

		# Parameters
		T = 1.0

		# Error lists
		E_u  = []; E_p1 = []; E_p2 = []; E_p3 = []; E_p4 = []; tau = []
		E_eta1 = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

		# Solve for each N
		for dt in [0.02, 0.01, 0.005, 0.0025, 0.00125]:

			u_h, p_h, u_e, p_e, eta, mesh = ce.compute_estimates(mesh, T, dt, u_s, p_s, fs, gs, params, refinement)

			# Error norms
			u_error = errornorm(u_e, u_h,  "H1", mesh=mesh)
			p1_error = errornorm(p_e[0], p_h[0], "L2", mesh=mesh)
			p2_error = errornorm(p_e[1], p_h[1], "L2", mesh=mesh)
			p3_error = errornorm(p_e[2], p_h[2], "L2", mesh=mesh)
			p4_error = errornorm(p_e[3], p_h[3], "L2", mesh=mesh)
			
			# Save error to lists
			E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error); E_p3.append(p3_error); E_p4.append(p4_error)
			tau.append(dt)
			
			# Print error
			print("num_cells = %.d, dt = %.3e, u_err: %.3e, p1_err: %.3e, p2_err: %.3e, p3_err: %.3e, p4_err: %.3e" % (mesh.num_cells(), dt, u_error, p1_error, p2_error, p3_error, p4_error))
			print("eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3]))

		
		# Convergence rate
		error = dict(u=E_u, p1=E_p1, p2=E_p2, p3=E_p3, p4=E_p4)
		convergence_rate(error, tau)
		eta = dict(eta4=E_eta4)
		convergence_rate(eta, tau)


	file = File("fig/mesh.pvd")
	file << mesh


def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = list(error.keys())
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print("rate %s = %.3f" % (key, rate))

	
def viz_sol(sol, plotname, refinement):
	""" Visualize solutions """
	cwd = os.getcwd()
	newpath = r"%s/fig/%s/solution/" % (cwd, refinement)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)
	filename = "%s/4mpet_%s.pvd" % (newpath, plotname)
	file = File(filename)
	file << sol

def plot_loglog(h, error, refinement, plotname):
	""" Error plots """
	plt.figure()
	keys = list(error.keys())
	for i in range(len(keys)):
		key = keys[i]
		plt.loglog(h, error[key], label="%s" %key)
	plt.legend(loc=2)

	cwd = os.getcwd()
	newpath = r"%s/fig/%s/error/" % (cwd, refinement)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	plt.savefig("%s/4mpet_%s.png" % (newpath, plotname))


def run_biomedical():
	""" Run biomedically suited parameters """

	print("========= Biomedical experiment =========")

	# Default parameters
	alpha = (0.49, 0.25, 0.01, 0.25)
	nu = 0.4999
	E = 1500.0
	mu = E/(2.0*((1.0 + nu)))
	lamb = nu*E/((1.0-2.0*nu)*(1.0+nu))
	c = (3.9e-4, 2.9e-4, 1.5e-5, 2.9e-4)
	K = (1.57e-5, 3.75e-2, 3.75e-2, 3.75e-2)
	xi = (0.0, 1.0e-6, 1.0e-6, 0.0, 1.0e-6, 1.0e-6) 						
		
	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, nu=nu, E=E, c=c, K=K, xi=xi)

	run(params, refinement="space")
	run(params, refinement="time")


if __name__ == "__main__":

	# Delete all folders within figure
	import os, shutil
	folder = 'fig/'
	for the_file in os.listdir(folder):
	    file_path = os.path.join(folder, the_file)
	    try:
	        if os.path.isfile(file_path):
	            os.unlink(file_path)
	        elif os.path.isdir(file_path): shutil.rmtree(file_path)
	    except Exception as e:
	        print(e)

	run_biomedical()



	