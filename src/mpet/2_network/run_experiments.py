from dolfin import *
import numpy as np
import os
import matplotlib.pyplot as plt
import get_source_terms as st
import compute_estimates as ce

# Remove logging
set_log_level(30)

def run(params, experiment="default", refinement="space"):

	# Get source terms using sympy
	u_str = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))" 	# string expression
	p_str = "(sin(pi*x)*cos(pi*y)*sin(2*pi*t), cos(pi*x)*sin(pi*y)*sin(2*pi*t))" 	# string expression
	
	u_s = st.str2exp(u_str); p_s = st.str2exp(p_str) 	# create FEniCS expressions
	(fs, gs) = st.get_source_terms(u_str, p_str) 		# get source terms

	if refinement == "space":

		print("============= Space refinement =============")

		# Parameters
		T = 0.1
		dt = 5.0e-5

		# Error lists
		E_u  = []; E_p1 = []; E_p2 = []; h = []; tau = []
		E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []
		
		# Solve for each N
		for N in [4,8,16,32,64]:
			u_h, p_h, u_e, p_e, eta, mesh = ce.compute_estimates(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)

			# Compute error norms
			u_error = errornorm(u_e, u_h, "H1", mesh=mesh)
			p1_error = errornorm(p_e[0], p_h[0], "L2", mesh=mesh)
			p2_error = errornorm(p_e[1], p_h[1], "L2", mesh=mesh)

			# Append error to list
			E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error)
			E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
			h.append(mesh.hmin())

			# Print info to terminal
			print("N = %.d, dt = %.1e, u: %.3e, p1: %.3e, p2: %.3e" % (N, dt, u_error, p1_error, p2_error))
			print("eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3]))

			# Solution plots for each N
			viz_sol(u_h, "u_num_%d" % N, experiment, refinement)
			viz_sol(u_e, "u_exact_%d" % N, experiment, refinement)
			viz_sol(p_h[0], "p1_num_%d" % N, experiment, refinement)
			viz_sol(p_e[0], "p1_exact_%d" % N, experiment, refinement)
			viz_sol(p_h[1], "p2_num_%d" % N, experiment, refinement)
			viz_sol(p_e[1], "p2_exact_%d" % N, experiment, refinement)

		# Convergence rate
		E = dict(u=E_u, p1=E_p1, p2=E_p2, eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		convergence_rate(E, h)

		# Error plots
		error = dict(u=E_u, p1=E_p1, p2=E_p2)
		plot_loglog(h, error, experiment, refinement, plotname="solution")
		est = dict(eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		plot_loglog(h, est, experiment, refinement, plotname="estimator")



	if refinement == "time":
		
		print("============= Time refinement =============")

		# Parameters
		N = 128
		T = 1.0

		# Error lists
		E_u  = []; E_p1 = []; E_p2 = []; h = []; tau = []
		E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

		# Solve for each dt
		for dt in [0.02, 0.01, 0.005, 0.0025, 0.00125]:
			u_h, p_h, u_e, p_e, eta, mesh = ce.compute_estimates(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)

			# Compute error norms
			u_error = errornorm(u_e, u_h, "H1", mesh=mesh)
			p1_error = errornorm(p_e[0], p_h[0], "L2", mesh=mesh)
			p2_error = errornorm(p_e[1], p_h[1], "L2", mesh=mesh)

			# Append error to list
			E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error)
			E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
			h.append(mesh.hmin()); tau.append(dt)

			# Print info to terminal
			print("N = %.d, dt = %.1e, u: %.3e, p1: %.3e, p2: %.3e" % (N, dt, u_error, p1_error, p2_error))
			print("eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (eta[0], eta[1], eta[2], eta[3]))

			# Solution plots for each dt
			viz_sol(u_h, "u_num_%.f" % dt, experiment, refinement)
			viz_sol(u_e, "u_exact%.f" % dt, experiment, refinement)
			viz_sol(p_h[0], "p1_num%.f" % dt, experiment, refinement)
			viz_sol(p_e[0], "p1_exact%.f" % dt, experiment, refinement)
			viz_sol(p_h[1], "p2_num%.f" % dt, experiment, refinement)
			viz_sol(p_e[1], "p2_exact%.f" % dt, experiment, refinement)


		# Convergence rate
		E = dict(u=E_u, p1=E_p1, p2=E_p2, eta4=E_eta4)
		convergence_rate(E, tau)

		# Error plots
		error = dict(u=E_u, p1=E_p1, p2=E_p2)
		plot_loglog(tau, error, experiment, refinement, plotname="solution")
		est = dict(eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
		plot_loglog(tau, est, experiment, refinement, plotname="estimator")


	if refinement == None:

		print("============= Time/space refinement =============")

		# Parameters
		T = 1.0

		# Error lists
		E_u  = []; E_p1 = []; E_p2 = []; h = []; tau = []
		E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

		for dt in [(1./4)**2,(1./8)**2,(1./16)**2,(1./32)**2]:
			
			# Error lists
			E_u  = []; E_p1 = []; E_p2 = []; h = []; tau = []
			E_eta1  = []; E_eta2 = []; E_eta3 = []; E_eta4 = []

			# Solve for each N
			for N in [4,8,16,32]:
				u_h, p_h, u_e, p_e, eta, mesh = ce.compute_estimates(N, T, dt, u_s, p_s, fs, gs, params, experiment, refinement)

				# Compute error norms
				u_error = errornorm(u_e, u_h, "H1", mesh=mesh)
				p1_error = errornorm(p_e[0], p_h[0], "L2", mesh=mesh)
				p2_error = errornorm(p_e[1], p_h[1], "L2", mesh=mesh)

				# Append error to list
				E_u.append(u_error); E_p1.append(p1_error); E_p2.append(p2_error)
				E_eta1.append(eta[0]); E_eta2.append(eta[1]); E_eta3.append(eta[2]); E_eta4.append(eta[3])
				h.append(mesh.hmin())

				# Print info to terminal
				print("N = %.d, dt = %.1e, u: %.3e, p1: %.3e, p2: %.3e, eta1: %.3e, eta2: %.3e, eta3: %.3e, eta4: %.3e" % (N, dt, u_error, p1_error, p2_error, eta[0], eta[1], eta[2], eta[3]))

			# Convergence rate
			error = dict(u=E_u, p1=E_p1, p2=E_p2, eta1=E_eta1, eta2=E_eta2, eta3=E_eta3, eta4=E_eta4)
			convergence_rate(error, h)


def convergence_rate(error, h):
	""" Compute convergence rates """
	keys = list(error.keys())
	for i in range(len(keys)):
		key = keys[i]
		E = error[key]
		for i in range(len(E)-1):
			rate = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
			print("rate %s = %.3f" % (key, rate))
	
def viz_sol(sol, plotname, experiment, refinement):
	""" Visualize solutions """
	cwd = os.getcwd()
	newpath = r"%s/fig/%s/%s/solution/" % (cwd, experiment, refinement)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)
	filename = "%s/mpet2_%s.pvd" % (newpath, plotname)
	file = File(filename)
	file << sol

def plot_loglog(h, error, experiment, refinement, plotname):
	""" Error plots """
	plt.figure()
	keys = list(error.keys())
	for i in range(len(keys)):
		key = keys[i]
		plt.loglog(h, error[key], label="%s" %key)
	plt.legend(loc=2)

	cwd = os.getcwd()
	newpath = r"%s/fig/%s/%s/error/" % (cwd, experiment, refinement)
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	plt.savefig("%s/mpet2_%s_error.png" % (newpath, plotname))


def run_biomedical():
	""" Run biomedically suited parameters """

	print("========= Biomedical experiment =========")

	# Default parameters
	alpha = (0.49, 0.25)
	nu = 0.4999
	E = 1500.0
	mu = E/(2.0*((1.0 + nu)))
	lamb = nu*E/((1.0-2.0*nu)*(1.0+nu))
	c = (3.9e-4, 2.9e-4)
	K = (1.57e-5, 3.75e-2)
	xi = (0.0, 0.0)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, nu=nu, E=E, c=c, K=K, xi=xi)
		
	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	#run(params, experiment="biomed", refinement="space")
	run(params, experiment="biomed", refinement="time")

def run_default(transfer=True):
	""" Run default parameters, all set to 1. """

	if transfer == False:
		print("==== Barenblatt-Biot, default parameters, no transfer ====")
		xi = (0.0, 0.0)
		experiment = "default_no_transfer"

	else:
		print("====== Barenblatt-Biot, default parameters ======")
		xi = (1.0, 1.0)
		experiment = "default"

	# Default parameters
	alpha = (1.0, 1.0)
	nu = None
	E = None
	mu = 0.5
	lamb = 1.0
	c = (1.0, 1.0)
	K = (1.0, 1.0)

	params = dict(alpha=alpha, mu=mu, lamb=lamb, nu=nu, E=E, c=c, K=K, xi=xi)
		
	# Print parameter-info to terminal
	print("alpha =", alpha, "nu =", nu, "E = ", E, "mu =", mu)
	print("lamb = ", lamb, "c = ",  c, "K =", K, "xi = ", xi)

	#run(params, experiment, refinement="space")
	run(params, experiment, refinement="time")
	#run(params, experiment, refinement=None)


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

	run_default(transfer=False)
	run_default(transfer=True)
	run_biomedical()

