from dolfin import *
import numpy as np

# Remove logging
set_log_active(False)

"""
Adaptive solver for the Poisson equation

	- div(grad(u)) = f 

on unit square with homogenous Dirichlet boundary conditions. 

"""

TOL = 1e-4 				# error tolerance
REFINE_RATIO = 0.5 		# refine 50% of cells in each iteration
MAX_ITER = 10			# max number of iterations

print "TOL = %.2e, REFINE_RATIO = %g, MAX_ITER = %d" % (TOL, REFINE_RATIO, MAX_ITER)

def adaptive_solver(N, adaptive=True):

	# Mesh
	mesh = UnitSquareMesh(N,N)

	for i in range(MAX_ITER):
		# Function space
		V = FunctionSpace(mesh, "CG", 1)

		# Functions
		u = TrialFunction(V)
		v = TestFunction(V)

		# Source term
		f = Expression("4*pi*pi*sin(2*pi*x[1])", degree=2)

		# Boundary condition
		u0 = Constant(0.0)
		bc = DirichletBC(V, u0, "on_boundary")

		# Variational form
		a = inner(grad(u), grad(v))*dx
		L = f*v*dx

		# Numerical solution
		u_ = Function(V)
		solve(a==L, u_, bc)

		# Error estimator
		h = CellSize(mesh)
		n = FacetNormal(mesh)
		DG = FunctionSpace(mesh, "DG", 0)
		w = TestFunction(DG)

		# Element and edge residuals
		R_K = div(grad(u_))-f
		R_E = jump(grad(u_),n)
		R = w*(h*R_K)**2*dx + avg(h)*avg(w)*R_E**2*dS

		gamma = Function(DG)
		gamma = assemble(R)
		assemble(R, tensor=gamma.vector())
		gamma = gamma.vector()
		E = np.sqrt((gamma*gamma).sum())
		
		print("Level %d: E = %g (TOL = %g)" % (i, E, TOL))

		set_log_active(True)

		# Check convergence
		if E < TOL:
			info("Success, solution converged after %d iterations" % i)
			break

		# Mark cells for refinement
		cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
		gamma_0 = sorted(gamma, reverse=True)[int(len(gamma)*REFINE_RATIO)]
		gamma_max = MPI.max(mesh.mpi_comm(), gamma_0)
		for c in cells(mesh):
			cell_markers[c] = gamma[c.index()] > gamma_max

		# Refine mesh
		if adaptive:
			mesh = refine(mesh, cell_markers)

			# Plot adaptive mesh
			#mesh_plot = File("fig/adaptive_mesh_%d.pvd" %i)
			#mesh_plot << mesh

		else:
			mesh = refine(mesh)
			
			# Plot uniform mesh
			#mesh_plot = File("fig/uniform_mesh_%d.pvd" %i)
			#mesh_plot << mesh

		set_log_active(False)




def simple_test(N):
	mesh = UnitIntervalMesh(N)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "CG", 2)

	f = Expression("1", degree=1); f = project(f, W)
	u = Expression("x[0]*x[0] + 1", degree=2); u = project(u, W)

	h = CellSize(mesh)
	n = FacetNormal(mesh)
	DG = FunctionSpace(mesh, "DG", 0)
	w = TestFunction(DG)

	RK = div(grad(u))-f
	RE = jump(grad(u),n)

	RK_exact = Expression("1", degree=1)
	RE_exact = Expression("0", degree=1)

	R = w*(h*RK)**2*dx + avg(h)*avg(w)*RE**2*dS
	R_exact = w*(h*RK_exact)**2*dx + avg(h)*avg(w)*RE_exact**2*dS

	gamma = Function(DG)
	gamma_exact = Function(DG)
	assemble(R, tensor=gamma.vector())
	assemble(R_exact, tensor=gamma_exact.vector())

	gamma = gamma.vector().array()
	gamma_exact = gamma_exact.vector().array()

	assert np.allclose(gamma, gamma_exact), "Verification not successful"



def discont_test(N):
	mesh = UnitIntervalMesh(N)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "CG", 2)

	f = Expression("1", degree=1); f = project(f, W)
	u = Expression("x[0]*x[0] + 1", degree=2); u = project(u, W)

	h = CellSize(mesh)
	n = FacetNormal(mesh)
	DG = FunctionSpace(mesh, "DG", 0)
	w = TestFunction(DG)

	RK = div(grad(u))-f
	RE = jump(grad(u),n)

	RK_exact = Expression("1", degree=1)
	RE_exact = Expression("0", degree=1)

	R = w*(h*RK)**2*dx + avg(h)*avg(w)*RE**2*dS
	R_exact = w*(h*RK_exact)**2*dx + avg(h)*avg(w)*RE_exact**2*dS

	gamma = Function(DG)
	gamma_exact = Function(DG)
	assemble(R, tensor=gamma.vector())
	assemble(R_exact, tensor=gamma_exact.vector())

	gamma = gamma.vector().array()
	gamma_exact = gamma_exact.vector().array()

	assert np.allclose(gamma, gamma_exact), "Verification not successful"


def run():
	N = 4
	adaptive_solver(N, adaptive=False)
	#simple_test(N)
	#discont_test(N)

if __name__ == "__main__":
	run()

