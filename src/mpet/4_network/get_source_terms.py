from sympy2fenics import str2sympy, sympy2exp, grad, div, sym, eps
from sympy import symbols, sympify, simplify, diff, eye, sin, cos, pi

# Symbols for differentiation
x, y, t, alpha1, alpha2, alpha3, alpha4, mu, lamb, c1, c2, c3, c4, K1, K2, K3, K4, xi1, xi2, xi3, xi4, xi5, xi6 = symbols("x y t alpha1 alpha2 alpha3 alpha4 mu lamb c1 c2 c3 c4 K1 K2 K3 K4 xi1 xi2 xi3 xi4 xi5 xi6")

def get_source_terms(u, p):
	""" Function that computes the source terms f and g=[g1,g2] using sympy, 
	returning FEniCS expressions in C-code. 

	Exact solutions u and p=[p1,p2] must be given.

	Find (f,g) given (u,p) s.t.

	-div(sigma(u)) + alpha1*grad(p1) +.. + alpha4*grad(p4) = f
		 c1*p1_t + alpha1*div(u_t) - K1*div(grad(p1)) + S1 = g1
		 c2*p2_t + alpha2*div(u_t) - K2*div(grad(p2)) + S2 = g2
		 c3*p3_t + alpha3*div(u_t) - K3*div(grad(p3)) + S3 = g3
		 c4*p4_t + alpha4*div(u_t) - K4*div(grad(p4)) + S4 = g4

	where,
			S1 = xi21*(p1-p2) + xi31*(p1-p3) + xi41*(p1-p4)
			S2 = xi12*(p2-p1) + xi32*(p2-p3) + xi42*(p2-p4)
			S3 = xi13*(p3-p1) + xi23*(p3-p2) + xi43*(p3-p4)
			S4 = xi14*(p4-p1) + xi24*(p4-p2) + xi34*(p4-p3)

	"""
	u = str2sympy(u); p = str2sympy(p)
	p1 = p[0]; p2 = p[1]; p3 = p[2]; p4 = p[3]

	# Transfer parameters
	S1 = xi1*(p1-p2) + xi2*(p1-p3) + xi3*(p1-p4)
	S2 = xi1*(p2-p1) + xi4*(p2-p3) + xi5*(p2-p4)
	S3 = xi2*(p3-p1) + xi4*(p3-p2) + xi6*(p3-p4)
	S4 = xi3*(p4-p1) + xi5*(p4-p2) + xi6*(p4-p3)

	f = -div(sigma(u)) + alpha1*grad(p1).T + alpha2*grad(p2).T + alpha3*grad(p3).T + alpha3*grad(p3).T
	g1 = c1*diff(p1,t) + alpha1*div(diff(u,t)) - K1*div(grad(p1)) + S1
	g2 = c2*diff(p2,t) + alpha2*div(diff(u,t)) - K2*div(grad(p2)) + S2
	g3 = c3*diff(p3,t) + alpha3*div(diff(u,t)) - K3*div(grad(p3)) + S3
	g4 = c4*diff(p4,t) + alpha4*div(diff(u,t)) - K4*div(grad(p4)) + S4

	g = [sympy2exp(simplify(g1)),sympy2exp(simplify(g2)),sympy2exp(simplify(g3)),sympy2exp(simplify(g4))]

	return sympy2exp(simplify(f)), g

def sigma(u):
	""" Stress tensor """
	return 2.0*mu*eps(u) + lamb*div(u)*eye(len(u))

def str2exp(u):
	""" Converts strings to FEniCS string expresions """
	u = str2sympy(u)
	return sympy2exp(u)

if __name__ == "__main__":
	u = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))"
	p = "(sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t), \
		  sin(pi*x)*sin(pi*y)*sin(2*pi*t), sin(pi*x)*sin(pi*y)*sin(2*pi*t))"
	f, g = get_source_terms(u,p)






