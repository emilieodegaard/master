from sympy2fenics import str2sympy, sympy2exp, grad, div, sym, eps
from sympy import symbols, sympify, simplify, diff, eye, sin, cos, pi

# Symbols for differentiation
x, y, t, alpha1, alpha2, mu, lamb, c1, c2, K1, K2, xi1, xi2 = symbols("x y t alpha1 alpha2 mu lamb c1 c2 K1 K2 xi1 xi2")

def get_source_terms(u, p):
	""" Function that computes the source terms f and g=[g1,g2] using sympy, 
	returning FEniCS expressions in C-code. 

	Exact solutions u and p=[p1,p2] must be given.

	Find (f,g) given (u,p) s.t.

		 -div(sigma(u)) + alpha1*grad(p1) + alpha2*grap(p2) = f
		c1*p1_t + alpha1*div(u_t) - K1*div(grad(p1)) + xi1*(p1-p2) = g1
		c2*p2_t + alpha2*div(u_t) - K2*div(grad(p2)) + xi2*(p2-p1) = g2

	"""
	u = str2sympy(u); p = str2sympy(p)
	p1 = p[0]; p2 = p[1]

	f = -div(sigma(u)) + alpha1*grad(p1).T + alpha2*grad(p2).T
	g1 = c1*diff(p1,t) + alpha1*div(diff(u,t)) - K1*div(grad(p1)) + xi1*(p1-p2)
	g2 = c2*diff(p2,t) + alpha2*div(diff(u,t)) - K2*div(grad(p2)) + xi2*(p2-p1)
	g = [sympy2exp(simplify(g1)),sympy2exp(simplify(g2))]
	return sympy2exp(simplify(f)), g

def sigma(u):
	""" Stress tensor """
	return 2.0*mu*eps(u) + lamb*div(u)*eye(len(u))


def str2exp(u):
	""" Converts strings to FEniCS string expresions """
	u = str2sympy(u)
	return sympy2exp(u)

if __name__ == "__main__":
	u = "(cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*t), sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*t))"
	p = "(sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*t), cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*t))"
	f, g = get_source_terms(u,p)






