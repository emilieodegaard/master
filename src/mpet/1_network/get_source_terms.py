from sympy2fenics import str2sympy, sympy2expr, grad, div, sym, eps
from sympy import symbols, sympify, simplify, diff, eye, sin, cos, pi, exp

# Symbols for differentiation
x, y, t, alpha, mu, lamb, c, K = symbols("x y t alpha mu lamb c K")

def get_source_terms(u, p, version="biot"):
	""" Function that computes the source terms f and g using sympy, 
	returning FEniCS expressions in C-code. 

	Find (f,g) given (u,p) s.t.

			f = - div(sigma(u)) + alpha*grad(p)
			g = c*p_t + alpha*div(u_t) - K*div(grad(p))

	Args: 
		u, p (str) : functions

	Returns: 
		f, g (str) : FEniCS string expressions

	"""
	u = str2sympy(u); p = str2sympy(p)

	u_t = diff(u,t)
	div_u = div(u)
	p_t = diff(p,t)
	grad_p = grad(p)
	sigma_u = sigma(u)

	f = -div(sigma_u) + alpha*grad_p.T
	g = c*p_t + alpha*div(u_t) - K*div(grad_p)

	return sympy2expr(simplify(f)), sympy2expr(simplify(g))

def sigma(u):
	""" Stress tensor """

	# Infer dimension
	dim = len(u)

	# Handle zero-matrices
	for i in range(dim):
		if u[i] == 0:
			u = 0*eye(dim)*eye(dim)
			return u
			break
	
	return 2.0*mu*eps(u) + lamb*div(u)*eye(dim)


def str2expr(u):
	""" Converts strings to FEniCS string expresions """
	u = str2sympy(u)
	return sympy2expr(u)

if __name__ == "__main__":
	u = "(cos(pi*x)*sin(pi*y), sin(pi*x)*cos(pi*y))"
	p = "sin(pi*x)*sin(pi*y)" 

	u = "(t*t*sin(pi*x)*sin(2*pi*y), t*sin(3*pi*x)*sin(4*pi*y))"
	p = "t*exp(1-t)*x*y*(1-x)*(1-y)"
	u = "(0.0,0.0)"
	p = "x*y*(1-x)*(1-y)"
	f, g = get_source_terms(u,p)











