from autograd import grad

def f(x):
    return x**2


x = 5.0
fx = f(x)


dfdx = grad(grad(f))

print(dfdx(x))

