""" 
Matrices that can be used to differentiate or solve PDEs using finite differences

"""

import numpy as np
from scipy.sparse import csc_matrix, vstack, hstack
from scipy.sparse.linalg import inv, spsolve, splu
from scipy.special import factorial
from fractions import Fraction


def lcm_arr(arr):
    """ Calculate least common multiplier for array of integers
    """
    result = np.lcm(arr[0], arr[1])
    for i in range(2, len(arr)-1):
        result = np.lcm(result, arr[i])

    return result

def wrap(n, N):
    """ if n is an array of indices in an N-element array, this function will 
        map indices that are < 0 and > N-1 to the opposite end of the array
        roughly.
        Used for periodic boundaries

        n must be an array
    """
    n[n  < 0] = N + n[n < 0]
    n[n >= N] = n[n >= N] - N
    return(n)




def stencil(evaluation_points, order = 1, h = 1, fraction = False):
    """ 
    Calculate stencil for finite difference calculation of derivative

    Parameters
    ----------
    evaluation_points: array_like
        evaluation points in regular grid. e.g. [-1, 0, 1] for 
        central difference or [-1, 0] for backward difference
    order: integer, optional
        order of the derivative. Default 1 (first order)
    h: scalar, optional
        Step size. Default 1
    fraction: bool, optional
        Set to True to return coefficients as integer numerators
        and a common denomenator. Be careful with this if you use
        a very large number of evaluation points...

    Returns
    -------
    coefficients: array
        array of coefficients in stencil. Unless fraction is set 
        to True - in which case a tuple will be returned with
        an array of numerators and an integer denominator. If 
        fraction is True, h is ignored - and you should multiply the 
        denominator by h**order to get the coefficients

    Note
    ----
    Algorithm from Finte Difference Coefficient Calculator
    (https://web.media.mit.edu/~crtaylor/calculator.html)
    """

    # calculate coefficients:
    evaluation_points = np.array(evaluation_points).flatten().reshape((1, -1))
    p = np.arange(evaluation_points.size).reshape((-1, 1))
    d = np.zeros(evaluation_points.size)
    d[order] = factorial(order)

    coeffs = np.linalg.inv(evaluation_points**p).dot(d)

    if fraction:
        # format nicely:
        fracs = [Fraction(c).limit_denominator() for c in coeffs]
        denominators = [c.denominator for c in fracs]
        numerators   = [c.numerator for c in fracs]
        cd = lcm_arr(denominators)
        numerators = [int(c * cd / a) for (c, a) in zip(numerators, denominators)]
        return (numerators, cd)
    else:
        return coeffs / h ** order


class Diffmatrix(object):
    """ 
    Class to calculate matrix that differentiates a function defined on a regular 2D grid
    """
    def __init__(self, stencil_steps, I, J, order = 1, axis = 0, skip_boundaries = False, periodic_boundaries = False, h = 1):
        """
        Parameters
        ----------
        stencil_steps: int
            number of steps to go in each direction to evaluate derviative. For example, 
            simple central difference would be 1. A five-point stencil would be 2. 
        I, J: int
            dimensions of grid along axis 0 and 1
        axis: int, optional
            0 or 1 - the axis along which to differentiate. Axis 0 has J elements and
            axis 1 has I elements (corresponding to 'xy' indexing)
        skip_boundaries: bool, optional
            set to True if you want a matrix that is all zeros on the elements that correspond
            to the boundary grid cells. I thought this could be useful when using the matrix
            together with boundary conditions when solving PDEs, but I'm not so sure if it matters
        periodic_boundaries: bool, optional
            Set to True for periodic boundary conditions: On the boundaries, cells at the opposite
            side of the grid will be used to evaluate derivative. If periodic_boundaries is True,
            it overrides the skip_boundaries keyword.
        h: float, optional
            step size in physical units. By default it is 1, and you will have to divide the matrix
            by the actual step size^order to get correct values

        Example
        -------
        Calculate third order derivative of a 2D function that is cubic in x:
        In [1]: I, J = 100, 200
           ...: xx, yy = np.meshgrid(np.linspace(-10, 10, I), np.linspace(-10, 10, J), indexing = 'xy')
           ...: hx, hy = np.diff(xx[0])[0], np.diff(yy[:, 0])[0] # step sizes
           ...: ff = (xx**3 + yy*2).flatten()
           ...: D = Diffmatrix(3, I, J, order = 3).D/hx**3
           ...: np.all(np.isclose(D.dot(ff), 6))
           ...:
        Out[2]: True

        """


        self.rows, self.cols = np.array([], dtype = np.int32), np.array([], dtype = np.int32), 
        self.data = np.array([])
        self.shape = (I * J, I * J)

        if skip_boundaries and not periodic_boundaries:
            start = 1
            ss = slice(1, -1)
        else:
            start = 0
            ss = slice(0, None)


        ind = lambda i, j: np.ravel_multi_index((i, j), (I, J), order = 'F')

        S = stencil_steps

        # index arrays (0 to I, J)
        i_arr = np.arange(I)
        j_arr = np.arange(J)

        # meshgrid versions:
        ii, jj = np.meshgrid(i_arr, j_arr, indexing = 'xy')

        # inner
        points = np.r_[-S:S+1:1]
        coefficients = stencil(points, order = order, h = h)
        if axis == 0:
            if periodic_boundaries:
                i, j = ii  [ss, :], jj  [ss, :]
            else:
                i, j = ii  [ss, S:-S], jj  [ss, S:-S]
        if axis == 1:
            if periodic_boundaries:
                i, j = ii.T[ss, :], jj.T[ss, :]
            else:
                i, j = ii.T[ss, S:-S], jj.T[ss, S:-S]

        iwrap = lambda x: wrap(x, I)
        jwrap = lambda x: wrap(x, J)


        for ll in range(len(points)):
            if axis == 0:
                self.add(ind(i, j), ind(iwrap(i + points[ll]), j), coefficients[ll])
            if axis == 1:
                self.add(ind(i, j), ind(i, jwrap(j + points[ll])), coefficients[ll])


        # boundaries
        for kk in np.arange(start, S)[::-1]:
            if periodic_boundaries: # skip
                break

            # LEFT
            points = np.r_[-kk:S+1:1] 
            coefficients = stencil(points, order = order, h = h)
            if axis == 0:
                i, j = ii  [ss, kk], jj  [ss, kk]
            if axis == 1:
                i, j = ii.T[ss, kk], jj.T[ss, kk]

            for ll in range(len(points)):
                if axis == 0:
                    self.add(ind(i, j), ind(i + points[ll], j), coefficients[ll])
                if axis == 1:
                    self.add(ind(i, j), ind(i, j + points[ll]), coefficients[ll])

            # RIGHT
            points = np.r_[-S:kk+1:1] 
            coefficients = stencil(points, order = order, h = h)
            if axis == 0:
                i, j = ii  [ss, -(kk + 1)], jj  [ss, -(kk + 1)]
            if axis == 1:
                i, j = ii.T[ss, -(kk + 1)], jj.T[ss, -(kk + 1)]

            for ll in range(len(points)):
                if axis == 0:
                    self.add(ind(i, j), ind(i + points[ll], j), coefficients[ll])
                if axis == 1:
                    self.add(ind(i, j), ind(i, j + points[ll]), coefficients[ll])

        self.D = csc_matrix((self.data, (self.rows, self.cols)), shape = self.shape)

    def add(self, rows, cols, data):
        self.rows = np.hstack((self.rows, rows.flatten()))
        self.cols = np.hstack((self.cols, cols.flatten()))
        self.data = np.hstack((self.data, np.full(rows.size, data)))



class Dirichlet(object):
    def __init__(self, b, where, I, J, include_corners = True):
        """ 
        b is the value at the boundaries. 'where' indicates which 
        boundary, and it can be either x0, x1 (left or right on x axis)
        or y0, y1 (bottom, top on y axis)
        """
        self.rows, self.cols = np.array([], dtype = np.int32), np.array([], dtype = np.int32), 
        self.data = np.array([])
        self.shape = (I * J, I * J)

        ind = lambda i, j: np.ravel_multi_index((i, j), (I, J), order = 'F')

        # index arrays (0 to I, J)
        i_arr = np.arange(I)
        j_arr = np.arange(J)

        # meshgrid versions:
        ii, jj = np.meshgrid(i_arr, j_arr, indexing = 'xy')
        
        if include_corners:
            ss = slice(0, None)
        else:
            ss = slice(1, -1)

        if where == 'x0':
            i, j = ii[ss,  0], jj[ss,  0]
        if where == 'x1':
            i, j = ii[ss, -1], jj[ss, -1]
        if where == 'y0':
            i, j = ii[ 0, ss], jj[ 0, ss]
        if where == 'y1':
            i, j = ii[-1, ss], jj[-1, ss]

        self.add(ind(i, j), ind(i, j), 1)

        self.b = csc_matrix((b[ss], (self.rows, np.zeros_like(self.rows))), shape = (self.shape[0], 1))
        self.D = csc_matrix((self.data, (self.rows, self.cols)), shape = self.shape)


    def add(self, rows, cols, data):
        self.rows = np.hstack((self.rows, rows.flatten()))
        self.cols = np.hstack((self.cols, cols.flatten()))
        self.data = np.hstack((self.data, np.full(cols.size, data)))




class Neumann(object):
    def __init__(self, b, where, I, J, stencil_steps, h = 1):
        """ 
        b is the value of the derivative at the boundaries. 'where' indicates which 
        boudary and derivative. It can be the derivative in the x-direction at either
        the left (x0) or right (x1) boundary, or the derivative in the y-direction 
        on either the bottom (y0) or top (y1) boundary - stencil_steps is the number 
        of steps inward in the grid to include in the finite difference estimate of the
        derivative
        """
        self.rows, self.cols = np.array([], dtype = np.int32), np.array([], dtype = np.int32), 
        self.data = np.array([])
        self.shape = (I * J, I * J)
        S = stencil_steps

        ind = lambda i, j: np.ravel_multi_index((i, j), (I, J), order = 'F')

        # index arrays (0 to I, J)
        i_arr = np.arange(I)
        j_arr = np.arange(J)

        # meshgrid versions:
        ii, jj = np.meshgrid(i_arr, j_arr, indexing = 'xy')
        
        ss = slice(0, None)

        if where == 'x0':
            i, j = ii[ :,  0], jj[ :,  0]
            points = np.r_[0:S+1:1] 
            coefficients = stencil(points, order = 1, h = h)

        if where == 'x1':
            i, j = ii[ :, -1], jj[ :, -1]
            points = np.r_[-S:1:1] 
            coefficients = stencil(points, order = 1, h = h)

        if where == 'y0':
            i, j = ii[ 0,  :], jj[ 0,  :]
            points = np.r_[0:S+1:1] 
            coefficients = stencil(points, order = 1, h = h)

        if where == 'y1':
            i, j = ii[-1,  :], jj[-1,  :]
            points = np.r_[-S:1:1] 
            coefficients = stencil(points, order = 1, h = h)

        for ll in range(len(points)):
            #if np.isclose(coefficients[ll], 0):
            #    continue
            if 'x' in where:
                self.add(ind(i, j), ind(i + points[ll], j), coefficients[ll])
            if 'y' in where:
                self.add(ind(i, j), ind(i, j + points[ll]), coefficients[ll])


        self.b = csc_matrix((b, (ind(i, j), np.zeros(b.size))), shape = (self.shape[0], 1))
        self.D = csc_matrix((self.data, (self.rows, self.cols)), shape = self.shape)


    def add(self, rows, cols, data):
        self.rows = np.hstack((self.rows, rows.flatten()))
        self.cols = np.hstack((self.cols, cols.flatten()))
        self.data = np.hstack((self.data, np.full(cols.size, data)))



if __name__ == '__main__': # demo/test code:

    step = 4 # Use a stencil that includes integer point at +- step
    order = 1 # order of differentiation
    I = 120 # number of points in direction of differentiation
    J = 100 # number of points in perpendicular direction


    Dx = Diffmatrix(step, I, J, order = order, axis = 0).D
    Dy = Diffmatrix(step, I, J, order = order, axis = 1).D




    def f(xx, yy):
        return np.sin(xx) * np.cos(yy) * xx

    def dfdx(xx, yy):
        return np.cos(xx) * np.cos(yy) * xx + np.sin(xx) * np.cos(yy)

    def ddfdx(xx, yy):
        return -np.sin(xx) * np.cos(yy) * xx + np.cos(xx) * np.cos(yy) + np.cos(xx) * np.cos(yy)

    def dfdy(xx, yy):
        return -np.sin(xx) * xx * np.sin(yy)

    def ddfdy(xx, yy):
        return -np.sin(xx) * xx * np.cos(yy)

    def delf(xx, yy):
        return ddfdy(xx, yy) + ddfdx(xx, yy)


    x, y = np.linspace(-3* np.pi,  3* np.pi, I), np.linspace(-2* np.pi, 2 * np.pi, J)
    hx = np.diff(x)[0]
    hy = np.diff(y)[0]
    xx, yy = np.meshgrid(x, y, indexing = 'xy')
    ff = f(xx, yy).flatten()

    dfx = Dx.dot(ff) / hx**1
    ddfx = Dx.dot(Dx).dot(ff).flatten() / hx**2

    dfy = Dy.dot(ff) / hy**1
    ddfy = Dy.dot(Dy).dot(ff).flatten() / hy**2



    ####################################
    # Solving Poisson's equation
    #  Find f(x, y), given its Laplacian
    #  and mixed boundary conditions 
    ####################################

    # Laplacian matix:
    Dx2 = Diffmatrix(step, I, J, order = 2, skip_boundaries = True)
    Dy2 = Diffmatrix(step, I, J, order = 2, axis = 1, skip_boundaries = True)
    DL = Dx2.D / hx**2 + Dy2.D / hy**2

    # define boundary conditions (without any Dirichlet boundaries, the
    # solution will only be determined up to a constant I think)
    db_x0 = Dirichlet(f(xx[:, 0], yy[:, 0]), 'x0', I, J) 
    #nb_x0 = Neumann(dfdx(xx[ :,  0], yy[ :,  0]), 'x0', I, J, step, h = hx)
    nb_x1 = Neumann(dfdx(xx[ :, -1], yy[ :, -1]), 'x1', I, J, step, h = hx)
    nb_y0 = Neumann(dfdy(xx[ 0,  :], yy[ 0,  :]), 'y0', I, J, step, h = hy)
    nb_y1 = Neumann(dfdy(xx[-1,  :], yy[-1,  :]), 'y1', I, J, step, h = hy)

    # combine the equations:
    G = vstack((DL, db_x0.D, nb_x1.D, nb_y0.D, nb_y1.D))
    d = hstack((delf(xx, yy).flatten(), db_x0.b.T, nb_x1.b.T, nb_y0.b.T, nb_y1.b.T))

    # transform to square matrix / normal equations
    # (don't think this is how one does this, but it works)
    GTG = G.T.dot(G)
    GTd = G.T.dot(d.T)

    # LU decomposition (this takes a couple of seconds):
    invA = splu(GTG.tocsc())

    # solve (this is fast!)
    ff_estimate = invA.solve(GTd.toarray())

    # ff_estimate should be very similar to f(xx, yy).flatten()

