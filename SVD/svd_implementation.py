import sympy as s
import itertools

flatten = lambda l: [x for x in itertools.chain(*l)]

def SVD(A): # -> U, S, V
    A = s.Matrix(A)
    (n,m) = A.shape
    AAT = A*A.T # nxn
    ATA = A.T*A # mxm
    AAT_eigs = sorted(AAT.eigenvects(simplify=True), reverse=True)
    ATA_eigs = sorted(ATA.eigenvects(simplify=True), reverse=True)
    AAT_bases = [vect for (eigenval, multiplicity, vect) in AAT_eigs]
    U = s.Matrix([x for x in flatten(AAT_bases)]).reshape(n,n)
    ATA_bases = [vect for (eigenval, multiplicity, vect) in ATA_eigs]
    V = s.Matrix([x for x in flatten(ATA_bases)]).reshape(m,m)
    S = s.zeros(n,m)
    S[:min(n,m),:min(n,m)] = s.diag(*flatten([[s.sqrt(val)]*mul for (val,mul,vec) in ATA_eigs]))
    return (U,S,V)

fm = lambda x: x[0]*(x[1]*x[2])

#s.pprint = lambda x: print(x[0], s.latex(x[1]))

def print_example(A):
    s.pprint(('A', A))
    [s.pprint(x) for x in zip('U S V'.split(), SVD(A))]
    s.pprint(('A\'', fm(SVD(A)).evalf()))

if __name__ == "__main__":
    A = s.Matrix([2,1,1,2]).reshape(2,2)
    print_example(A)
    A = s.Matrix([2,0,0,2]).reshape(2,2)
    print_example(A)
