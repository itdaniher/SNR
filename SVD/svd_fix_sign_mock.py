import sympy as s
from svd_implementation import SVD

A = s.Matrix(range(1,4))
U,S,V = SVD(A)
Y = A-U*S*V

for k in range(U.cols):
    for j in range(Y.cols):
        S_ =  (U.col(k).T * Y.col(j))
        print('sign', s.functions.sign(S_[0])*S_*S_**2)

for k in range(V.cols):
    for i in range(Y.T.rows):
        S_ =  (V.col(k).T * Y.row(i).T)
        print('sign', s.functions.sign(S_[0])*S_*S_**2)

print(U*S*V)

