import argparse
import tensorflow as tf
import math

tf.enable_eager_execution()
#default values
c=1.0
v=1.0
basis_size=10
domain=(-1,1)

def get_arguments():  #pragma: no cover
    parser=argparse.ArgumentParser(description='schrodinger')
    parser.add_argument('--c', type=float, default=c, help='kinetic energy')
    parser.add_argument('--v',type=float,default=v, help='potential energy')
    parser.add_argument('--domain',type=tuple, default=basis_size, help='tuple length 2')
    parser.add_argument('--basis_size', type=int, default=basis_size, help='size of basis set')
    return parser.parse_args()

def hamiltonian(coefficients, V=v, C=c):
    hammy=[V]+ [(i**2)*C*coefficients[i] for i in range(1, len(coefficients))]
    return hammy

def hamiltonian_mat(basis_size=basis_size, domain=domain, C=c, V=v):
    k=domain[1] #symmetric around origin
    empty_mat=[[] for i in range(basis_size)]
    for i in range(basis_size):
        for j in range(basis_size):
            p1=2*v*math.sin(k*j)/k
            if i==j:
                if j==0:
                    p2=0
                else:
                    p2=C*(j**2)*(k+math.sin(2*k*j)/(2*j))
            else:
                p2=0
            empty_mat[i].append(p1+p2)
    hmat=tf.convert_to_tensor(empty_mat)
    return hmat



print(hamiltonian_mat())
