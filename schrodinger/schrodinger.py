import argparse
import tensorflow as tf
import math
from os import path
import numpy as np
import math




tf.enable_eager_execution()
#default values
c=1.0
v=open('potential_energy.dat','r')
basis_size=3
domain=(0, 3*math.pi)

def get_arguments():  #pragma: no cover
    parser=argparse.ArgumentParser(description='schrodinger')
    parser.add_argument('--c', type=float, default=c, help='constant')
    parser.add_argument('--v',type=argparse.FileType('r'), default=v)
    parser.add_argument('--domain',type=tuple, default=basis_size, help='tuple length 2')
    parser.add_argument('--basis_size', type=int, default=basis_size, help='size of basis set')
    return parser.parse_args()

args=get_arguments()

def create_potential_tensor(args):
    array=np.genfromtxt(args.v)
    test=array[:,0]
    tensor=tf.convert_to_tensor(array[:,1],dtype=tf.float32)
    domain=tf.convert_to_tensor(array[:,0],dtype=tf.float32)
    return tensor,domain,test

stuff=create_potential_tensor(args)

def basis_set(args):
    array=[None for i in range(args.basis_size)]
    for i in range(args.basis_size):
        if i==0:
            array[i]=lambda x: 1
        elif i%2==1:
            array[i]=lambda x: math.sin(math.ceil(i/2)*x)
        else:
            array[i]=lambda x: math.cos((i/2)*x)
    return array
basis=basis_set(args)
def evaluate_basis(args,basis,stuff):
    numerical=[[] for i in range(args.basis_size)]
    for i in range(len(basis)):
        for j in range(len(stuff[2])):
            numerical[i].append(basis[i](stuff[2][j]))
    tensor=tf.transpose(tf.convert_to_tensor(numerical))
    return tensor
    

print(evaluate_basis(args,basis,stuff))






























'''def hamiltonian(coefficients, V=v, C=c):
    hammy=[V]+ [(i**2)*C*coefficients[i] for i in range(1, len(coefficients))]
    return hammy'''

'''def hamiltonian_mat(basis_size=basis_size, domain=domain, C=c, V=v):
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
    return hmat'''

