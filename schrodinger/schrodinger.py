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
basis_size=5
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

potential=create_potential_tensor(args)
delta_x=potential[1][1]-potential[1][0]
def lambda_1(i):
    return lambda x: 1 
def lambda_sin(i):
    return lambda x: math.sin(math.ceil(i/2)*x)
def lambda_cos(i):
    return lambda x: math.cos((i/2)*x)

def basis_set(args):
    array=[None for i in range(args.basis_size)]
    for i in range(args.basis_size):
        if i==0:
            array[i]=lambda_1(i)
        elif i%2==1:
            array[i]=lambda_sin(i)
        else:
            array[i]=lambda_cos(i)
    return array
basis=basis_set(args)
def evaluate_basis(args,basis,potential):
    numerical=[[] for i in range(args.basis_size)]
    for i in range(len(basis)):
        for j in range(len(potential[2])):
            numerical[i].append(basis[i](potential[2][j]))
    tensor=tf.transpose(tf.convert_to_tensor(numerical))
    return tensor
num_basis=evaluate_basis(args,basis,potential)   
def riemann_sum(tensor,delta=delta_x):
    temp=tf.reduce_sum(tensor,axis=0)
    return delta_x*temp

def projection(potential, basis,args):
    rhs=tf.Variable(riemann_sum(potential[0])*riemann_sum(basis))
    lhs=[[] for i in range(args.basis_size)]
    for i in range(args.basis_size):
        temp=basis*basis[i]
        lhs[i]=riemann_sum(temp)
    lhs_t=tf.convert_to_tensor(lhs)
    return tf.linalg.solve(lhs_t,tf.reshape(rhs,[args.basis_size,1]))
print(projection(potential, num_basis, args))

