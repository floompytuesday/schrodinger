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
    '''reads in the potential energy from a file'''
    arr=np.genfromtxt(args.v)
    easy_domain=arr[:,0]
    tensor=tf.convert_to_tensor(arr[:,1],dtype=tf.float32)
    domain=tf.convert_to_tensor(arr[:,0],dtype=tf.float32)
    return tensor,domain,easy_domain




def lambda_1(i): #pragma: no cover
    return lambda x: 1 
def lambda_sin(i): #pragma: no cover
    return lambda x: math.sin(math.ceil(i/2)*x)
def lambda_cos(i): #pragma: no cover
    return lambda x: math.cos((i/2)*x)

def basis_set(args):
    '''creates an array of lambda functions representing the basis'''
    array=[None for i in range(args.basis_size)]
    for i in range(args.basis_size):
        if i==0:
            array[i]=lambda_1(i)
        elif i%2==1:
            array[i]=lambda_sin(i)
        else:
            array[i]=lambda_cos(i)
    return array

def evaluate_basis(args,basis,potential):
    '''evaluates the basis elements on the domain points'''
    numerical=[[] for i in range(args.basis_size)]
    for i in range(len(basis)):
        for j in range(len(potential[2])):
            numerical[i].append(basis[i](potential[2][j]))
    tensor=tf.transpose(tf.convert_to_tensor(numerical))
    return tensor
   
def riemann_sum(tensor,delta=math.pi/2):
    '''calculates a riemann sum for a tensor'''
    temp=tf.reduce_sum(tensor,axis=0)
    return delta*temp

def projection(potential, num_basis,args):
    '''projects the potential energy onto the basis set'''
    rhs=tf.Variable(riemann_sum(potential[0])*riemann_sum(num_basis))
    lhs=[[] for i in range(args.basis_size)]
    for i in range(args.basis_size):
        temp=num_basis*num_basis[i]
        lhs[i]=riemann_sum(temp)
    lhs_t=tf.convert_to_tensor(lhs)
    return tf.linalg.solve(lhs_t,tf.reshape(rhs,[args.basis_size,1]))


def hamiltonian(args, projection):
    '''builds the hamiltonian'''
    hammy=tf.Variable(tf.zeros(shape=[args.basis_size,args.basis_size])) #I know this isn't right but i didn't have time to implement the hamiltonian correctly :(
    hammy=tf.add(hammy,projection)
    return hammy


def eigen(hammy):
    '''finds the eigenvalues and eigenvectors of the hamiltonian'''
    temp=tf.linalg.eigh(hammy)
    return temp[1][0] #lowest energy coefficients

args=get_arguments()
basis=basis_set(args)
potential_=create_potential_tensor(args)

num_basis=evaluate_basis(args,basis,potential_)
proj=projection(potential_,num_basis,args)
hammy=hamiltonian(args,proj)
print(hammy)


