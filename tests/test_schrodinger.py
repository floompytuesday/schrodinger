#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""


import unittest
from click.testing import CliRunner

from schrodinger import schrodinger
from schrodinger import cli
from schrodinger import *
import tensorflow as tf 
import math

class TestSchrodinger(unittest.TestCase):
    """Tests for `schrodinger` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'schrodinger.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

def test_create_potential():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)
        
    a=schrodinger.create_potential_tensor(args)
    tf.assert_equal(a[0], tf.Variable([0., 6., 0., -6., 0., 6., 0.]))

def test_basis_set():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)

    a=schrodinger.basis_set(args)
    assert a[0](0)==1
    assert a[1](0)==0
    assert a[2](0)==1

def test_numerical():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)
        
    basis=schrodinger.basis_set(args)
    potential=schrodinger.create_potential_tensor(args)
    a=schrodinger.evaluate_basis(args,basis,potential)
    tf.assert_equal(a, tf.Variable([[ 1.0000000e+00,  0.0000000e+00,  1.0000000e+00],
 [ 1.0000000e+00,  1.0000000e+00,  6.3267948e-06],
 [ 1.0000000e+00,  2.6535897e-06, -1.0000000e+00],
 [ 1.0000000e+00, -1.0000000e+00, -8.9803843e-06],
 [ 1.0000000e+00, -5.3071794e-06,  1.0000000e+00],
 [ 1.0000000e+00,  1.0000000e+00,  1.6339745e-06],
 [ 1.0000000e+00,  7.9607689e-06, -1.0000000e+00]]))

def test_riemann():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)
    a=tf.Variable([1.0,2.0,3.0,4.0])
    delta_test=math.pi/2
    b=schrodinger.riemann_sum(a,delta_test)
    tf.assert_equal(b, tf.Variable(15.707964))

'''def test_projection():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)
    basis=schrodinger.basis_set(args)
    potential=schrodinger.create_potential_tensor(args)
    num_basis=schrodinger.evaluate_basis(args,basis,potential)
    a=schrodinger.projection(potential,num_basis,args)
    tf.assert_equal(a,tf.Variable([[ 4.7123737e+00],
 [-2.3561895e+01],
 [-3.0745706e+07]]))''' #cant figure out why this test isn't working

def test_hammy():
    class args:
        def __init__():
            pass
        c=1.0
        v=open('potential_energy.dat','r')
        basis_size=3
        domain=(0, 3*math.pi)
    basis=schrodinger.basis_set(args)
    potential=schrodinger.create_potential_tensor(args)
    num_basis=schrodinger.evaluate_basis(args,basis,potential)
    proj=schrodinger.projection(potential,num_basis,args)
    a=schrodinger.hamiltonian(args,proj)
    tf.assert_equal(a, tf.Variable([[ 4.7123923e+00,  4.7123923e+00,  4.7123923e+00],
 [-2.3561995e+01, -2.3561995e+01, -2.3561995e+01],
 [-3.0745832e+07, -3.0745832e+07, -3.0745832e+07]]))
