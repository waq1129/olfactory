from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import matplotlib.pyplot as plt

import edward as ed
from edward.util import rbf
from edward.models import Bernoulli, MultivariateNormalTriL, Normal

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import mvn_kron
import mvn_kron_autograd
from mvn_ed import MatrixNormal
from mvn_kron_ed import MatrixNormal_Kron
from mvn_kron_fast_ed import MatrixNormal_Kron_fast
from mvn_kron_fast_ed_sample import MatrixNormal_Kron_fast_sample

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import grad
from autograd import value_and_grad
from autograd.misc.optimizers import adam

import scipy.io
from scipy import stats
from scipy.optimize import minimize



