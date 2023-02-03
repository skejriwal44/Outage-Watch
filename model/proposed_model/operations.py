"""Operations Module

This module provides the custom defined loss function for the model.

This module contains the following functions:

    * exp: returns exp for an input.
    * slice_parameter_vectors: returns unpacked list of parameter
                               vectors.
    * gnll_loss: computes the negative log likelihood loss of
                 gaussian mixture.
"""
import numpy as np
import keras
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def exp(input):
    """Returns exp for an input.

    Args:.
        input: A tensor whose exponential is to be calculated.
    Returns:
        Returns the exponential of input tensor.
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.exp(input))   #exp or nnelu??


def slice_parameter_vectors(parameter_vector,components=3,no_parameters=3):
    """ Returns an unpacked list of paramter vectors.
    
    Args:.
        parameter_vector: A list containing mean, std and mixing coefficients.
        components: An integer representing number of components.
        no_parameters: An integer representing number of parameters.
    Returns:
        Returns the exponential of input tensor.
    """
    return [parameter_vector[:,i*components:(i+1)*components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the gaussian mixture negative log likelihood loss.
    
    Args:.
        y: A tensor containing the true value of the forecast.
        parameter_vector: A list containing mean, std and mixing coefficients.
    Returns:
        Returns the negative log likelihood loss of the Gaussian mixture distribution
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector) # Unpack parameter vectors
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,       
            scale=sigma))
    y= tf.cast(y, tf.float32)
    log_likelihood = gm.log_prob(tf.transpose(y)) # Evaluate log-probability of y
    return -tf.reduce_mean(log_likelihood, axis=-1)