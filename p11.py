#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:42:58 2021

@author: dgiron
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
from uncertainties import ufloat, unumpy
from uncertainties.umath import *

tau_er_led = 10.10
tau_er_las = 8.2

p_0_led = 116
p_0_laser = 912


def f(x, a, b):
    return a * x + b

def ajuste(x, y, xlabel, ylabel, label, col):
    """
    Makes a linear fit y = ax + +b for the x-y data

    Parameters
    ----------
    x : np.ndarray
        x-axis.
    y : np.ndarray
        y-axis.
    xlabel : str
        label for the x-axis.
    ylabel : str
        label for the y-axis.
    label : str
        label for the data (LED or laser).
    col : color
        color of the data in the plot.

    Returns
    -------
    ufloat(ppot[0], perr[0])
        ufloat (mean value + uncertainty) of the slope

    """

    ppot, pcov = cf(f, x, y)
    perr = np.sqrt(np.diag(pcov))
    xx = np.linspace(min(x), max(x), 1000)
    yy = f(xx, *ppot)

    plt.plot(x, y, col+'o', label=label)
    plt.plot(xx, yy, col+'-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    return ufloat(ppot[0], perr[0])

def main():
    datos = np.genfromtxt('datos.txt', delimiter=',')
    datos = datos[1:]

    L = datos[:, 0]
    # First value of laser tau was removed because it was incoherent, so
    # a new distances vector (L2) is defined without that measure.
    L2 = L[1:]

    tau_led = datos[:, 1]
    tau_las = datos[:, 2]


    tau_f_led = np.sqrt(tau_led ** 2 - tau_er_led ** 2)
    tau_f_las = np.sqrt(tau_las[1:] ** 2 - tau_er_las ** 2)

    sigma_las = tau_f_las / 2.56
    sigma_led = np.sqrt(((tau_f_led[1:] / 2.56) ** 2 - (tau_f_las / 2.56) ** 2)) * 1000

    br_las = 0.26 / (tau_f_las / 2.56)
    br_led = 0.26 / (tau_f_led / 2.56)

    datos2 = np.genfromtxt('datos2.txt', delimiter=',')
    datos2 = datos2[1:]

    # Data for the attenuation coefficient
    L3 = datos2[:, 0]
    p_led = datos2[:, 1]
    p_las = datos2[:, 2]

    def tau():
        """
        Plots the rise-time as a function of the distance for both laser and LED

        Returns
        -------
        None.

        """
        plt.clf()
        plt.plot(L, tau_f_led, 'go', label='LED')

        plt.plot(L2, tau_f_las, 'ro', label='Laser')
        plt.xlabel(r'$L$/km')
        plt.ylabel(r'$\tau$/ns')
        plt.legend()
        plt.grid()
        plt.savefig('tau.png', dpi=720)
        plt.show()

    def d_laser():
        """
        Plots the temporal width for the laser as a function of the distance.
        Also, it calculates the intermodal dispersion coefficient.

        Returns
        -------
        None.

        """

        plt.clf()
        d_laser = ajuste(L2, sigma_las, r'$L$/km', r'$\sigma_i$/ns', 'Laser', 'r')
        plt.savefig('d_laser.png', dpi=720)
        plt.show()
        print('D_las = ', d_laser, 'ns/km')

    def d_led():
        """
        Plots the temporal width for the LED as a function of the distance.
        Also, it calculates the  absolute value of the material dispersión coefficient.

        Returns
        -------
        None.

        """
        plt.clf()
        d_led = ajuste(L2, sigma_led, r'$L$/km', r'$\sigma_{mat}$/ps', 'LED', 'g')
        plt.savefig('d_led.png', dpi=720)
        plt.show()

        print('D_led = ', d_led/21, 'ps/(km nm)')

    def BR():
        """
        Plots the Bit-rate as a function of distance for both laser and LED

        Returns
        -------
        None.

        """
        plt.clf()
        plt.plot(L, br_led, 'go',label='LED')
        plt.plot(L2, br_las, 'ro',label='Laser')
        plt.ylabel(r'$BR$/bits/ns')
        plt.xlabel(r'$L$/km')
        plt.legend()
        plt.grid()
        plt.savefig('bitrate.png', dpi=720)
        plt.show()

    def alpha():
        """
        Calculates the attenuation coefficient (alpha) for laser and LED via a linear
        plot, which is also shown.

        Returns
        -------
        None.

        """
        plt.clf()
        y_led = 10 * np.log10(p_0_led/p_led)
        y_las = 10 * np.log10(p_0_laser/p_las)

        alpha_laser = ajuste(L3, y_led, r'$L$/km', r'$10\cdot\log_{10}(\frac{P(0)}{P(L)})$', 
                             'LED', 'g')
        alpha_led = ajuste(L3, y_las, r'$L$/km', r'$10\cdot\log_{10}(\frac{P(0)}{P(L)})$', 
                           'Laser', 'r')

        plt.savefig('at_coef.png', dpi=720)
        plt.show()
        print('alpha LED', alpha_led)
        print('alpha Laser', alpha_laser)

    # Uncomment the lines below to execute the functions above

    # tau()
    # d_laser()
    # d_led()
    # BR()
    # alpha()
main()
