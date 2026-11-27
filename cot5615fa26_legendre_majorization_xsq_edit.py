#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:57:40 2026

@author: anand
"""

import numpy as np
from matplotlib import pyplot as plt


def objective(x, epsilon):
    return epsilon / 4 * x**4 - 0.5 * x**2


def legendre_maj(x, y_k, epsilon):
    return epsilon / 4 * x**4 - x * y_k + 0.5 * y_k**2


N = 1000
x_min = -15
x_max = 15
x = np.linspace(x_min, x_max, N)
epsilon = 1.0e-02
E_obj = objective(x, epsilon)

M = 7
y = np.linspace(x_min, x_max, M)

fig = plt.figure()
plt.plot(x, E_obj)
plt.xlabel(r"$x$")
plt.ylabel(r"$E_{obj}(x)=\frac{1}{4}\epsilon x^4 -\frac{1}{2} x^2$ ")
plt.savefig("cot5615fa26_quartic_obj.pdf")
plt.show()

fig = plt.figure()
for y_k in y:
    E_legmaj = legendre_maj(x, y_k, epsilon)
    plt.ylim([-50, 200])
    plt.plot(x, E_legmaj)
plt.plot(x, E_obj)
plt.xlabel(r"$x$")
plt.ylabel(r"$E_{legmaj}(x,y)=\frac{1}{4}\epsilon x^4 -xy + \frac{1}{2}y^2$")
plt.savefig("cot5615fa26_legmaj.pdf")
plt.show()
