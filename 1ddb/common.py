

import numpy as np
import os

import  matplotlib.pyplot as plt

from arkane.statmech import ScanLog
from rmgpy.statmech.torsion import HinderedRotor

from arc.common import read_yaml_file
from arc.species import ARCSpecies
from arc.species.converter import get_element_mass


def get_center_of_mass(coordinates, masses, top = None):
    """
    Logic: choose the relevant masses and coords, then create a triplet of the masses as an array:
    coords:
    |x_1 , y_1, z_1|
    |x_2 , y_2, z_2|
    |x_3 , y_3, z_3|
    |x_4 , y_4, z_4|
    masses:
    |m_2 , m_2, m_2|
    |m_3 , m_3, m_3|
    |m_4 , m_4, m_4|
    Then, performing  element wise multiplication between them:
    |m_1x_1 , m_1y_1, m_1z_1|
    |m_2x_2 , m_2y_2, m_2z_2|
    |m_3x_3 , m_3y_3, m_3z_3|
    |m_4x_4 , m_4y_4, m_4z_4|
    Summing in axis 0:
    |sum(m_ix_i), sum(m_iy_i), sum(m_iy_i)|
    and dividing by the sum of masses, to recive:
    |sum(m_ix_i)\sum(m_i), sum(m_iy_i)\sum(m_i), sum(m_iy_i)\sum(m_i)|
    """
    relevant_coords = coordinates if top is None else coordinates[[atom -1 for atom in top]]
    relevant_masses = masses if top is None else masses[[atom -1 for atom in top]]
    mass_matrix = np.array([relevant_masses, relevant_masses, relevant_masses]).T
    
    return np.multiply(relevant_coords, mass_matrix).sum(axis = 0)/masses.sum()
    

def get_principle_moment_of_inertia(mass, coordinates, pivots, top1):

        # The total number of atoms in the geometry
        n_atoms = mass.shape[0]
        # Check that exactly one pivot atom is in the specified top
        if pivots[0] not in top1 and pivots[1] not in top1:
            raise ValueError('No pivot atom included in top; you must specify which pivot atom belongs with the'
                             ' specified top.')
        elif pivots[0] in top1 and pivots[1] in top1:
            raise ValueError('Both pivot atoms included in top; you must specify only one pivot atom that belongs'
                             ' with the specified top.')
        elif 0 in top1:
            raise ValueError('Top must be one indexed, top1: {}'.format(top1))
        # Enumerate atoms in other top
        top2 = [i + 1 for i in range(n_atoms) if i + 1 not in top1]
        # Determine centers of mass of each top
        top1_center_of_mass = get_center_of_mass(coordinates, masses, top = top1)
        top2_center_of_mass = get_center_of_mass(coordinates, masses, top = top2)

        axis = (top1_center_of_mass - top2_center_of_mass)
        axis /= np.linalg.norm(axis)
        # Determine moments of inertia of each top
        I1 = 0.0
        for atom in top1:
            r1 = coordinates[atom - 1, :] - top1_center_of_mass
            r1 -= np.dot(r1, axis) * axis
            I1 += mass[atom - 1] * np.linalg.norm(r1) ** 2
        I2 = 0.0
        for atom in top2:
            r2 = coordinates[atom - 1, :] - top2_center_of_mass
            r2 -= np.dot(r2, axis) * axis
            I2 += mass[atom - 1] * np.linalg.norm(r2) ** 2
        return I1 * I2 / (I1 + I2)


def fix_scan(angle, v_list):
    sort = np.argsort(angle)
    angle, v_list = angle[sort], v_list[sort]

    zero = np.where(v_list == 0)[0][0]
    new = np.zeros(shape = v_list.shape)
    for i in range(v_list.shape[0]):
        new[i-zero] = v_list[i]
    return np.arange(0, 2*np.pi, 2*np.pi/new.shape[0]), new


def get_partition_function(angles, vlist, xyz, top, pivot, masses, symmetry = 1):
    angles, vlist = fix_scan(angles, vlist)
    inertia = get_principle_moment_of_inertia(mass = masses, coordinates=xyz, pivots = pivot, top1 = top)
    
    cosine_rotor = HinderedRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
    cosine_rotor.fit_cosine_potential_to_data(angles, vlist)
    fourier_rotor = HinderedRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
    fourier_rotor.fit_fourier_potential_to_data(angles, vlist)
    Vlist_cosine = np.zeros_like(angles)
    Vlist_fourier = np.zeros_like(angles)
    
    for i in range(angles.shape[0]):
        Vlist_cosine[i] = cosine_rotor.get_potential(angles[i])
        Vlist_fourier[i] = fourier_rotor.get_potential(angles[i])
    
    rms_cosine = np.sqrt(np.sum((Vlist_cosine - vlist) **2) /
                     (len(vlist) - 1))
    rms_fourier = np.sqrt(np.sum((Vlist_fourier - vlist) **2) /
                     (len(vlist) - 1))
    rotor = cosine_rotor if rms_cosine < rms_fourier else fourier_rotor

    T = np.arange(300, 3000, 10)
    partition = list()
    for t in T:
        partition.append(rotor.get_partition_function(t))
    return T, partition
