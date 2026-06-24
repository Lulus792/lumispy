# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 08:30:18 2016

@author: coene
"""
import numpy as np
import matplotlib.pyplot as plt



def TRfarfieldcalculation_nm(thetalistdeg,wavelengthlist,electronenergy,epsilon1list,epsilon_eVornm):

# Calculation of Transition Radiation (TR) from a single interface, as a function of wavelength instead of energy.
# This code uses the "TRfarfieldcalculation" code, as energy space and atomic
# units are the basis for the equations, see that function for more details.
# This function calculates the wavelength and angle dependence of the asymptotic part 
# of the (electric) far field amplitude and the related and emission probability, for
# an electron traversing the interface between two materials. 
# For the equations used here, the first medium must be vacuum (epsilon of medium 0 = epsilon0 = 1), 
# the second one is the material to be studied (epsilon of medium 1 = epsilon1, to be given as input).
# All of the input lists are preferably given as column vectors, but the
# code will recognize and flip them if they are rows. This is because the
# calculation is done as an element-wise multiplication of matrices with
# dimensions (# of wavelengths, # of angles), so the input vectors have to
# be replicated in the right way to obtain the correct dimensions.
 
# The input parameters are as follows:
# - thetalistdeg:       list of emission angles, in degrees (typically 0 to 90, with as many steps as desired, but at least 2 values)
# 
# - wavelengthlist:     list of wavelengths to calculate, in nm (a single wavelength is also possible) 
#                       (be careful here, with respect to the range over which epsilon1 is defined, extrapolate at your own risk)
# 
# - electronenergy:     the electron energy, in keV
# 
# - epsilon1list:       list of epsilon values for the material the electron impinges upon, 
#                       it should be formatted as 3 columns -> [frequency (hbar*omega) in eV, Re{eps}, Im{eps}]
# 
# - interpmethod:       Interpolation method used for epsilon1. Common options are: 'linear', 'pchip', 'spline'. 
#                       Javier's c++ code seems to use a linear method, but pchip gives a more smooth response 
#                       (using Gold as a benchmark, the difference with Javier's code is negligible when using linear,
#                       and a maximum difference of 2% for pchip, but with much better agreement over almost all of angle-energy space)
# 
# - epsilon_eVornm:     A string, 'eV' or 'nm', that indicates whether the list
#                       of epsilons is given as a function of energy or wavelength
# 
# The output values are as follows:
# - TREmProbeV:   the emission probability, as a function of energy, per unit nm (photons/electron/nm)
# 
# - Etr:          the (complex) asymptotic part of the far field TR amplitude, 
#                 equivalent to the equation just before eq.56 of Javier's review (the quantity f_H)
#                 The amplitude is in atomic units of E field.
#                 For a fully complete result at a given electron-observer distance,
#                 the exp(i*k*r)/r factor needs to be taken into account, see the
#                 TRfieldcalculation function, which also converts the field to SI
#                 units (V/m/eV).


# DEFINE CONSTANTS
    
    
    c=299792458 # speed of light in m/s
    h_eVs=4.135667516e-15 # Planck's constant in eV*s
    c_meV=c*h_eVs # speed of light in m*eV
   
# eV_au=0.0367493; % 1 eV in atomic units
# au_Efield=5.14220624e11; % 1 atomic unit of E field, in V/m (volt/m)
# a0_au=0.529177208; % Bohr radius a0 in Angstroms
# nm_au=10/a0_au; % 1 nm in atomic units
# c_au=137.03599971; % speed of light in atomic units
# au_eV=27.2113834; % 1 atomic unit of enery in eV


## DEFINE VARIABLES FOR EQUATIONS

    frequencylisteV=np.flipud(c_meV*1e9/wavelengthlist)# list of frequencies in eV, converted from the list of wavelengths
  
    
#    if np.size(thetalistdeg,0)==1: # if thetalistdeg is a row vector
#        thetalistdeg=np.transpose(thetalistdeg) # this converts it to a column vector
#    
#
#    if np.size(frequencylisteV,0)==1: # if frequencylisteV is a row vector
#        frequencylisteV=np.transpose(frequencylisteV) # this converts it to a column vector
#    
#
#    if np.size(epsilon1list,0)==3: # if epsilon1list has rows for energy, Re{eps}, Im{eps}
#        epsilon1list=np.transpose(epsilon1list) # this converts it to columns
    
    

    if epsilon_eVornm=='nm':  # if epsilon1list is defined as a function of wavelength, these have to be converted to energy
        epsilon1list[:,0]=c_meV*1e9/epsilon1list[:,0] # Thanks to the previous if statement, the first column should in all cases contain the wavelengths, which we convert to energy.
    
    #elif strcmp(epsilon_eVornm,'eV'): # if epsilon1list is defined as a function of energy we don't need to change it
    # This part of the statement is not necessary as we do nothing, but added for completeness
    
#    plt.figure()
#    plt.plot(frequencylisteV)
#    plt.plot(epsilon1list[:,0])

## CALCULATION OF THE TRANSITION RADIATION FAR FIELD COMPONENT & EMISSION PROBABILITY

    TREmProbeV, Itr_eV=TRfarfieldcalc(thetalistdeg,frequencylisteV,electronenergy,epsilon1list) # Calculation of the emission probability and electric far field amplitude using the "Master" function
    
    jacobian=frequencylisteV**2/(c_meV*1e9)
    TREmProbnm=jacobian*TREmProbeV # Emission probability, converted to per unit nm
    
    #convert to nm
    jacobianmat=np.tile(jacobian,[np.size(thetalistdeg),1])
    
    Itr_nm=jacobianmat*Itr_eV
    #flip the data back
    return np.flipud(TREmProbnm), np.flipud(TREmProbeV), np.fliplr(Itr_nm), np.fliplr(Itr_eV)




def TRfarfieldcalc(thetalistdeg,frequencylisteV,electronenergy,epsilon1list):
    
# Calculation of Transition Radiation (TR) from a single interface.
# This corresponds to the general code in the asymptotic limit of r -> infinity 
# (this is the asymptotic part of the electric far field amplitude, f from the equations of Javier)
# This code is based on the equations in Javier's review (pages 239 & 240) 
# (alternatively the 2016 PRB of Benjamin, Albert & Javier, pages 155412-6&7, E & H are equivalent for TR).
# This function calculates the energy/frequency and angle dependence of the asymptotic part 
# of the far field amplitude and the related emission probability, for
# an electron traversing the interface between two materials. 
# For the equations used here, the first medium must be vacuum (epsilon of medium 0 = epsilon0 = 1), 
# the second one is the material to be studied (epsilon of medium 1 = epsilon1, to be given as input).
# All of the input lists are preferably given as column vectors, but the
# code will recognize and flip them if they are rows. This is because the
# calculation is done as an element-wise multiplication of matrices with
# dimensions (# of frequencies, # of angles), so the input vectors have to
# be replicated in the right way to obtain the correct dimensions.
 
# The input parameters are as follows:
# - thetalistdeg:       list of emission angles, in degrees (typically 0 to 90, with as many steps as desired, but at least 2 values)
# 
# - frequencylisteV:    list of energies/frequencies (hbar*omega) to calculate, in eV (a single energy is also possible) 
#                       (be careful here, with respect to the range over which epsilon1 is defined, extrapolate at your own risk)
# 
# - electronenergy:     the electron energy, in keV
# 
# - epsilon1list:       list of epsilon values for the material the electron impinges upon, 
#                       it should be formatted as 3 columns -> [frequency (hbar*omega) in eV, Re{eps}, Im{eps}]
# 
# - interpmethod:       Interpolation method used for epsilon1. Common options are: 'linear', 'pchip', 'spline'. 
#                       Javier's c++ code seems to use a linear method, but pchip gives a more smooth response 
#                       (using Gold as a benchmark, the difference with Javier's code is negligible when using linear,
#                       and a maximum difference of 2% for pchip, but with much better agreement over almost all of angle-energy space)
 
# The output values are as follows:
# - TREmProbeV:  the emission probability, as a function of energy, per unit eV (photons/electron/eV)
# 
# - Etr:         the (complex) asymptotic part of the far field TR amplitude, 
#                equivalent to the equation just before eq.56 of Javier's review (the quantity f_H)
#                The amplitude is in atomic units of E field.
#                For a fully complete result at a given electron-observer distance,
#                the exp(i*k*r)/r factor needs to be taken into account, see the
#                TRfieldcalculation function, which also converts the field to SI
#                units (V/m/eV).

# DEFINE CONSTANTS

# c=299792458; % speed of light in m/s
# h_eVs=4.135667516e-15; % Planck's constant in eV*s
# c_meV=c*h_eVs; % speed of light in m*eV
# eV_au=0.0367493; % 1 eV in atomic units
# au_Efield=5.14220624e11; % 1 atomic unit of E field, in V/m (volt/m)
# a0_au=0.529177208; % Bohr radius a0 in Angstroms
# nm_au=10/a0_au; % 1 nm in atomic units

    c_au=137.03599971 # speed of light in atomic units
    au_eV=27.2113834 # 1 atomic unit of enery in eV


# DEFINE INPUT PARAMETERS (frequency, angles, electron energy & velocity, material parameters

#    if np.size(thetalistdeg,0)==1: # if thetalistdeg is a row vector
#        thetalistdeg=np.transpose(thetalistdeg) # this converts it to a column vector


#    if np.size(frequencylisteV,0)==1: # if frequencylisteV is a row vector
#        frequencylisteV=np.transpose(frequencylisteV) # this converts it to a column vector


#    if np.size(epsilon1list,0)==3: # if epsilon1list has rows for energy, Re{eps}, Im{eps}
#        epsilon1list=np.transpose(epsilon1list) # this converts it to columns


    Thetalistrad=thetalistdeg*np.pi/180. # list of emission angles, in radian
    frequencylistau=frequencylisteV/au_eV # list of frequencies (hbar*omega) to calculate, in atomic units
    omega=np.tile(frequencylistau,[np.size(Thetalistrad),1]) # matrix of frequencies for all angles for faster computation
    Thetarad=np.transpose(np.tile(Thetalistrad,[np.size(frequencylistau),1])) # matrix of angles for all frequencies for faster computation

    
    electronvelocity=c_au*np.sqrt(1-1/(1000*electronenergy/au_eV/c_au**2+1)**2) # electron velocity, in atomic units

    epsilon0=1 # the epsilon of the material from which the electron arrives, must be vacuum for the validity of these equations

#    epsilon1interp=[frequencylisteV,interp1(epsilon1list(:,0),epsilon1list[:,1],frequencylisteV,interpmethod,'extrap'),
#    interp1(epsilon1list[:,0],epsilon1list[:,2],frequencylisteV,interpmethod,'extrap')] # interpolated list of epsilon values, the 3 columns are [frequency (hbar*omega) in eV, Re{eps}, Im{eps}]
#    print(frequencylisteV)
#    print(np.flipud(epsilon1list[:,0]))
#    print(np.shape(epsilon1list))
    #warning, this interpolation function gives bogus results when a decreasing x list is given
    epsilon1interpreal=np.interp(frequencylisteV,epsilon1list[:,0],epsilon1list[:,1])
    epsilon1interpimag=np.interp(frequencylisteV,epsilon1list[:,0],epsilon1list[:,2])
    epsilon1values=epsilon1interpreal+1j*epsilon1interpimag # list of interpolated complex epsilon1 values of the impinging material
    epsilon1=np.tile(epsilon1values,[np.size(Thetalistrad),1]) # matrix of interpolated epsilon1 for all frequencies and angles for faster computation

    # Check used optical constants, in particular check extrapolation errors
#    plt.figure()
#    plt.plot(frequencylisteV,epsilon1interpreal,frequencylisteV,epsilon1interpimag)


   # DEFINE VARIABLES FOR EQUATIONS

    kk=(omega/c_au)**2 #k^2 in free space
    kk0=kk*epsilon0 # k0^2, in the incoming medium 0 (vacuum)
    kk1=kk*epsilon1 # k1^2, in the impinging medium 1 (defined by epsilon1)

    Q=np.sin(Thetarad)*np.sqrt(kk) # Q, the momentum parallel to the interface (of the electron field)
    qq=Q**2+(omega/electronvelocity)**2 # q^2, with q the total momentum (of the electron field)
    qz0=np.sqrt(kk0-Q**2) # qz0, the momentum perpendicular to the interface (of the electron field) in medium 0 (vacuum)
    qz1=np.sqrt(kk1-Q**2) # qz1, the momentum perpendicular to the interface (of the electron field) in medium 1 (defined by epsilon1)


    #CALCULATION OF THE TRANSITION RADIATION FAR FIELD COMPONENT

    D=2*1j*Q/c_au/(qz0*epsilon1+qz1*epsilon0) ##factor equivalent to eq.54 of Javier's review

    mu1=(((qz1*epsilon0-(omega/electronvelocity)*epsilon1)/(kk0-qq))-((qz1*epsilon0-(omega/electronvelocity)*epsilon0)/(kk1-qq))) # factor equivalent to eq.52 of Javier's review

    Etr=(1j*np.sqrt(kk))*np.cos(Thetarad)*D*mu1 # far field TR amplitude, equivalent to the equation just before eq.56 of Javier's review

    Itr=np.abs(Etr)**2 # far field TR intensity
    
    
#    print(np.shape(np.tile(np.sin(Thetalistrad),[1,np.size(frequencylisteV)])))
    Itr_integral=Itr*np.transpose(np.tile(np.sin(Thetalistrad),[np.size(frequencylisteV),1]))

    #CALCULATION OF THE TRANSITION RADIATION EMISSION PROBABILITY
       
    TRProbIntegral=np.abs(np.trapezoid(Thetarad,Itr_integral,axis=0)) # Integral part of eq.56 of Javier's review
    TREmProbeV=(c_au/(2*np.pi*frequencylisteV))*TRProbIntegral # Emission probability, per unit eV, complete part of eq.56 of Javier's review


    return TREmProbeV, Itr


