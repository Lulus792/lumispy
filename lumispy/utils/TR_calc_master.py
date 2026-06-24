# -*- coding: utf-8 -*-
"""
Created on Thu Nov 08 09:18:33 2018

@author: coene
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 08:39:04 2016

@author: coene
"""

import numpy as np
import matplotlib.pyplot as plt
import TRfarfieldcalculation as TR
import normalize as nm
from odemis.dataio import hdf5
import calctools as ct

plt.close('all')



optical_constants1 = np.loadtxt('optical_constants//Au_johnson.txt')
optical_constants2 = np.loadtxt('optical_constants//au_werner.csv',skiprows=1)
optical_constants3 = np.loadtxt('optical_constants//Au_Palik1.txt',skiprows=1)
optical_constants4 = np.loadtxt('optical_constants//Au_SC.dat',skiprows=1)

mcpeak_n = np.loadtxt('optical_constants//McPeak_n.csv',delimiter=',',skiprows=1)
mcpeak_k = np.loadtxt('optical_constants//McPeak_k.csv',delimiter=',',skiprows=1)
wl5 = mcpeak_n[:,0]*1e3
epsilon5 = (mcpeak_n[:,1]+1j*mcpeak_k[:,1])**2

rakic_n = np.loadtxt('optical_constants//Rakic_n.csv',delimiter=',',skiprows=1)
rakic_k = np.loadtxt('optical_constants//Rakic_k.csv',delimiter=',',skiprows=1)
wl6 = rakic_n[:,0]*1e3
epsilon6 = (rakic_n[:,1]+1j*rakic_k[:,1])**2

johnson_ag = np.loadtxt('optical_constants//Ag_johnson.csv',delimiter=',',skiprows=1)
wl7 = johnson_ag[:,0]*1e3
epsilon7 = (johnson_ag[:,1]+1j*johnson_ag[:,3])**2


wl1 = optical_constants1[:,0]*1e3
wl2 = optical_constants2[:,0]*1e3
wl3 = np.flipud(1239.84/optical_constants3[:,0])
wl4 = np.flipud(1239.84/optical_constants4[:,0])
epsilon1 = (optical_constants1[:,1]+1j*optical_constants1[:,2])**2
epsilon2 = (optical_constants2[:,1]+1j*optical_constants2[:,2])**2
epsilon3 = np.flipud(optical_constants3[:,1]+1j*optical_constants3[:,2])
epsilon4 = np.flipud((optical_constants4[:,1]+1j*optical_constants4[:,2])**2)
#epsilon3 = np.flipud((optical_constants3[:,1]+1j*optical_constants3[:,2])**2)
#
#plot of gold optical constants
plt.figure()
plt.plot(wl1,np.real(epsilon1),'r',wl1,np.imag(epsilon1),'r',wl2,np.real(epsilon2),'k',wl2,np.imag(epsilon2),'k',wl3,np.real(epsilon3),'b',wl3,np.imag(epsilon3),'b',
         wl4,np.real(epsilon4),'g',wl4,np.imag(epsilon4),'g')
plt.axis([300, 1000,-40,15])

#plot of Al optical constants
plt.figure()
plt.plot(wl5,np.real(epsilon5),'r',wl5,np.imag(epsilon5),'r',wl6,np.real(epsilon6),'k',wl6,np.imag(epsilon6),'k')
plt.axis([300, 1000,-90,60])

#plot of ag optical constants
plt.figure()
plt.plot(wl7,np.real(epsilon7),'r',wl7,np.imag(epsilon7),'r')
plt.axis([300, 1000,-90,60])


#
electronenergy = 30.
epsilon_eVornm = 'nm'
thetalistdeg = np.linspace(0,90,300)
wllist = np.linspace(200,1100,801)
#
##test=np.interp(1240/wavelengthlist,np.flipud(optical_constants[:,0]),np.flipud(optical_constants[:,1]))
#
#
##test=np.tile(wavelengthlist,[np.size(thetalistdeg),1])
##print(np.shape(test))

#flip data again to fix interpolation (code works with increasing eV, not decreasing eV)
optical_data1 = np.flipud(np.transpose(np.vstack((wl1,np.real(epsilon1),np.imag(epsilon1)))))
optical_data2 = np.flipud(np.transpose(np.vstack((wl2,np.real(epsilon2),np.imag(epsilon2)))))
optical_data3 = np.flipud(np.transpose(np.vstack((wl3,np.real(epsilon3),np.imag(epsilon3)))))
optical_data4 = np.flipud(np.transpose(np.vstack((wl4,np.real(epsilon4),np.imag(epsilon4)))))
optical_data5 = np.flipud(np.transpose(np.vstack((wl5,np.real(epsilon5),np.imag(epsilon5)))))
optical_data6 = np.flipud(np.transpose(np.vstack((wl6,np.real(epsilon6),np.imag(epsilon6)))))
optical_data7 = np.flipud(np.transpose(np.vstack((wl7,np.real(epsilon7),np.imag(epsilon7)))))


TRemprob_nm1,TRemprob_eV1, Itr_nm1, Itr_eV1 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data1,epsilon_eVornm)
TRemprob_nm2,TRemprob_eV2, Itr_nm2, Itr_eV2 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data2,epsilon_eVornm)
TRemprob_nm3,TRemprob_eV3, Itr_nm3, Itr_eV3 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data3,epsilon_eVornm)
TRemprob_nm4,TRemprob_eV4, Itr_nm4, Itr_eV4 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data4,epsilon_eVornm)
TRemprob_nm5,TRemprob_eV5, Itr_nm5, Itr_eV5 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data5,epsilon_eVornm)
TRemprob_nm6,TRemprob_eV6, Itr_nm6, Itr_eV6 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data6,epsilon_eVornm)
TRemprob_nm7,TRemprob_eV7, Itr_nm7, Itr_eV7 = TR.TRfarfieldcalculation_nm(thetalistdeg,wllist,electronenergy,optical_data7,epsilon_eVornm)

plotorder = -7

plt.figure()
plt.plot(wllist,TRemprob_nm1/10**plotorder,linewidth=2)
plt.xlabel('Wavelength (nm)',fontsize=18)
plt.ylabel('Emission Probability (10$^{'+np.str(plotorder)+'}$ nm$^{-1}$)',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.locator_params(axis = 'x', nbins = 6)  
plt.locator_params(axis = 'y', nbins = 6) 
plt.axis([200, 1100,0,4])
plt.savefig("Au_em_prob_JohnsonChristy.png", dpi=200,bbox_inches='tight',frameon=False,transparent=True)

plt.figure()
plt.plot(wllist,TRemprob_nm6/10**plotorder,linewidth=2)
plt.xlabel('Wavelength (nm)',fontsize=18)
plt.ylabel('Emission Probability (10$^{'+np.str(plotorder)+'}$ nm$^{-1}$)',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.locator_params(axis = 'x', nbins = 6)  
plt.locator_params(axis = 'y', nbins = 6) 
plt.axis([200, 1100,0,8])
plt.savefig("Al_em_prob_Rakic.png", dpi=200,bbox_inches='tight',frameon=False,transparent=True)

plt.figure()
plt.plot(wllist,TRemprob_nm7/10**plotorder,linewidth=2)
plt.xlabel('Wavelength (nm)',fontsize=18)
plt.ylabel('Emission Probability (10$^{'+np.str(plotorder)+'}$ nm$^{-1}$)',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.locator_params(axis = 'x', nbins = 6)  
plt.locator_params(axis = 'y', nbins = 6) 
plt.axis([200, 1100,0,8])
#plt.savefig("Ag_JC.png", dpi=200,bbox_inches='tight',frameon=False,transparent=True)

#export johnson & christy
dataset_Au = np.transpose(np.vstack((wllist,TRemprob_nm1)))
np.savetxt('emission_prob_Au_JohnsonChristy.txt',dataset_Au)

#export Al from rakic
dataset_Al = np.transpose(np.vstack((wllist,TRemprob_nm6)))
np.savetxt('emission_prob_Al_Rakic.txt',dataset_Al)

dataset_Ag = np.transpose(np.vstack((wllist,TRemprob_nm7)))
#np.savetxt('emission_prob_Ag_JC.txt',dataset_Ag)

#plt.figure()
#plt.plot(thetalistdeg,Itr_eV[:,100])
#
#plt.figure()
#plt.imshow(Itr_eV)
#
#plt.figure()
#plt.imshow(Itr_nm)
#plt.axis([300, 1000, 0, 1.1*TRemprob.max()/10**plotorder_theory])
#plt.legend(['Palik','Single-crystal'],fontsize=18)
#plt.xlabel('Wavelength (nm)',fontsize=18)
#plt.ylabel('Emission Probability (10$^{'+np.str(plotorder_theory)+'}$ nm$^{-1}$)',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.locator_params(axis = 'x', nbins = 4)  
#plt.locator_params(axis = 'y', nbins = 4) 