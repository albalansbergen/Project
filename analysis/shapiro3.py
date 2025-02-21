import numpy as np
import matplotlib.pyplot as plt
import linecache
import pickle
import sys
import pandas as pd
import os

from scipy.stats import linregress
from scipy import stats



def get_first_index_smaller_than(arr, value, grace):
    T_index = 1
    P_index = 1
    for i in range(len(arr)):
        if arr[i,0] <= grace:
            T_index = i
        elif arr[i,1] <= value:
            P_index = i
            break
    
    return T_index, P_index 
    

def get_last_index_smaller_than(arr, value, grace):
    T_index = -2
    P_index = -2
    for i in range(len(arr)-1, -1, -1):
        if arr[i,0] >= grace:
            T_index = i
        elif arr[i,1] <= value:
            P_index = i
            break
            
    return T_index, P_index  

def shapiro(energy, sysType, setn, setN,  label="Label", p_value=0.05, grace=40, verbose=0, save='shapiro'):
    """
    Function to find Tg from a .xvg file containing density-temperature data.

    Parameters
    ----------
    file_path : str
        Path to the .xvg file with density-temperature data.
    label : str
        Label for the plots.
    p_value : float
        p-value threshold for determining the transition points.
    grace : float or None
        Grace period temperature range. If None, defaults to 20% of the full temperature span.
    verbose : int
        Verbosity level:
        - 0: Minimal output
        - 1: Key results
        - 2: Detailed diagnostics.
    save : str or None
        Path to save the plot. If None, plot will be displayed but not saved.

    Returns
    -------
    Tg : float
        Glass transition temperature, calculated as the intersection of liquid and glass fits.
    """

    #data = np.genfromtxt(energy, skip_header=27)
    data = pd.DataFrame(energy, columns=['Time', 'Temperature', 'Pressure', 'Volume', 'Density'])
    data = data.sort_values(by='Temperature')

    Temperature = data['Temperature'].values
    Density = data['Density'].values
    density_array = np.array([Temperature, Density])

    return Tg_finder(density_array, label=label, p_value=p_value, grace=grace, verbose=verbose, save=save, sysType=sysType, setn=setn, setN=setN)

def Tg_finder(density, label="Label", p_value=0.05, grace=40, verbose=0, save='shapiro', sysType=None, setn=None, setN=None):
    """
    This function is taken from the original paper
    Function that finds Tg for a density temperature series by cumputing 
    their intersection of longest linear regimes for the glass and liquid 
    phases given a critical p-value and grace period.

    Parameters
    ----------
    density : Numpy array 
        Density vs. thermostat of sim with shape of 
        [Temperature, Density]^N with legth=N of cooling ramp.
        The function expects that the series goes from 
        low T, High D ==> High T, Low D 
    Label=label : str
        Simple string to label graph and save file   
    p_value=0.05 : float
        Level of confidence for break in linear behaviour.
        If this is not met, by default the temparture range
        picked will be the last 3 density-temperature pairs for
        the liquid phase nad the first 3 for the liquid phase. 
    grace=None : 
        grace period to waive p-value criteria starting from the 
        upper and lower temperature bound or when at least 3 
        points for each fitting range. If none, by default it will
        pick 20% of the spanned temperture range.       
    verbose=0 :
        Outputs for diagnotsics. verbose=0 will not print anything.
        Verbose 1 will print the label, grace period and result
        and figure. verbose=2 will print density array, p-values 
        and picked fitting range for each phase.     
    save=None :
        Path to save figure. 
           
    Returns
    -------
    Tg : float
        Computed Tg by the intersection of linear fits for each
        phase fitted according to the tempereature range defined 
        by p vs. thermostat. In unit of provide temperature scale.
    """
    Temperature = density[0]
    Density = density[1]
    
    skip = 3 # number of points to start with

    #specify grace 
    if grace is None:
         grace = (Temperature[-1]-Temperature[0]) * 0.20 #pick by default 20% of T-range 
         print(f"grace: {grace}")

    if verbose > 0:
        print(f"\n{label}, grace: {grace}")
        
    #glass fitting range
    G_p_value = np.zeros((len(Temperature)-skip+1,2))
    for t in range(len(Temperature)-skip+1):
        #fits of phase
        m, b, R, p, std = linregress(Temperature[:skip+t], Density[:skip+t])
        
        #residuals
        Residuals = Density[:skip+t] - m*Temperature[:skip+t]+b
        
        #pvalue      
        G_p_value[t][0] = Temperature[skip+t-1]

        if len(Residuals) == 3: 
            if round(stats.shapiro(Residuals).statistic, 2) < 0.75:
                G_p_value[t][1] = 0 
            else:
                G_p_value[t][1] = 1-np.pi/6*np.arccos(np.sqrt(stats.shapiro(Residuals).statistic))
        else:
            G_p_value[t][1] = stats.shapiro(Residuals).pvalue
        
    G_Tstart, G_phase_limit = get_first_index_smaller_than(G_p_value, p_value, G_p_value[0][0]+grace)
    G_Temperature = Temperature[:np.where(Temperature==G_p_value[G_phase_limit][0])[0][0]]
    G_Density = Density[:np.where(Temperature==G_p_value[G_phase_limit][0])[0][0]]

    # Diagnostics
    if verbose==2:
        print("\nGlass Diagnostics")
        print("Temperature:", Temperature)
        print("Glass p-values:\n", G_p_value)
        print("Glass grace ends:", G_p_value[G_Tstart][0])
        print(f"Glass phase fitting range [{Temperature[0]}, {G_Temperature[-1]}]")
        print("Glass phase Fitting points:\n ", np.array([G_Temperature, G_Density]).T)
        
    #liquid phase fitting range
    L_p_value = np.zeros((len(Temperature)-skip+1, 2))
    for t in range(len(Temperature)-skip, -1, -1):
        #the fits of phase
        m, b, R, p, std = linregress(Temperature[t:], Density[t:])
        
        #residuals
        Residuals = Density[t:] - m*Temperature[t:]+b
        
        #pvalue
        L_p_value[t][0] = Temperature[t]

        if len(Residuals) == 3:
            if round(stats.shapiro(Residuals).statistic, 2) < 0.75:
                L_p_value[t][1] = 0 
            else:
                L_p_value[t][1] = 1-np.pi/6*np.arccos(np.sqrt(stats.shapiro(Residuals).statistic))
        else:
            L_p_value[t][1] = stats.shapiro(Residuals).pvalue

    L_Tstart, L_phase_limit = get_last_index_smaller_than(L_p_value, p_value, L_p_value[-1][0]-grace)
    L_Temperature = Temperature[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]+1:]
    L_Density = Density[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]+1:]

    if verbose==2:
        print("\nLiquid Diagnostics")
        print("Temperature:", Temperature)
        print("Liquid p-values:\n", L_p_value)
        print("Liquid grace ends: ", L_p_value[L_Tstart][0])
        print(f"Liquid phase fitting range: [{Temperature[-1]}, {L_Temperature[0]}]")
        print("Liquid phase fitting points:\n", np.array([L_Temperature, L_Density]).T)      
    
    #Compute Tg
    G_m, G_b, G_R, trash, std = linregress(G_Temperature, G_Density)
    L_m, L_b, L_R, trash, std = linregress(L_Temperature, L_Density)
    Tg = (L_b - G_b)/(G_m - L_m)
    if verbose > 0:
        print("\nTg: ", Tg)

    if verbose >= 1 or save is not None:

        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=75)

        # fig.suptitle(f'{label}, My Method')
        
        # A  
        axs[0, 0].plot(G_Temperature, [G_m*T+G_b for T in G_Temperature], label="Glass Range", lw=2, color = '#FF1493')
        axs[0, 0].plot(Temperature, [G_m*T+G_b for T in Temperature], lw=1, linestyle='dotted', color='#FF1493', alpha = 0.7)
        axs[0, 0].plot(L_Temperature, [L_m*T+L_b for T in L_Temperature], label="Liquid Range", lw=2, color = '#39FF14')
        axs[0, 0].plot(Temperature, [L_m*T+L_b for T in Temperature], linestyle='dotted', color='#39FF14', alpha = 0.7)
        axs[0, 0].set_ylim((density[1].min(), density[1].max()))
        axs[0, 0].vlines(Tg, density[1].min(), density[1].max(), label=r'$T_g$='+f"{round(Tg, 2)}K", color='#00008B')
        axs[0, 0].set_ylabel(r"Density [$g/cm^3$]")
        axs[0, 0].set_xlabel("Temperature [K]")
        axs[0, 0].legend(loc=3)
        axs[0, 0].scatter(Temperature, density[1], label="Density", s=1, marker="x", color='dimgray')
            
        # B
        axs[0, 1].scatter(L_p_value[L_phase_limit+1:,0],L_p_value[L_phase_limit+1:,1], s=30, color='#39FF14')
        axs[0, 1].scatter(L_p_value[:,0], L_p_value[:,1], s=12, color="dimgray")
        axs[0, 1].set_ylabel("p-value")
        axs[0, 1].set_xlabel("Temperature [K]")
        axs[0, 1].vlines(L_p_value[L_Tstart][0], -2, 2, label=f"grace period", color='#39FF14', alpha = 0.6)
        axs[0, 1].hlines(p_value, Temperature.min(), Temperature.max(), label=f"p-value = {p_value}", color="grey", lw=1, linestyle='dotted')
        axs[0, 1].set_xlim((Temperature.min(), Temperature.max()))
        axs[0, 1].legend(loc=2)
        axs[0, 1].set_ylim((0,1))

        # C
        axs[1, 0].scatter(Temperature, Density - [G_m*T+G_b for T in Temperature], marker="x", s=15, color="#ECD3E0")
        axs[1, 0].scatter(Temperature, Density - [L_m*T+L_b for T in Temperature], marker="x", s=15, color="#D0E0CD")
        axs[1, 0].scatter(G_Temperature, G_Density - [G_m*T+G_b for T in G_Temperature], label="Glass", s=15, color="#FF1493")
        axs[1, 0].scatter(L_Temperature, L_Density - [L_m*T+L_b for T in L_Temperature], label="Liquid", s=15, color="#39FF14")
        Glass_Residuals = np.array([Density - [G_m*T+G_b for T in Temperature]])
        Liquid_Residuals = np.array([Density - [L_m*T+L_b for T in Temperature]])
        Residuals = np.hstack((Glass_Residuals,Liquid_Residuals))
        axs[1, 0].set_ylabel("Residuals")
        axs[1, 0].set_xlabel("Temperature [K]")
        axs[1, 0].legend() 
    
        # D
        axs[1, 1].scatter(G_p_value[:G_phase_limit,0], G_p_value[:G_phase_limit,1], s=30, color='#FF1493')
        axs[1, 1].scatter(G_p_value[:,0], G_p_value[:,1], s=12, color="dimgray")
        axs[1, 1].vlines(G_p_value[G_Tstart][0], -2, 2, label=f"grace period", color='#FF1493', alpha = 0.6)
        axs[1, 1].hlines(p_value, Temperature.min(), Temperature.max(), label=f"p-value = {p_value}", color="grey", lw=1, linestyle='dotted')
        axs[1, 1].set_xlabel("Temperature [K]")
        axs[1, 1].set_ylabel("p-value")
        axs[1, 1].legend(loc=1)
        axs[1, 1].set_xlim((Temperature.min(), Temperature.max()))
        axs[1, 1].set_ylim((0,1))
    

        axs[0, 0].text(0.5, 0.94, "A", transform=axs[0, 0].transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")
        axs[0, 0].text(0.6, 0.90, label, transform=axs[0, 0].transAxes, ha="left", va="bottom", fontsize=12)
        axs[0, 1].text(0.5, 0.94, "B", transform=axs[0, 1].transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")
        axs[1, 0].text(0.5, 0.94, "C", transform=axs[1, 0].transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")
        axs[1, 1].text(0.5, 0.94, "D", transform=axs[1, 1].transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")
    
  
        plt.tight_layout()
        fig.suptitle(sysType +' with n=' + setn +' and N='+ setN, fontsize=16, y = 0.98)
        fig.subplots_adjust(top=0.9)

    if save is not None:
        #plt.savefig(f"{save}/{label}_{grace}_{np.round(p_value,2)}.png", dpi=300)
        plt.savefig(f"shapiro.png", dpi=300)
        #plt.show()
    #elif verbose==1 or verbose==2:
        #plt.show()        
        
    return Tg
#os.chdir(r'D:\pythongraphs\results_pdms\pdms18100')
#os.chdir(r'D:\pythongraphs\random stuff dump\pdms10\results\rand5n28e8')
#data = pd.DataFrame("energy001.xvg", columns=['Time', 'Temperature', 'Pressure', 'Volume', 'Density'])
#Tg = shapiro("energycr0001.xvg", sysType='RAND', setn='28', setN='100', label="Tg", verbose=2)
#print(f"Determined Tg: {Tg} K")
