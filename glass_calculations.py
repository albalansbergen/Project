import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
import opt_curvefit_notOG
import test_derivative as td
import shapiro3 as sp3
#changes to the directory of the system as defined by the user
def dir_system(system):
    #import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    from pathlib import Path

    #set the right directory
    os.chdir(r'D:\pythongraphs') 
    #calculation of Cinf
    if system[0:4]== 'pdms':
        CinfT = 5.7
    else: 
        CinfT = 4
    
    path = str(Path().absolute())
    newpath = path +  "/" + system 
    dataFolder = Path(newpath)
    os.chdir(dataFolder)
    return CinfT, dataFolder

def plot_density_volume(system,sysType, setn, setN, dv_bool, linear_reg, opt_reg, filter_value, hyper_fit, sqrt_fit, piece_wise, cluster, shapiro):
    #import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os

    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from scipy.stats import linregress
    from itertools import compress
    from sklearn.linear_model import LinearRegression
    import store_glass as sg
    
    #dir_system(system)
    print(str(Path().absolute()))
    #import data for the needed system. 
    #since we sometimes want to compare different systems,we need to load them within the function
    cr_list = ['10', '1', '05' , '01', '005', '001', '0001']
    cr_compressed = list(compress(cr_list, dv_bool))
    #print(cr_compressed)
    
    T_glass, d_glass = None, None

    #determine size of figure
    size_fig = [[1,1], [1,2], [1,3], [2,2], [2,3], [2,3]]
    fig, axes = plt.subplots(size_fig[(len(cr_compressed)-1)][0], size_fig[(len(cr_compressed)-1)][1], layout='tight',figsize=(20,10))
    axes  = axes.flatten()

    for n in range(0,len(cr_compressed)):
        ax = axes[n]
        

        try:
            energy = np.genfromtxt('energycr' + cr_compressed[n] + ".xvg",skip_header=27)
        except OSError:
            print(f"File not found, skipping.")
            continue

        if linear_reg:
            result = reg_scatter(110, 340, 360, 400, energy)
            T_glass = result[0]
            d_glass = result[1]
            links_res = result[2]
            rechts_res = result[3]
            ax.plot(energy[:,1], links_res.intercept + links_res.slope*energy[:,1], 'r', label='fitted line', linewidth=1, zorder = 5)
            ax.plot(energy[:,1], rechts_res.intercept + rechts_res.slope*energy[:,1], 'b', label='fitted line', linewidth=1, zorder = 6)
            ax.scatter(T_glass,d_glass, facecolors='none', edgecolors='black', s = 40, zorder = 10)
            ax.annotate('Tg = {:.0f} K'.format(T_glass), xy=(T_glass,d_glass) , xytext = (-5,40), textcoords='offset points',arrowprops={'arrowstyle': '-|>', 'color': 'blue'})
            ax.axvline(x = T_glass, color='blue',alpha = 0.5, linestyle='--')
            ax.set_xlim([75, 475])

            #make sure we are in the same directory
            curr_path = str(Path().absolute())      
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'lin_regress', T_glass)
            os.chdir(curr_path)

            textstr = r'T$_{g}^{lin}$ = ' + str(round(T_glass))
            props = dict(boxstyle='square', facecolor='green', alpha=0.6)
            ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)
            fig_name = 'dv_lin_reg'
            fit_mode = 'Linear regression'   

        if opt_reg:
            best_split, best_models, r2_combined = opt_scatter(energy)
            fit_mode = 'Optimized Linear Regression'

            
            #T_glass = (best_models[1].intercept - best_models[0].intercept)/ (best_models[0].slope - best_models[1].slope)
            T_glass = energy[best_split,1]
            d_glass = best_models[0].intercept + best_models[0].slope*T_glass
            
            print(energy[best_split,1])
            print(T_glass)
            #error in Tg
            slope1_err = best_models[0].stderr
            intercept1_err = best_models[0].intercept_stderr
            slope2_err = best_models[1].stderr
            intercept2_err = best_models[1].intercept_stderr

            delta_slope = best_models[0].slope - best_models[1].slope
            delta_intercept = best_models[1].intercept - best_models[0].intercept

            T_glass_opt_error = np.sqrt((1 / delta_slope) ** 2 * (intercept1_err ** 2 + intercept2_err ** 2) + (delta_intercept / delta_slope ** 2) ** 2 * (slope1_err ** 2 + slope2_err ** 2))
            #print(f"Energy shape: {energy.shape}")
            #print(f"Filter value: {filter_value}")

            if filter_value is None:
                T1 = np.linspace(T_glass, 450, 100)
                T2 = np.linspace(100,T_glass,100)
                ax.plot(T1, best_models[0].intercept + best_models[0].slope*T1, 'r', label='fitted line', linewidth=1, zorder = 5)
                ax.plot(T2, best_models[1].intercept + best_models[1].slope*T2, 'b', label='fitted line', linewidth=1, zorder = 6)

            #links_energy_filtered_ini = energy[energy[:,1] > links_Tlow]
            else:
                upper = T_glass + filter_value
                lower = T_glass - filter_value
                mask1 = energy[:best_split, 1] > upper
                mask2 = energy[best_split:, 1] < lower 

                #print(f"mask1: {mask1}")
                #print(f"mask2: {mask2}")
                #print(f"Sum of mask1: {np.sum(mask1)}")  # Should be > 0 if there are True values
                #print(f"Sum of mask2: {np.sum(mask2)}")

                if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                    energy_filtered1 = energy[:best_split][mask1]
                    energy_filtered2 = energy[best_split:][mask2]
                    #print(f"Energyfiltered1 shape: {energy_filtered1}")
                    #print(f"Energyfiltered2 shape: {energy_filtered2}")
                    best_models = linregress(energy_filtered1[:, 1], energy_filtered1[:, 4]), linregress(energy_filtered2[:, 1], energy_filtered2[:, 4])
                    ax.plot(energy_filtered1[:, 1], best_models[0].intercept + best_models[0].slope * energy_filtered1[:, 1], 'r', label='fitted line', linewidth=1, zorder=5)
                    ax.plot(energy_filtered2[:, 1], best_models[1].intercept + best_models[1].slope * energy_filtered2[:, 1], 'b', label='fitted line', linewidth=1, zorder=6)
                    T_glass = (best_models[1].intercept - best_models[0].intercept)/ (best_models[0].slope - best_models[1].slope)
                    d_glass = best_models[0].intercept + best_models[0].slope*T_glass
                else:
                    print("No data points for the filtered ranges.")

                #energy_filtered1 = energy[energy[:best_split,1] < (round(T_glass) - filter_value)]
                #energy_filtered2 = energy[energy[best_split:,1] > (round(T_glass) + filter_value)] 

                #best_models = linregress(energy_filtered1[:,1],energy_filtered1[:,4]), linregress(energy_filtered2[:,1],energy_filtered2[:,4])
                #ax.plot(T1, best_models[0].intercept + best_models[0].slope*T1, 'r', label='fitted line', linewidth=1, zorder = 5)
                #ax.plot(T2, best_models[1].intercept + best_models[1].slope*T2, 'b', label='fitted line', linewidth=1, zorder = 6)
            
            #make sure we are in the same directory
            curr_path = str(Path().absolute())
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'opt_linregress', T_glass)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'opt_linregress_err', T_glass_opt_error)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'opt_linregress_r2', r2_combined)
            os.chdir(curr_path)
            
            ax.set_xlim([75, 475])
            ax.axvline(x = T_glass, color='blue',alpha = 0.5, linestyle='--')
            ax.scatter(T_glass,d_glass, facecolors='none', edgecolors='black', s = 40, zorder = 10)
            ax.annotate('Tg = {:.0f} K'.format(T_glass), xy=(T_glass,d_glass) , xytext = (-5,40), textcoords='offset points',arrowprops={'arrowstyle': '-|>', 'color': 'blue'})
            textstr = r'T$_{g}^{opt, lin}$ =' + str(round(T_glass)) + 'K'
            props = dict(boxstyle='square', facecolor='green', alpha=0.6)
            ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)
            fit_mode = 'Opt fit'
            print('I am here '+ curr_path)
            fig_name = 'dv_opt_reg'  

        if hyper_fit:
            trialT, d_fit_hyp, d_fit_sqrt, d_fit_piece, T_glass_hyp, T_glass_sqrt, T_glass_piece, d_glass_hyp, d_glass_sqrt, d_glass_piece, T_glass_hyp_err, T_glass_sqrt_err, T_glass_piece_err, R2_hyp, R2_sqrt, R2_piece = opt_curvefit_notOG.curve_fit_op(energy,30,120)
            ax.plot(trialT, d_fit_hyp, color = 'blue')
            ax.annotate('Tg = {:.0f} K'.format(T_glass_hyp), xy=(T_glass_hyp,d_glass_hyp) , xytext = (-6,40), textcoords='offset points',arrowprops={'arrowstyle': '-|>', 'color': 'blue'})
            ax.scatter(T_glass_hyp,d_glass_hyp, facecolors='none', edgecolors='black', s = 40, zorder = 10)
            ax.axvline(x = T_glass_hyp, color='blue',alpha = 0.5, linestyle='--')
            textstr = r'T$_{g}^{hyp}$ = ' + str(round(T_glass_hyp)) + 'K'
            props = dict(boxstyle='square', facecolor='green', alpha=0.6)
            ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)
            ax.set_xlim([75, 475])

            curr_path = str(Path().absolute())
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'hyper_fit', T_glass_hyp)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'hyper_fit_err', T_glass_hyp_err)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'hyper_fit_r2', R2_hyp)
            os.chdir(curr_path)
            fit_mode = 'Hyper fit'
            fig_name = 'dv_hyp'

        if sqrt_fit:
            trialT, d_fit_hyp, d_fit_sqrt, d_fit_piece, T_glass_hyp, T_glass_sqrt, T_glass_piece, d_glass_hyp, d_glass_sqrt, d_glass_piece, T_glass_hyp_err, T_glass_sqrt_err, T_glass_piece_err, R2_hyp, R2_sqrt, R2_piece = opt_curvefit_notOG.curve_fit_op(energy,30,120)
            ax.plot(trialT, d_fit_sqrt, color='#39FF14')
            ax.annotate('Tg = {:.0f} K'.format(T_glass_sqrt), xy=(T_glass_sqrt,d_glass_sqrt) , xytext = (-6,40), textcoords='offset points',arrowprops={'arrowstyle': '-|>', 'color': 'blue'})
            ax.scatter(T_glass_sqrt,d_glass_sqrt, facecolors='none', edgecolors='black', s = 40, zorder = 10)
            ax.axvline(x = T_glass_sqrt, color='#39FF14',alpha = 0.5, linestyle='--')
            textstr = (r'T$_{g}^{sqrt}$ = ' + str(round(T_glass_sqrt)) + 'K')
            props = dict(boxstyle='square', facecolor='green', alpha=0.6)
            ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)

            curr_path = str(Path().absolute())
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'sqrt_fit', T_glass_sqrt)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'sqrt_fit_err', T_glass_sqrt_err)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'sqrt_fit_r2', R2_sqrt)
            os.chdir(curr_path)
            fit_mode = 'Sqrt fit'
            fig_name = 'dv_sqrt'  

        
        if piece_wise: 
            trialT, d_fit_hyp, d_fit_sqrt, d_fit_piece, T_glass_hyp, T_glass_sqrt, T_glass_piece, d_glass_hyp, d_glass_sqrt, d_glass_piece, T_glass_hyp_err, T_glass_sqrt_err, T_glass_piece_err, R2_hyp, R2_sqrt, R2_piece = opt_curvefit_notOG.curve_fit_op(energy,30,120)
            ax.plot(trialT, d_fit_piece, color = 'red')
            ax.annotate('Tg = {:.0f} K'.format(T_glass_piece), xy=(T_glass_piece,d_glass_piece) , xytext = (-6,40), textcoords='offset points',arrowprops={'arrowstyle': '-|>', 'color': 'blue'})
            ax.scatter(T_glass_piece,d_glass_piece, facecolors='none', edgecolors='blue', s = 40, zorder = 10)
            ax.axvline(x = T_glass_piece, color='red',alpha = 0.5, linestyle='--')
            textstr = (r'T$_{g}^{piece}$ = ' + str(round(T_glass_piece)) + 'K')
            props = dict(boxstyle='square', facecolor='green', alpha=0.6)
            ax.text(0.1, 0.3, textstr, transform=ax.transAxes, fontsize=20, verticalalignment='center', bbox=props)

            curr_path = str(Path().absolute())
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'piece_fit', T_glass_piece)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'piece_fit_err', T_glass_piece_err)
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'piece_fit_r2', R2_piece)
            os.chdir(curr_path)
            fit_mode = 'Piecewise'
            fig_name = 'dv_piece_wise'  

        if cluster:
            intersection_temp, intersection_density, slope1, slope2, intercept1, intercept2, model1, model2, region1_temp, region2_temp  =   td.identify_glass_transition(energy)
            ax.plot(region1_temp, model1.predict(region1_temp.reshape(-1, 1)), color='blue', label='Fit Region 1')
            ax.plot(region2_temp, model2.predict(region2_temp.reshape(-1, 1)), color='green', label='Fit Region 2')
            ax.axvline(x=intersection_temp, color='red', linestyle='--', label=f'Tg = {intersection_temp:.2f} K')
            ax.plot([region1_temp.min(), intersection_temp], [slope1 * region1_temp.min() + intercept1, intersection_density], 'b--')
            ax.plot([intersection_temp, region2_temp.max()], [intersection_density, slope2 * region2_temp.max() + intercept2], 'g--')
            fit_mode = 'Cluster'
            fig_name = 'dv_cluster'
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Density (kg/m^3)')
            ax.legend()

        if not shapiro:
            ax.scatter(energy[:,1],energy[:,4], s=1, c = 'dimgray', zorder = 0)
            ax.set_ylim(min(energy[:,4]) - 20,max(energy[:,4]) + 20 )
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Density (kg/m${^3}$)")
            ax.set_title(r'$\rho$ vs. T with coolingrate ' + cr_compressed[n] + r' K/ps and $\tau_p = 10$' '\n of '+ sysType +' with n=' + setn +' and N='+ setN)
            ax.set_xlim([75, 475])
            fig.suptitle(sysType +' with n=' + setn +' and N='+ setN + ' with ' + fit_mode, fontsize=16)
        
        if shapiro:
            fig_name = 'shapiro'
            T_glass_shapiro = sp3.shapiro(energy, sysType, setn, setN, label="Shapiro Tg", verbose=1, save = 'shapiro')
            curr_path = str(Path().absolute())
            sg.store_glass(sysType, setn, setN, cr_compressed[n], 'shapiro', T_glass_shapiro)
            os.chdir(curr_path)
    
    if not shapiro:
        plt.savefig(fig_name)
        plt.clf() 
        plt.close()   
    return 

def plot_cp_vs_T(system,sysType, setn, setN, cr_bool):
    #import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    import pandas as pd

    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from scipy.stats import linregress
    from itertools import compress
    from sklearn.linear_model import LinearRegression
    import store_glass as sg
    
    #dir_system(system)
    print(str(Path().absolute()))
    #import data for the needed system. 
    # since we sometimes want to compare different systems,we need to load them within the function
    cr_list = ['10', '1', '01', '001', '0001']
    cr_compressed = list(compress(cr_list, cr_bool))
    
    T_glass, d_glass = None, None

    #determine size of figure
    size_fig = [[1,1], [1,2], [1,3], [2,2], [2,3], [2,3]]
    fig, axes = plt.subplots(size_fig[(len(cr_compressed)-1)][0], size_fig[(len(cr_compressed)-1)][1], layout='tight',figsize=(20,10))
    axes  = axes.flatten()


    for n in range(0,len(cr_compressed)):
        ax = axes[n]
        energy = np.genfromtxt('alpha' + cr_compressed[n] + ".xvg",skip_header=26)
        df = pd.DataFrame(energy)
        df.columns = ['Time', 'Temperature', 'Volume', 'Enthalpy']     

        ax.scatter(df['Temperature'],df['Enthalpy'], s=1, c = 'dimgray', zorder = 0)
        #ax.set_ylim(min(energy[:,4]) - 20,max(energy[:,4]) + 20 )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"C$_{p}$")
        ax.set_title(r'$\rho$ vs. T with coolingrate ' + cr_compressed[n] + r' K/ps and $\tau_p = 10$' '\n of '+ sysType +' with n=' + setn +' and N='+ setN)
        #ax.set_xlim([75, 475])
        #fig.suptitle(sysType +' with n=' + setn +' and N='+ setN + ' with ', fontsize=16)

    #plt.savefig(fig_name)
    plt.show()
    plt.clf() 
    plt.close()   
    return 

#plot_cp_vs_T('pdms8100','pdms', '8', '100', [0,0,1,1,0])

def reg_scatter(links_Tlow, links_Thigh, rechts_Tlow, rechts_Thigh, energy):
    #import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os

    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from scipy.stats import linregress
    from itertools import compress

    links_energy_filtered_ini = energy[energy[:,1] > links_Tlow]
    links_energy_filtered_fin = links_energy_filtered_ini[links_energy_filtered_ini[:,1] < links_Thigh]

    rechts_energy_filtered_ini = energy[energy[:,1] > rechts_Tlow]
    rechts_energy_filtered_fin = rechts_energy_filtered_ini[rechts_energy_filtered_ini[:,1] < rechts_Thigh]

    rechts_res = linregress(rechts_energy_filtered_fin[:,1],rechts_energy_filtered_fin[:,4])
    links_res = linregress(links_energy_filtered_fin[:,1],links_energy_filtered_fin[:,4])

    T_glass = (rechts_res.intercept - links_res.intercept)/ (links_res.slope - rechts_res.slope)
    d_glass = rechts_res.intercept + rechts_res.slope*T_glass
    #print(T_glass)
    return T_glass, d_glass, links_res, rechts_res

def opt_scatter(data):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from scipy.stats import linregress

    data = data[data[:,1] < 450]
    

    min_total_residual = float('inf')
    best_split = None
    best_models = None

    for split_point in range(2,len(data[:,1])-2):
        #Split data in two segments
        T1, d1 = data[:split_point,1], data[:split_point,4]
        T2, d2 = data[split_point:,1], data[split_point:,4]

        #d1 = d1.reshape(-1, 1)
        #d2 = d2.reshape(-1, 1)

        lin1 = linregress(T1,d1)
        lin2 = linregress(T2,d2)
        lin1_pred = lin1.intercept + lin1.slope * T1
        lin2_pred = lin2.intercept + lin2.slope * T2
        #model1 = LinearRegression().fit(d1, T1)
        #model2 = LinearRegression().fit(d2, T2)

        residual1 = np.sum((lin1_pred - d1) ** 2)
        residual2 = np.sum((lin2_pred - d2) ** 2)

        total_residual = residual1 + residual2
        if total_residual < min_total_residual:
            min_total_residual = total_residual
            best_split = split_point
            best_models = (lin1, lin2)
            rss_segment1, rss_segment2 = residual1, residual2
    #print('intercept1 =' + str(lin1.intercept))
    #print('coef1 =' + str(lin1.slope))
    #print('intercept2 =' + str(lin2.intercept))
    #print('coef2 =' + str(lin2.slope))
    #print('T_glass_opt =' + str(T_glass_opt))
    #print('d_glass_opt =' + str(d_glass_opt))
    tss = np.sum((data[:,4] - np.mean(data[:,4]))**2)
    rss_bilinear = rss_segment1 + rss_segment2
    r2_combined = 1 - (rss_bilinear / tss)

    return best_split, best_models, r2_combined

#energy = np.genfromtxt("energy01.xvg",skip_header=27)
#opt_scatter(energy)
#plot_density_volume('pdms28200','pdms', '28', '200', [0,0,1,1,0], False, True, 20)