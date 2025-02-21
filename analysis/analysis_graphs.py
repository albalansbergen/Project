#analysis(system_list, systype, sysn, sysN)
def analysis(option,option_err,table_path,pdes_vs_pdms,Tg_model_vs_sys,Tg_model_vs_sys_test,cp_vs_T,Tg_vs_n,Tg_model_vs_sys_cr,Tg_model_vs_sys_left_right,fox_flory_correct):
    import pandas as pd
    import matplotlib.pyplot as plt
    import parameters as par
    import matplotlib.cm as cm
    import numpy as np
    from scipy.optimize import curve_fit

    #table_path = 'table_pdms_real_final.txt'
    list_headers = ['type', 'n', 'N', 'cr', 'density', 'lin_regress','lin_regress_err', 'lin_regress_r2', 'opt_linregress', 'opt_linregress_err', 'opt_lin_regress_r2', 'hyper_fit', 'hyper_fit_err','hyper_fit_r2', 'sqrt_fit', 'sqrt_fit_err','sqrt_fit_r2', 'piece_fit', 'piece_fit_err','piece_fit_r2', 'shapiro']    
    data = pd.read_csv(table_path, names=list_headers, header=0)
    sysn_list = data['n'].unique()
    sysN_list = data['N'].unique()
    sysCR_list = [0.1, 0.01, 0.001]
    systype_list = data['type'].unique()
    
    #pdes_vs_pdms = False
    #Tg_model_vs_sys = False
    #Tg_model_vs_sys_test = False
    #cp_vs_T = False
    #Tg_vs_n = False
    #fox_flory = False
    #Tg_model_vs_sys_cr = False  
    #Tg_model_vs_sys_left_right = False
    #fox_flory_correct = True
    
    cr_to_color = {
        10: '#FFA500',
        1: '#39FF14',
        0.1: '#FF1493',
        0.5: '#FF45CC', # Neon pink hex code
        0.01: '#B026FF',
        0.05: '#0000FF',
        0.001: '#00008B'  # Dark blue hex code
    }

    if pdes_vs_pdms:
        fig, ax = plt.subplots(figsize=(12, 6))

        for nloop in range(len(sysn_list)):
            for Nloop in range(len(sysN_list)):
                data_pdms = data[(data['type'] == 'PDMS') & (data['n'] == sysn_list[nloop]) & (data['N'] == sysN_list[Nloop])]
                data_pdes = data[(data['type'] == 'PDES') & (data['n'] == sysn_list[nloop]) & (data['N'] == sysN_list[Nloop])]

                data_pdms = data_pdms[data_pdms['cr'].isin(sysCR_list)]
                data_pdes = data_pdes[data_pdes['cr'].isin(sysCR_list)]

                if not data_pdms.empty and not data_pdes.empty:
                    data_pdms.reset_index(drop=True, inplace=True)
                    data_pdes.reset_index(drop=True, inplace=True)

                    valid_cr_mask = data_pdms['cr'].isin(data_pdes['cr'])
                    data_pdms = data_pdms[valid_cr_mask].reset_index(drop=True)
                    data_pdes = data_pdes[valid_cr_mask].reset_index(drop=True)

                    if not data_pdms.empty and not data_pdes.empty:
                        error_threshold = 1000
                        abs_pdes_pdms = data_pdes[option] - data_pdms[option]
                        combined_error = np.sqrt(data_pdes[option_err] ** 2 + data_pdms[option_err] ** 2)
                        mask = combined_error < error_threshold

                        data_pdms = data_pdms[mask].reset_index(drop=True)
                        data_pdes = data_pdes[mask].reset_index(drop=True)
                        abs_pdes_pdms = abs_pdes_pdms[mask].reset_index(drop=True)
                        combined_error = combined_error[mask].reset_index(drop=True)

                        if not data_pdms.empty and not data_pdes.empty:
                            df = pd.DataFrame({
                                'cr': data_pdms['cr'],
                                'abs_difference': abs_pdes_pdms,
                                'combined_error': combined_error
                            }).dropna(subset=['cr', 'abs_difference', 'combined_error'])

                            df['cr'] = df['cr'].astype(float)
                            df['x_labels'] = ['(PDES-PDMS)\n' + f'$_{{N={round(sysN_list[Nloop])},n={round(sysn_list[nloop])}}}$' for _ in df['cr']]

                            for index in df.index:
                                cr_value = df.at[index, 'cr']
                                if pd.isna(cr_value):
                                    continue  
                                ax.errorbar(
                                    df.at[index, 'x_labels'], 
                                    df.at[index, 'abs_difference'], 
                                    yerr=df.at[index, 'combined_error'],
                                    fmt='o',
                                    label=f'$\\dot{{\\gamma}} = {cr_value:.3f}$ K/ps' if f'$\\dot{{\\gamma}} = {cr_value:.3f}$ K/ps' not in plt.gca().get_legend_handles_labels()[1] else "",
                                    color=cr_to_color[cr_value], 
                                    capsize=4,
                                    markersize=7,
                                    marker='o', 
                                    linestyle='None'
                                )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1.2)

        ax.set_ylabel(r"$\Delta T_g$ (K)$_{\text{(PDES - PDMS)}}$", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(loc='upper right', fontsize=10, title=r'$\dot{\gamma}$ (K/ps)', title_fontsize='13')

        plt.grid(True, which='both', axis='y', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        fig_name = 'correct_pdms_vs_pdes_tg_diff_opt_error_bars.png'
        plt.savefig(fig_name, dpi=300)
        plt.clf() 
        plt.close()

    if Tg_model_vs_sys:
        fit_to_color = {
            'sqrt_fit': '#B026FF',   # Purple
            'opt_linregress': '#FF1493',  # Pink
            'hyper_fit': '#00008B',  # Dark blue
            'piece_fit': '#39FF14',  # Neon green
            'shapiro': '#21F8F6' #Neon light blue
        }
        N_p = {8: 3, 18: 8, 28: 13, 38: 18}

        fit_models = ['opt_linregress', 'hyper_fit', 'piece_fit', 'sqrt_fit', 'shapiro']
        fit_models_label = ['Optimized bilinear fit', 'Integrated tanh function', 'Piecewise linear function', 'Hyperbolic fit', 'Shapiro fit']

        for crloop in range(len(sysCR_list)):
            fig, ax = plt.subplots(figsize=(14, 7)) 

            xlabel_list = []
            n_list = []
            N_list = []

            y_min, y_max = float('inf'), float('-inf')

            for nloop in range(len(sysn_list)):
                for Nloop in range(len(sysN_list)):
                    for typeloop in range(len(systype_list)):
                        data_filtered = data[(data['type'] == systype_list[typeloop]) & (data['cr'] == sysCR_list[crloop]) & (data['n'] == sysn_list[nloop]) & (data['N'] == sysN_list[Nloop])]
                        
                        if not data_filtered.empty:
                            x_label = f'${{N_p={round(N_p[sysn_list[nloop]])},N_c={round(sysN_list[Nloop])}}}$'
                            xlabel_list = np.append(xlabel_list, x_label)
                            n_list = np.append(n_list, sysn_list[nloop])
                            N_list = np.append(N_list, sysN_list[Nloop])
                            fitn = 0
                            #data_filtered['polymerization'] = data_filtered['n'].map(N_p)

                            for fit in fit_models:
                                Tg_values = data_filtered[fit]
                                Tg_errors = data_filtered[fit + '_err'] if fit + '_err' in data_filtered.columns else None
                                
                                if Tg_errors is not None:
                                    error_threshold = 400
                                    valid_mask = Tg_errors < error_threshold
                                    Tg_values = Tg_values[valid_mask]
                                    Tg_errors = Tg_errors[valid_mask]

                                if not Tg_values.isnull().all():
                                    if Tg_errors is not None and not Tg_errors.isnull().all():
                                        ax.errorbar(
                                            [x_label for _ in Tg_values],
                                            Tg_values,
                                            yerr=Tg_errors,
                                            fmt='o',
                                            label=f'{fit_models_label[fitn]}' if f'{fit_models_label[fitn]}' not in plt.gca().get_legend_handles_labels()[1] else "",
                                            color=fit_to_color[fit],
                                            capsize=4,
                                            markersize=5,  
                                            elinewidth=1.5, 
                                            capthick=1.5  
                                        )
                                    else:
                                        ax.scatter(
                                            [x_label for _ in Tg_values],
                                            Tg_values,
                                            label=f'{fit_models_label[fitn]}' if f'{fit_models_label[fitn]}' not in plt.gca().get_legend_handles_labels()[1] else "",
                                            color=fit_to_color[fit],
                                            s=30,  
                                            edgecolor='#21F8F6', 
                                            linewidth=0.5
                                        )
                                    fitn += 1
                                    
                                    y_min = min(y_min, Tg_values.min())
                                    y_max = max(y_max, Tg_values.max())

            ax.grid(True, which='both', linestyle=':', linewidth=0.7, color='lightgray')
            ax.set_ylabel(r"${T_g}$" + ' (K)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)

            #shading
            numeric_positions = np.arange(len(xlabel_list))
            for x in range(0, len(numeric_positions) - 1, 2):
                if x < len(numeric_positions):
                    ax.axvspan(numeric_positions[x] - 0.5, numeric_positions[x + 1] - 0.5, facecolor='grey', alpha=0.15, zorder=0)
                
            if len(numeric_positions) % 2 == 1: 
                ax.axvspan(numeric_positions[-1] - 0.5, numeric_positions[-1] + 0.5, facecolor='grey', alpha=0.15, zorder=0)

            y_buffer = 10
            ax.set_ylim([max(200, y_min - y_buffer), min(400, y_max + y_buffer)])

            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.legend(loc='upper left', fontsize=9)

            fig.suptitle(f'Glass Transition Temperature ($T_g$) for {systype_list[typeloop]} with $\\dot{{\\gamma}} = {sysCR_list[crloop]}$ K/ps', fontsize=16, y=0.95)

            fig.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.9)
            fig_name = r"figures/model_vs_sys/" + f'Tg_per_fit_model_cr{sysCR_list[crloop]}_with_errors'
            plt.savefig(fig_name + '.png', dpi=300)
            plt.show()
            plt.clf()
            plt.close()

    if Tg_vs_n:
        mw_pdms = {8: 681.437, 18: 1426.937, 28: 2172.437, 38: 2917.937}
        mw_pdes = {8: 874.788, 18: 1896.904, 28: 2919.020, 38: 3941.136}
        colors = {
            0.01: '#B026FF',   # purple
            0.1: '#FF1493',    # pink
            0.001: '#00008B',  # dark blue
        }



        for systype in ['PDMS', 'PDES']:
            fig, ax = plt.subplots(figsize=(7, 6))

            data_filtered = data[(data['type'] == systype) & (data['N'] != 50)]

            for crloop in sysCR_list:
                data_cr_filtered = data_filtered[data_filtered['cr'] == crloop]

                if option_err is None:
                    grouped_result = data_cr_filtered.groupby('n').agg(
                        mean_value_Tg=(option, 'mean')
                    ).reset_index()

                if option_err is not None:
                    grouped_result = data_cr_filtered.groupby('n').agg(
                        mean_value_Tg=(option, 'mean'), 
                        combined_error=(option_err, lambda x: np.sqrt((x ** 2).sum()) / len(x))
                    ).reset_index()

                if systype == 'PDMS':
                    grouped_result['molecular_weight'] = grouped_result['n'].map(mw_pdms)
                elif systype == 'PDES':
                    grouped_result['molecular_weight'] = grouped_result['n'].map(mw_pdes)

                if option_err is not None and 'combined_error' in grouped_result.columns:
                    ax.errorbar(
                        grouped_result['molecular_weight'],
                        grouped_result['mean_value_Tg'], 
                        yerr=grouped_result['combined_error'],
                        fmt='o', 
                        color=colors[crloop],
                        ecolor=colors[crloop],
                        capsize=5, 
                        capthick=2, 
                        elinewidth=1.5,
                        markersize=8,
                        marker='o',  
                        linestyle='None',  
                        label=f'$\\dot{{\\gamma}} = {crloop}$ K/ps'  
                    )
                else:
                    ax.scatter(
                    grouped_result['molecular_weight'],  
                    grouped_result['mean_value_Tg'], 
                    color=colors[crloop],
                    s=64,  
                    label=f'$\\dot{{\\gamma}} = {crloop}$ K/ps',
                    edgecolor='black',
                    linewidth=0.5
                )
            ax.text(0.764, 0.33, systype,  
        horizontalalignment='right', 
        verticalalignment='bottom', 
        transform=ax.transAxes, 
        fontsize=24,  
        color='#4B4B4B',  
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='#808080', boxstyle='square,pad=0.3'))  
 

            ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')
            ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray') 

            ax.set_xlabel(r"$M_w$ (g/mol)", fontsize=12) 
            ax.set_ylabel(r"T$_g$ (K)", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

            
            ax.legend(loc='upper left', fontsize=10)

           
            plt.tight_layout()
            fig_name = r"figures/Tg_vs_n/" + f'Tg_vs_molecular_weight_{systype}_' + option
            plt.savefig(fig_name + '.png', dpi=300)
            plt.show()
            plt.clf()
            plt.close()
    
    if Tg_model_vs_sys_cr:
        cr_to_color = {
            1.0: '#39FF14',
            0.1: '#B026FF',    # Pink
            0.01: '#FF1493',   # Neon green
            0.001: '#00008B',  # Dark blue
        }

        
        fit_models_label = {
            'sqrt_fit': 'Square-root fit',
            'opt_linregress': 'Optimized bilinear fit',
            'hyper_fit': 'Integrated tanh function',
            'piece_fit': 'Piecewise linear function',
            'shapiro': 'Shapiro'
        }

        fit_models = ['opt_linregress', 'hyper_fit', 'piece_fit', 'sqrt_fit', 'shapiro']
        N_p = {8: 3, 18: 8, 28: 13, 38: 18}

        for fit in fit_models:
            fig, ax = plt.subplots(figsize=(14, 7))
            legend_handles = []  
            xlabel_list = []

            for cr_value in sysCR_list:
                data_filtered = data[(data['cr'] == cr_value)]
                
                if not data_filtered.empty:
                    for nloop in range(len(sysn_list)):
                        for Nloop in range(len(sysN_list)):
                            data_system = data_filtered[(data_filtered['n'] == sysn_list[nloop]) & (data_filtered['N'] == sysN_list[Nloop])]
                            
                            if not data_system.empty:
                                x_label = f'${{N_p={round(N_p[sysn_list[nloop]])},N_c={round(sysN_list[Nloop])}}}$'
                                xlabel_list = np.append(xlabel_list, x_label)
                                Tg_values = data_system[fit]
                                if f'{fit}_err' in data_system.columns:
                                    Tg_errors = data_system[f'{fit}_err']
                                else:
                                    Tg_errors = None

                                if not Tg_values.isnull().all():
                                    if cr_value not in legend_handles:
                                        label = rf'$\dot{{\gamma}}$ = {cr_value} K/ps'
                                        legend_handles.append(cr_value)
                                    else:
                                        label = None  
                                    if Tg_errors is not None:
                                        ax.errorbar(
                                            [x_label] * len(Tg_values),
                                            Tg_values,
                                            yerr=Tg_errors,
                                            fmt='o',
                                            color=cr_to_color[cr_value],  
                                            label=label,
                                            capsize=4,
                                            markersize=5,
                                            elinewidth=1.5,
                                            capthick=1.5
                                        )
                                    else:
                                        ax.scatter(
                                            [x_label] * len(Tg_values),
                                            Tg_values,
                                            color=cr_to_color[cr_value],
                                            label=label,
                                            s=30,  
                                            edgecolor='black',
                                            linewidth=0.5
                                        )

            ax.set_xlim(left=-1, right=(len(sysn_list) * len(sysN_list))-2.5)

            numeric_positions = np.arange(len(xlabel_list))
            for x in range(0, len(numeric_positions) - 1, 2):
                if x < len(numeric_positions):
                    ax.axvspan(numeric_positions[x] - 0.5, numeric_positions[x + 1] - 0.5, facecolor='grey', alpha=0.15, zorder=0)
                
            if len(numeric_positions) % 2 == 1: 
                ax.axvspan(numeric_positions[-1] - 0.5, numeric_positions[-1] + 0.5, facecolor='grey', alpha=0.15, zorder=0)

            y_min, y_max = data[fit].min() - 10, data[fit].max() + 10
            plt.ylim(240, 375)
            plt.grid(True, which='both', linestyle=':', linewidth=0.7)

            ax.set_ylabel(r"${T_g}$" + ' (K)', fontsize=14, fontweight='bold')
            #ax.set_xlabel('Systems', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=12)

            plt.legend(loc='upper left', fontsize=10)

            fig_name = r"figures/Tg_model_system_cr/" + f'{fit_models_label[fit]}_Tg_vs_Systems_fixed.png'
            plt.tight_layout()
            plt.savefig(fig_name, dpi=300)
            plt.show()
            plt.clf()
            plt.close()

    if fox_flory_correct:
        def fox_flory(M_w, Tg_inf, K):
            return Tg_inf - K / M_w

        mw_pdms = {8: 681.437, 18: 1426.937, 28: 2172.437, 38: 2917.937}
        mw_pdes = {8: 874.788, 18: 1896.904, 28: 2919.020, 38: 3941.136}

        colors = {
            0.01: '#B026FF',   # purple
            0.1: '#FF1493',    # pink
            0.001: '#00008B',  # dark blue
        }

        for systype in ['PDMS', 'PDES']:
            fig, ax = plt.subplots(figsize=(7, 6))

            data_filtered = data[(data['type'] == systype) & (data['N'] != 50)]

            for crloop in sysCR_list:
                data_cr_filtered = data_filtered[data_filtered['cr'] == crloop] 

                if option_err is None:
                    grouped_result = data_cr_filtered.groupby('n').agg(
                        mean_value_Tg=(option, 'mean')
                    ).reset_index()

                if option_err is not None:
                    grouped_result = data_cr_filtered.groupby('n').agg(
                        mean_value_Tg=(option, 'mean'), 
                        combined_error=(option_err, lambda x: np.sqrt((x ** 2).sum()) / len(x))
                    ).reset_index()

                if systype == 'PDMS':
                    grouped_result['molecular_weight'] = grouped_result['n'].map(mw_pdms)
                elif systype == 'PDES':
                    grouped_result['molecular_weight'] = grouped_result['n'].map(mw_pdes)

                try:
                    popt, pcov = curve_fit(
                        fox_flory, 
                        grouped_result['molecular_weight'], 
                        grouped_result['mean_value_Tg'], 
                        p0=[grouped_result['mean_value_Tg'].max(), 1000],  
                        bounds=(0, np.inf)  
                    )
                    Tg_inf, K = popt
                    fit_label = f"Fox-Flory Fit: $T_g^\\infty={Tg_inf:.2f}$ K, $K={K:.2f}$"
                except RuntimeError:
                    Tg_inf, K = None, None
                    fit_label = "Fit Failed"

                #plt Fox-Flory fit
                if Tg_inf is not None:
                    mw_fit = np.linspace(grouped_result['molecular_weight'].min(), grouped_result['molecular_weight'].max(), 100)
                    Tg_fit = fox_flory(mw_fit, *popt)
                    ax.plot(mw_fit, Tg_fit, color=colors[crloop], linestyle='--', label=fit_label)

                if option_err is not None and 'combined_error' in grouped_result.columns:
                    ax.errorbar(
                        grouped_result['molecular_weight'],  
                        grouped_result['mean_value_Tg'], 
                        yerr=grouped_result['combined_error'],
                        fmt='o', 
                        color=colors[crloop],  
                        ecolor=colors[crloop],  
                        capsize=5, 
                        capthick=2, 
                        elinewidth=1.5,
                        markersize=8,  
                        marker='o',  
                        linestyle='None',  
                        label=f'$\\dot{{\\gamma}} = {crloop}$ K/ps'  
                    )
                else:
                    ax.scatter(
                    grouped_result['molecular_weight'],  
                    grouped_result['mean_value_Tg'], 
                    color=colors[crloop],
                    s=64,  # Adjust marker size
                    label=f'$\\dot{{\\gamma}} = {crloop}$ K/ps',
                    linewidth=0.5
                )

            ax.text(0.764, 0.33, systype, 
                    horizontalalignment='right', 
                    verticalalignment='bottom', 
                    transform=ax.transAxes, 
                    fontsize=24, 
                    color='#4B4B4B', 
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor='#808080', boxstyle='square,pad=0.3'))

            ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')  
            ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray') 

            ax.set_xlabel(r"$M_w$ (g/mol)", fontsize=12) 
            ax.set_ylabel(r"T$_g$ (K)", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

            ax.legend(
                loc='upper center',  # Place it at the top center
                bbox_to_anchor=(0.5, -0.15),  # Move it outside the plot, centered horizontally
                fontsize=10,  # Font size for the legend text
                ncol=2  # Arrange the legend items in two columns
            )

            plt.tight_layout()
            fig_name = r"figures/fox_flory/" + f'Tg_vs_molecular_weight_with_fox_flory_{systype}_' + option
            plt.savefig(fig_name + '.png', dpi=300)
            plt.show()
            plt.clf()
            plt.close()
    return 

#analysis('piece_fit','piece_fit_err')
