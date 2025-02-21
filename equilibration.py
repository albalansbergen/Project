def em_eq():
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))  # Adjusting the size to be similar to previous plots
    # 18: 1426.937, 28: 2172.437, 38: 2917.937
    # Plot data for each system with its unique color and label for N
    #em = np.genfromtxt("em18100.xvg", skip_header=24)
    #ax.plot(em[:, 0], em[:, 1]/(1426.937*100), color='#39FF14', linestyle='-', linewidth=1.0, marker='o', markersize=3, label=r"N${_m}$ = 18")

    em = np.genfromtxt("em28100.xvg", skip_header=24)
    ax.plot(em[:, 0], em[:, 1]/(2172.437*100), color='#00008B', linestyle='-', linewidth=1.0, marker='o', markersize=3, label=r"N${_p}$ = 13")

    em = np.genfromtxt("em38100.xvg", skip_header=24)
    ax.plot(em[:, 0], em[:, 1]/(2917.937*100), color='#FF1493', linestyle='-', linewidth=1.0, marker='o', markersize=3, label=r"N${_p}$ = 18")

    #em = np.genfromtxt("em28100.xvg", skip_header=24)
    #ax.plot(em[:, 0], em[:, 1]/(2172.437*100), color='#FF1493', linestyle='-', linewidth=1.0, marker='o', markersize=3, label="N = 100")

    #em = np.genfromtxt("em28150.xvg", skip_header=24)
    #ax.plot(em[:, 0], em[:, 1]/(2172.437*150), color='#B026FF', linestyle='-', linewidth=1.0, marker='o', markersize=3, label="N = 150")

    #em = np.genfromtxt("em28200.xvg", skip_header=24)
    #ax.plot(em[:, 0], em[:, 1]/(2172.437*200), color='#00008B', linestyle='-', linewidth=1.0, marker='o', markersize=3, label="N = 200")

    ax.set_xlabel("Energy Minimization Step", fontsize=12)
    ax.set_ylabel(r'V$_{tot}$ ' + "(kJ/mol/" + r'(N$_p{\cdot N_c}$))', fontsize=12)
    ax.set_ylim([0.1, 0.4])
    ax.set_xlim([0, 20000])

    # Style adjustments: gridlines
    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray')

    ax.tick_params(axis='both', which='major', labelsize=12)

    #ax.legend(fontsize=10, loc="best", title="Chainsize" + r" (N${_m}$)")

    plt.tight_layout()
    plt.savefig("em.png", dpi=300)  # Save the styled plot with the legend
    plt.clf()
    plt.close()
    return

def nvt_eq(system,sysType, setn, setN):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os

    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from scipy.stats import linregress
    print(str(Path().absolute()))
    #import nvt 
    energy = np.genfromtxt("energynvt.xvg", skip_header=25)
    gyration = np.genfromtxt("gyratenvt.xvg", skip_header=27)
    polystat = np.genfromtxt("polystatnvt.xvg",skip_header=28)

    lsisi = 0.155 #length of silicon bond
    #calculation of Cinf
    if system[0:4]== 'pdms':
        CinfT = 5.7
    else: 
        CinfT = 4

    #initialize plots
    plt.clf()
    fig, ax = plt.subplots(3,2, figsize = (40,20))

    def moving_average(win_size,col,data):
        i = 0
        moving_average = []
        time = []
        while i < len(data[:,1]) - 3*(win_size/2) + 1:
            start = int(i-1+(win_size/2))
            end = int(i-1+(3*win_size/2))
            window = data[start:end,col]
            timeave = np.average(data[i:(i+win_size),0])
            windowaverage = sum(window)/win_size
            time.append(timeave)
            moving_average.append(windowaverage)
            i += 1
        timetrans = np.transpose(time)
        maverage=np.transpose(moving_average)
        return(time, maverage)

    #temperature plot
    timeT, movingaverageT = moving_average(100,1,energy)
    fig.suptitle('NVT calibration of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[0,0].plot(timeT,movingaverageT,zorder = 5)
    ax[0,0].plot(energy[:,0],energy[:,1], zorder = 0)
    ax[0,0].set_xlabel("Time (ps)")
    ax[0,0].set_ylabel("T (K)")
    ax[0,0].set_title('Temperature equilibration of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[0,0].grid()

    #timeG, movingaverageG = moving_average(100,1,gyration)
    #ax[0,1].plot(timeG,movingaverageG,zorder = 5)
    ax[0,1].plot(polystat[:,0],pow(polystat[:,2],2), zorder = 0)
    ax[0,1].set_xlabel("Time (ps)")
    ax[0,1].set_ylabel("${<R_g>^2}$ (nm$^2$)")
    ax[0,1].set_title('<R$_g>^2$ at T = 450K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[0,1].grid()

    #timeRe, movingaverageRe = moving_average(100,1,polystat)
    #ax[1,1].plot(timeRe,movingaverageRe,zorder = 5)
    ax[1,1].plot(polystat[:,0],pow(polystat[:,1],2), zorder = 0)
    ax[1,1].axhline(y=(CinfT*int(setn)*1.55**2)/100, color='r', linestyle='-')
    ax[1,1].set_xlabel("Time (ps)")
    ax[1,1].set_ylabel("${<R_e>^2}$ (nm$^2$)")
    ax[1,1].set_title('End-to-end distance at T = 450K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[1,1].grid()


    Cinf = pow(polystat[:,1],2)/(int(setn)*pow(lsisi,2))
    ax[1,0].axhline(y=CinfT, color='r', linestyle='-')
    ax[1,0].plot(polystat[:,0],Cinf, zorder = 0)
    ax[1,0].set_xlabel("Time (ps)")
    ax[1,0].set_ylabel("${C_∞}$ (-)")
    ax[1,0].set_title('${C_∞}$ for T = 450 K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[1,0].grid()

    checkpoly = pow(polystat[:,1],2)/pow(polystat[:,2],2)
    ax[2,0].axhline(y=6, color='r', linestyle='-')
    ax[2,0].plot(polystat[:,0],checkpoly, zorder = 0)
    ax[2,0].set_xlabel("Time (ps)")
    ax[2,0].set_ylabel("$<R_e>^2/<R_g>^2$")
    ax[2,0].set_title('$<R_e>^2/<R_g>^2$ for T = 450 K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[2,0].grid()

    plt.savefig("nvt.png")
    plt.clf() 
    plt.close() 
    return

def npt_eq(system,sysType, setn, setN):
    #import the right libraries and read the necessary files
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    import store_glass as sg

    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from scipy.stats import linregress

    print(os.getcwd())
    print(os.listdir())
    #import npt
    energy = np.genfromtxt("presdens.xvg", skip_header=27)
    polystat = np.genfromtxt("polystatnpt.xvg",skip_header=28)

    #store the density average
    density_window = len(energy[:,0])/5
    density_start = int((4 * density_window) - 1)
    density_end = int(len(energy[:,0]) - 1)
    density_slice = energy[density_start:density_end, 4]
    N = len(density_slice)
    std_density = np.std(density_slice, ddof=1)
    density_average = np.mean(density_slice)
    std_error_density = std_density / np.sqrt(N)

    #store the density average
    Cinf_window = len(energy[:,0])/10
    #Cinf_start = int((4 * density_window) - 1)
    #Cinf_end = int(len(energy[:,0]) - 1)
    #Cinf = pow(polystat[:,1],2)/(int(setn)*pow(0.155,2))
    #Cinf_slice = Cinf[Cinf_start:Cinf_end, 1]
    #N = len(Cinf_slice)
    #std_density = np.std(density_slice, ddof=1)
    #Cinf_average = np.mean(Cinf_slice)
    #std_error_density = std_density / np.sqrt(N)

    curr_path = str(Path().absolute())
    #sg.store_glass(sysType, setn, setN, None, 'Cinf', Cinf_average)
    sg.store_glass(sysType, setn, setN, None, 'density', density_average)
    os.chdir(curr_path)

    lsisi = 0.155 
    #calculation of Cinf
    if system[0:4]== 'pdms':
        CinfT = 5.7
    else: 
        CinfT = 4

    #initialize plots
    plt.clf()
    fig, ax = plt.subplots(3,2, figsize = (40,20))

    def moving_average(win_size,col,data):
        i = 0
        moving_average = []
        time = []
        while i < len(data[:,1]) - 3*(win_size/2) + 1:
            start = int(i-1+(win_size/2))
            end = int(i-1+(3*win_size/2))
            window = data[start:end,col]
            timeave = np.average(data[i:(i+win_size),0])
            windowaverage = sum(window)/win_size
            time.append(timeave)
            moving_average.append(windowaverage)
            i += 1
        timetrans = np.transpose(time)
        maverage=np.transpose(moving_average)
        return(time, maverage)

    #pressure
    timeP, movingaverageP = moving_average(1000,1,energy)
    fig.suptitle('NPT calibration')
    ax[0,0].plot(timeP,movingaverageP,zorder = 5)
    ax[0,0].plot(energy[:,0],energy[:,1], zorder = 0)
    ax[0,0].set_xlabel("Time (ps)")
    ax[0,0].set_ylabel("P (atm)")
    ax[0,0].set_title('Pressure calibration at T = 450K  '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[0,0].grid()

    timeD, movingaverageD = moving_average(1000,2,energy)
    ax[1,0].plot(timeD,movingaverageD,zorder = 5)
    ax[1,0].plot(energy[:,0],energy[:,4], zorder = 0)
    ax[1,0].set_xlabel("Time (ps)")
    ax[1,0].set_ylabel("Density (kg/m${^3}$)")
    ax[1,0].set_title('Density at T = 450K  '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[1,0].grid()

    #timeG, movingaverageG = moving_average(1000,2,polystat)
    #ax[0,1].plot(timeG,movingaverageG,zorder = 5)
    ax[0,1].plot(polystat[:,0],pow(polystat[:,2],2), zorder = 0)
    ax[0,1].set_xlabel("Time (ps)")
    ax[0,1].set_ylabel("${<R_g>^2}$ (nm)")
    ax[0,1].set_title('Radius of gyration $^2$ at T = 450K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[0,1].grid()

    #timeRe, movingaverageRe = moving_average(1000,1,polystat)
    #ax[1,1].plot(timeRe,movingaverageRe,zorder = 5)
    ax[1,1].plot(polystat[:,0],pow(polystat[:,1],2), zorder = 0)
    ax[1,1].axhline(y=(CinfT*int(setn)*1.55**2)/100, color='r', linestyle='-')
    ax[1,1].set_xlabel("Time (ps)")
    ax[1,1].set_ylabel("${<R_e>^2}$ (nm)")
    ax[1,1].set_title('End-to-end distance $^2$ at T = 450K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[1,1].grid()

    lsisi = (0.155)
    Cinf = pow(polystat[:,1],2)/(int(setn)*pow(lsisi,2))
    ax[2,0].axhline(y=CinfT, color='r', linestyle='-')
    ax[2,0].plot(polystat[:,0],Cinf, zorder = 0)
    ax[2,0].set_xlabel("Time (ps)")
    ax[2,0].set_ylabel("${C_∞}$ (nm)")
    ax[2,0].set_title('${C_∞}$ for T = 450 K of '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[2,0].grid()

    checkpoly = pow(polystat[:,1],2)/pow(polystat[:,2],2)
    ax[2,1].axhline(y=6, color='r', linestyle='-')
    ax[2,1].plot(polystat[:,0],checkpoly, zorder = 0)
    ax[2,1].set_xlabel("Time (ps)")
    ax[2,1].set_ylabel("$<R_e>^2/<R_g>^2$")
    ax[2,1].set_title('$<R_e>^2/<R_g>^2$ for T = 450 K of  '+ sysType +' with n=' + setn +' and N='+ setN)
    ax[2,1].grid()

    plt.savefig("npt.png")
    plt.clf() 
    plt.close() 
    return

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def styled_nvt_eq(system, sysType, setn, setN):
    energy = np.genfromtxt("energynvt.xvg", skip_header=25)
    polystat = np.genfromtxt("polystatnvt.xvg", skip_header=28)

    lsisi = 0.164 
    if system[0:4] == 'pdms':
        CinfT = 5.7
    else:
        CinfT = 4.0
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))  # 3 plots side by side

    ax[0,0].plot(energy[:,0],energy[:,1], zorder = 0, color = '#B026FF')
    ax[0,0].set_xlabel("Time (ps)", fontsize=12)
    ax[0,0].set_ylabel("T (K)", fontsize = 12)
    ax[0,0].set_title('Temperature during NVT equilibration',fontsize=14, fontweight='bold')
    ax[0,0].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0,0].tick_params(axis='both', which='major', labelsize=10)

    ax[0,1].plot(polystat[:, 0], np.power(polystat[:, 2], 2), color='#B026FF', linestyle='-', linewidth=1.0)
    ax[0,1].set_xlabel("Time (ps)", fontsize=12)
    ax[0,1].set_ylabel(r"${<R_g^2>}$ (nm$^2$)", fontsize=12)
    ax[0,1].set_title(r'$<R_g^2>$ during NVT equilibration', fontsize=14, fontweight='bold')
    ax[0,1].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0,1].tick_params(axis='both', which='major', labelsize=10)

    ax[1,1].plot(polystat[:, 0], np.power(polystat[:, 1], 2), color='#FF1493', linestyle='-', linewidth=1.5)
    ax[1,1].axhline(y=(CinfT * int(setn) * 1.64**2) / 100, color='r', linestyle='--', linewidth=1.5)
    ax[1,1].set_xlabel("Time (ps)", fontsize=12)
    ax[1,1].set_ylabel(r"${<R_e^2>}$ (nm$^2$)", fontsize=12)
    ax[1,1].set_title(r'$<R_e^2>$ during NVT equilibration', fontsize=14, fontweight='bold')
    ax[1,1].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[1,1].tick_params(axis='both', which='major', labelsize=10)

    Cinf = np.power(polystat[:, 1], 2) / (int(setn) * np.power(lsisi, 2))
    ax[0, 2].axhline(y=CinfT, color='r', linestyle='--', linewidth=1.5)
    #ax[0, 2].axhline(y=7.6, color='r', linestyle='--', linewidth=1.5)
    ax[0, 2].plot(polystat[:, 0], Cinf, color='#FF1493', linestyle='-', linewidth=1.0)
    ax[0, 2].set_xlabel("Time (ps)", fontsize=12)
    ax[0, 2].set_ylabel("${C_∞}$ (-)", fontsize=12)
    ax[0, 2].set_title(r'C$_\infty$ during NVT equilibration', fontsize=14, fontweight='bold')
    ax[0, 2].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0, 2].tick_params(axis='both', which='major', labelsize=10)

    checkpoly = np.power(polystat[:, 1], 2) / np.power(polystat[:, 2], 2)
    ax[1,2].axhline(y=6, color='r', linestyle='--', linewidth=1.5)
    ax[1,2].plot(polystat[:, 0], checkpoly, color='#00008B', linestyle='-', linewidth=1.0)
    ax[1,2].set_xlabel("Time (ps)", fontsize=12)
    ax[1,2].set_ylabel(r"$<R_e^2>/<R_g^2>$", fontsize=12)
    ax[1,2].set_title(r'$<R_e^2>/<R_g^2>$ during NVT equilibration', fontsize=14, fontweight='bold')
    ax[1,2].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[1,2].tick_params(axis='both', which='major', labelsize=10)

    #fig.suptitle(f'NVT Calibration of {sysType} with n={setn} and N={setN}', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    plt.savefig("nvt_styled.png", dpi=300)
    plt.show()
    plt.clf()
    plt.close()

    return

def styled_npt_eq(system, sysType, setn, setN):
    energy = np.genfromtxt("presdens.xvg", skip_header=27)
    polystat = np.genfromtxt("polystatnpt.xvg", skip_header=28)

    lsisi = 0.164 
    if system[0:4] == 'pdms':
        CinfT = 5.7
    else:
        CinfT = 4.0
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))  # 6 plots: 3x2 grid

    ax[0, 0].plot(energy[:, 0], energy[:, 2], color='#B026FF', linestyle='-', linewidth=1.0)
    ax[0, 0].set_xlabel("Time (ps)", fontsize=12)
    ax[0, 0].set_ylabel("P (bar)", fontsize=12)
    ax[0, 0].set_title(r'Pressure at T = 450K', fontsize=14, fontweight='bold')
    ax[0, 0].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0, 0].tick_params(axis='both', which='major', labelsize=10)

    ax[1, 0].plot(energy[:, 0], energy[:, 4], color='#FF1493', linestyle='-', linewidth=1.0)
    ax[1, 0].set_xlabel("Time (ps)", fontsize=12)
    ax[1, 0].set_ylabel(r"Density (kg/m${^3}$)", fontsize=12)
    ax[1, 0].set_title(r'Density at T = 450K', fontsize=14, fontweight='bold')
    ax[1, 0].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[1, 0].tick_params(axis='both', which='major', labelsize=10)

    ax[0, 1].plot(polystat[:, 0], np.power(polystat[:, 2], 2), color='#00008B', linestyle='-', linewidth=1.5)
    ax[0, 1].set_xlabel("Time (ps)", fontsize=12)
    ax[0, 1].set_ylabel(r"${<R_g^2>}$ (nm$^2$)", fontsize=12)
    ax[0, 1].set_title(r'Radius of gyration $^2$ at T = 450K', fontsize=14, fontweight='bold')
    ax[0, 1].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0, 1].tick_params(axis='both', which='major', labelsize=10)

    ax[1, 1].plot(polystat[:, 0], np.power(polystat[:, 1], 2), color='#B026FF', linestyle='-', linewidth=1.0)
    #ax[1, 1].axhline(y=(CinfT * int(setn) * lsisi**2) / 100, color='r', linestyle='--', linewidth=1.5)
    ax[1, 1].set_xlabel("Time (ps)", fontsize=12)
    ax[1, 1].set_ylabel(r"${<R_e^2>}$ (nm$^2$)", fontsize=12)
    ax[1, 1].set_title(r'End-to-end distance $^2$ at T = 450K', fontsize=14, fontweight='bold')
    ax[1, 1].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[1, 1].tick_params(axis='both', which='major', labelsize=10)

    Cinf = np.power(polystat[:, 1], 2) / (int(setn) * np.power(lsisi, 2))
    ax[0, 2].axhline(y=CinfT, color='r', linestyle='--', linewidth=1.5)
    ax[0, 2].plot(polystat[:, 0], Cinf, color='#FF1493', linestyle='-', linewidth=1.0)
    ax[0, 2].set_xlabel("Time (ps)", fontsize=12)
    ax[0, 2].set_ylabel("${C_∞}$ (-)", fontsize=12)
    ax[0, 2].set_title(r'C$_\infty$ at T = 450K', fontsize=14, fontweight='bold')
    ax[0, 2].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[0, 2].tick_params(axis='both', which='major', labelsize=10)

    checkpoly = np.power(polystat[:, 1], 2) / np.power(polystat[:, 2], 2)
    ax[1, 2].axhline(y=6, color='r', linestyle='--', linewidth=1.5)
    ax[1, 2].plot(polystat[:, 0], checkpoly, color='#00008B', linestyle='-', linewidth=1.0)
    ax[1, 2].set_xlabel("Time (ps)", fontsize=12)
    ax[1, 2].set_ylabel(r"$<R_e^2>/<R_g^2>$", fontsize=12)
    ax[1, 2].set_title(r'$<R_e^2>/<R_g^2>$ at T = 450K', fontsize=14, fontweight='bold')
    ax[1, 2].grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax[1, 2].tick_params(axis='both', which='major', labelsize=10)

    #fig.suptitle(f'NPT Calibration of {sysType} with n={setn} and N={setN}', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    plt.savefig("npt_styled.png", dpi=300)
    plt.show()
    plt.clf()
    plt.close()




import os
os.chdir(r'C:\Users\albal\Downloads')
styled_nvt_eq("pdms18100", "PDMS", "18", "100")
#styled_npt_eq("pdms28100", "PDMS", "28", "100")
#styled_npt_eq("pdes28200", "PDES", "38", "200")
#em_eq()
