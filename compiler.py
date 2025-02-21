import glass_calculations1 as gc1
import glass_calculations as gc
import glass_calculations2 as gc2
import glass_calculations3 as gc3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
from PIL import Image
import store_glass
import equilibration as eq
import analysis_graphs as ag
import parameters as par

#creates pdf if needed
create_pdf = False

#Generate data
energy_min = False
nvt_equilibration = False
npt_equilibration = True

optimized_bi_fit = False
tanh_fit = False
squareroot_fit = False
piece_wise_fit = False
shapiro_fit = False

#Analyze data
table_path = 'table.txt'

option = 'opt_linregress'
option_err = 'opt_linregress_err'

pdes_vs_pdms = False
Tg_model_vs_sys = False
cp_vs_T = False
Tg_vs_n = False
Tg_model_vs_sys_cr = False
fox_flory_correct = False



#find all systems to iterate over
os.chdir(r'D:\pythongraphs\results')
folder = str(Path().absolute())
system_list = os.listdir(folder)

def sort_key(path):
    import re
    # Extracting the filename part
    filename = path.split('\\')[-1]
    # Using regex to separate the prefix and number
    match = re.match(r'([a-zA-Z]+)(\d+)', filename)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return (prefix, number)
    return (filename, 0)
sorted_system_list = sorted(system_list, key=sort_key)

image_list = []

for system in sorted_system_list:
    new_path = os.path.join(folder, system)
    os.chdir(new_path)
    sysType, sysN, sysn, syse = par.system_parameters(system)

    if energy_min:
        eq.em_eq(sysType,sysn, sysN)
        img_path = new_path + r'\em.png'
        image_list.append(img_path)
    
    if nvt_equilibration:
        eq.nvt_eq(system,sysType,sysn, sysN)
        img_path = new_path + r'\nvt.png'
        image_list.append(img_path)
    
    if npt_equilibration:
        eq.npt_eq(system,sysType,sysn, sysN)
        img_path = new_path + r'\npt.png'
        image_list.append(img_path)

    if optimized_bi_fit:
        gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = True, filter_value= None, hyper_fit = False, sqrt_fit = False, piece_wise = False, cluster = False, shapiro = False)
        img_path = new_path + r'\dv_opt_reg.png'
        image_list.append(img_path)
    
    if tanh_fit:
        gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = False,filter_value = None,hyper_fit = True, sqrt_fit = False, piece_wise = False,cluster = False, shapiro = False)
        img_path = new_path + r'\dv_hyp.png'
        image_list.append(img_path)

    if squareroot_fit:
        gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = False,filter_value = None,hyper_fit = False, sqrt_fit = True, piece_wise = False,cluster = False, shapiro = False)
        img_path = new_path + r'\dv_sqrt.png'
        image_list.append(img_path)

    if piece_wise_fit:
        gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = False,filter_value = None,hyper_fit = False, sqrt_fit = False, piece_wise = True,cluster = False, shapiro = False)
        img_path = new_path + r'\dv_piece_wise.png'
        image_list.append(img_path)

    if shapiro_fit:
        gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = False, filter_value= None, hyper_fit = False, sqrt_fit = False, piece_wise = False, cluster = False, shapiro = True)
        img_path = new_path + r'\shapiro.png'
        image_list.append(img_path)
    
    print(new_path)            
    #gc3.plot_density_volume(system,sysType,sysn, sysN, [0,0,0,0,0,1,0], True,True, True, True)
    #gc2.plot_density_volume(system,sysType,sysn, sysN, [0,0,0,1,0,1,1], True,False, False, False)
    #gc2.plot_density_volume(system,sysType,sysn, sysN, [0,0,0,1,0,1,1], False,False, True, False)
    #gc2.plot_density_volume(system,sysType,sysn, sysN, [0,0,0,1,0,1,1], False,False, False, True)
    #gc2.plot_density_volume(system,sysType,sysn, sysN, [0,0,0,1,0,1,1], False,True, False, False)
    
    #gc.plot_density_volume(system,sysType,sysn, sysN, [0,1,0,1,0,1,1], False, False,None,False, False, False,True)
    #img_path = new_path + r'\dv_cluster.png'
    #image_list.append(img_path)
    #gc1.plot_density_volume(system,sysType,sysn, sysN, [1,1,1,1,0], False, True,None,True, True, True)
    #img_path = new_path + r'\dv_combined_50.png'
    #image_list.append(img_path)
    #gc.plot_density_volume(system,sysType,sysn, sysN, dv_bool = [1,1,0,1,0,1,1], linear_reg = False, opt_reg = False, filter_value= None, hyper_fit = False, sqrt_fit = False, piece_wise = False, cluster = False, shapiro = True)
    
    #gc.plot_density_volume(system,sysType,sysn, sysN, [1,1,1,1,0], True, False,None,False, False, False)
    #img_path = new_path + r'\dv_lin_reg.png'
    #image_list.append(img_path)


os.chdir(r'D:\pythongraphs')
#new_path = str(os.path)
#img_path = new_path + r'\pdms_pdms_tg_diff'

#compile pdf file of the wanted images, such that it is easier to analyze
if create_pdf == True:
    pdf_images = []
    for image in image_list:
        img = Image.open(image)
        pdf_images.append(img)
    pdf_path = 'D:/pythongraphs/shapiro001.pdf'
    pdf_images[0].save(pdf_path,"PDF" ,resolution=100.0,save_all=True, append_images=pdf_images[1:])

#Analysis
if pdes_vs_pdms:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= True,Tg_model_vs_sys = False,Tg_model_vs_sys_test= False,cp_vs_T= False,Tg_vs_n= False,Tg_model_vs_sys_cr= False,Tg_model_vs_sys_left_right= False,fox_flory_correct= False)

if cp_vs_T:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= False,Tg_model_vs_sys = False,Tg_model_vs_sys_test= False,cp_vs_T= True,Tg_vs_n= False,Tg_model_vs_sys_cr= False,Tg_model_vs_sys_left_right= False,fox_flory_correct= False)

if Tg_vs_n:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= False,Tg_model_vs_sys = False,Tg_model_vs_sys_test= False,cp_vs_T= False,Tg_vs_n= True,Tg_model_vs_sys_cr= False,Tg_model_vs_sys_left_right= False,fox_flory_correct= False)

if Tg_model_vs_sys_cr:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= False,Tg_model_vs_sys = False,Tg_model_vs_sys_test= False,cp_vs_T= False,Tg_vs_n= False,Tg_model_vs_sys_cr= True,Tg_model_vs_sys_left_right= False,fox_flory_correct= False)

if Tg_model_vs_sys:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= False,Tg_model_vs_sys = True,Tg_model_vs_sys_test= False,cp_vs_T= False,Tg_vs_n= False,Tg_model_vs_sys_cr= False,Tg_model_vs_sys_left_right= False,fox_flory_correct= False)

if fox_flory_correct:
    ag.analysis(option,option_err,table_path,pdes_vs_pdms= False,Tg_model_vs_sys = False,Tg_model_vs_sys_test= False,cp_vs_T= False,Tg_vs_n= False,Tg_model_vs_sys_cr= False,Tg_model_vs_sys_left_right= False,fox_flory_correct= True)