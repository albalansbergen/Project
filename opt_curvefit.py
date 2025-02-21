import numpy as np
#energy = np.genfromtxt("energy01.xvg", skip_header=27)


#average data and put in bins

def average_fit(num,data):
    import numpy as np
    xdata = []
    ydata = []
    per_bin = (len(data[:,1])-27)/num
    for n in range(num):
        #filter for needed rows in data
        filtered_data = data[int(n*per_bin):int((n+1)*per_bin),:]
        median = np.median(filtered_data[:,1])
        xdata = np.append(xdata,median)
        y_averaged = np.mean(filtered_data[:, 4])
        ydata =np.append(ydata,y_averaged)
    return xdata, ydata

def curve_fit_op(energy,num_points,trial_points):
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    from scipy.optimize import curve_fit
    from scipy.optimize import least_squares
    import math
    from sklearn.cross_decomposition import PLSRegression
    from scipy.optimize import fsolve

    sorted_indices = np.argsort(energy[:,1])
    energy_sorted = energy[sorted_indices]

    T_noisy = energy_sorted[:,1]
    d_noisy = energy_sorted[:,4]
    #T_averaged, d_averaged = average_fit(num_points,energy_sorted)
    trialT = np.linspace(T_noisy[0], T_noisy[-1], trial_points)

    def block_averaging(data, block_size):
        num_blocks = len(data) // block_size
        block_means = [np.mean(data[i * block_size:(i + 1) * block_size]) for i in range(num_blocks)]
        block_std = np.std(block_means)
        return np.mean(block_means), block_std

    density_avg, density_std = block_averaging(d_noisy, block_size=100)

    #hyperbolic
    def funcHyp(xdata, w, M, G, Tg, c):
        return (w*((M-G)/2))*np.log(np.cosh((xdata-Tg)/2))+((xdata-Tg)*((M+G)/2))+c

    def funcSqrt(xdata,c, a, T,b,f):
        return c-(a*(xdata-T))-((b/2)*(xdata-T+np.sqrt(((xdata-T)**2)+(4*np.exp(f)))))
    
    def funcPiece(xdata,x1, x2, x3,y1, y2,y3):
        return np.piecewise(xdata, [(xdata < x3),(xdata >= x3)], [lambda xdata: ((y1*(x3-xdata))+ (y3*(xdata-x1)))/(x3-x1), lambda xdata: ((y3*(x2-xdata))+ (y2*(xdata-x3)))/(x2-x3)])

    #fit hyperbolic
    
    initial_guesses_hyp = [0.976064716, -0.795452921, -0.111192647, 303.696722, 1056.27978]
    initial_guesses_sqrt = [1062.97797, 0.246475025, 311.123764, 0.457563528, 6.70606196]
    initial_guesses_piece = [100, 450, 300, 1100, 1000, 900]

    lower_bound_hyp = [-np.inf, -np.inf, -np.inf, 200, -np.inf]
    upper_bound_hyp = [np.inf, np.inf, np.inf, 350, np.inf]

    lower_bound_sqrt = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    upper_bound_sqrt = [np.inf, np.inf, np.inf, np.inf, np.inf]

    lower_bound_piece = [0, 0, 0, -np.inf, -np.inf, -np.inf]
    upper_bound_piece = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]


    popt_hyp, pcov_hyp = curve_fit(funcHyp, T_noisy, d_noisy, p0=initial_guesses_hyp, maxfev=1000000, bounds = (lower_bound_hyp, upper_bound_hyp))
    popt_sqrt, pcov_sqrt = curve_fit(funcSqrt, T_noisy, d_noisy, p0=initial_guesses_sqrt, maxfev=1000000, bounds = (lower_bound_sqrt, upper_bound_sqrt))
    popt_piece, pcov_piece = curve_fit(funcPiece, T_noisy, d_noisy,p0=initial_guesses_piece, maxfev=1000000, bounds = (lower_bound_piece, upper_bound_piece))

    #popt_hyp_pls, pcov_hyp_pls = curve_fit(funcHyp, trialT, d_averaged, p0=initial_guesses_hyp, maxfev=1000000, bounds = (lower_bound_hyp, upper_bound_hyp))
    #popt_sqrt_pls, pcov_sqrt_pls = curve_fit(funcSqrt, trialT, d_averaged, p0=initial_guesses_sqrt, maxfev=1000000, bounds = (lower_bound_sqrt, upper_bound_sqrt))
    #popt_piece_pls, pcov_piece_pls = curve_fit(funcPiece, trialT, d_averaged,p0=initial_guesses_piece, maxfev=1000000, bounds = (lower_bound_piece, upper_bound_piece))



    T_glass_hyp_err = np.sqrt(np.diag(pcov_hyp))[3]
    T_glass_sqrt_err = np.sqrt(np.diag(pcov_sqrt))[2]
    T_glass_piece_err = np.sqrt(np.diag(pcov_piece))[2]
    #print(T_glass_hyp_err)
    #print(T_glass_sqrt_err)

    d_hyp = funcHyp(trialT,*popt_hyp)
    d_sqrt = funcSqrt(trialT,*popt_sqrt)
    d_piece = funcPiece(trialT,*popt_piece)

    #d_hyp_pls = pls.predict(trialT_reshaped)
    #d_sqrt_pls = funcSqrt(xdata_pls[:, 0],*popt_sqrt_pls)
    #d_piece_pls = funcPiece(xdata_pls[:, 0],*popt_piece_pls)

    def errfuncHyp(params, xdata, ydata):
        w, M, G, Tg, c = params
        return ydata - funcHyp(xdata, w, M, G, Tg, c)

    def errfuncSqrt(params, xdata, ydata):
        c, a, Tg,b,f = params
        return ydata - funcSqrt(xdata,c, a, Tg,b,f)
    
    def errfuncPiece(params, xdata, ydata):
        x1, x2, x3,y1, y2,y3 = params
        return ydata - funcPiece(xdata,x1, x2, x3,y1, y2, y3)
    
    def derivative_funcSqrt(xdata, c, a, T, b, f):
        return -a - (b / 2) * (1 + (xdata - T) / np.sqrt((xdata - T) ** 2 + 4 * np.exp(f)))

    result_hyp = least_squares(errfuncHyp, initial_guesses_hyp, args=(T_noisy, d_noisy))
    result_sqrt = least_squares(errfuncSqrt, initial_guesses_sqrt, args=(T_noisy, d_noisy))
    result_piece = least_squares(errfuncPiece, initial_guesses_piece, args=(T_noisy, d_noisy))
    
    # Error in T_g from least_squares residuals
    #J_hyp = result_hyp.jac
    #J_sqrt = result_sqrt.jac
    #cov_hyp = np.linalg.inv(J_hyp.T.dot(J_hyp))
    #cov_sqrt = np.linalg.inv(J_sqrt.T.dot(J_sqrt))

    #T_glass_hyp_err_jac = np.sqrt(np.diag(cov_hyp))[3]
    #T_glass_sqrt_err_jac = np.sqrt(np.diag(cov_sqrt))[2]

    #print(T_glass_hyp_err_jac)
    #print(T_glass_sqrt_err_jac)

    optimal_params_hyp = result_hyp.x
    optimal_params_sqrt = result_sqrt.x
    optimal_params_piece = result_piece.x
    #print(optimal_params_hyp)
    #print(optimal_params_sqrt)
    d_fit_hyp = funcHyp(trialT, *optimal_params_hyp)
    d_fit_sqrt = funcSqrt(trialT, *optimal_params_sqrt)
    d_fit_piece = funcPiece(trialT, *optimal_params_piece)

    T_glass_hyp = result_hyp.x[3]
    T_glass_sqrt = result_sqrt.x[2]
    T_glass_piece = result_piece.x[2]
    d_glass_hyp = funcHyp(result_hyp.x[3],result_hyp.x[0], result_hyp.x[1], result_hyp.x[2],result_hyp.x[3],result_hyp.x[4])
    d_glass_sqrt = funcSqrt(result_sqrt.x[2],result_sqrt.x[0], result_sqrt.x[1], result_sqrt.x[2],result_sqrt.x[3],result_sqrt.x[4])
    d_glass_piece = funcPiece(result_piece.x[2],result_piece.x[0], result_piece.x[1], result_piece.x[2],result_piece.x[3],result_piece.x[4],result_piece.x[5])


    c, a, T_0, b, f = optimal_params_sqrt  

    T_low = 100
    T_mid = 450

    #slope_low = -a
    slope_low = derivative_funcSqrt(T_low, c, a, T_0, b, f)
    rho_at_100K = funcSqrt(T_low, c, a, T_0, b, f)  
    intercept_low = rho_at_100K - slope_low * T_low 
    low_temp_tangent = slope_low * (T_noisy - T_low) + rho_at_100K

    #slope_mid = -(a+b)
    slope_mid = derivative_funcSqrt(T_mid, c, a, T_0, b, f)
    rho_at_450K = funcSqrt(T_mid, c, a, T_0, b, f)  
    intercept_mid = rho_at_450K - slope_mid * T_mid  
    mid_temp_tangent = slope_mid * (T_noisy - T_mid) + rho_at_450K

    #intersection of the two tangents (Tg)
    def find_intersection(Tg, T_low, slope_low, T_mid, slope_mid, rho_at_100K, rho_at_450K):
        return (slope_low * (Tg - T_low) + rho_at_100K) - (slope_mid * (Tg - T_mid) + rho_at_450K)

    #solve for Tg
    Tg_initial_guess = 300  # Initial guess for the glass transition temperature
    T_g_solution_sqrt = fsolve(find_intersection, Tg_initial_guess, args=(T_low, slope_low, T_mid, slope_mid, rho_at_100K, rho_at_450K))

    TSS = np.sum((d_noisy - np.mean(d_noisy)) ** 2)

    #RSS (Residual Sum of Squares)
    RSS_hyp = np.sum((d_noisy - funcHyp(T_noisy, *popt_hyp)) ** 2)
    RSS_sqrt = np.sum((d_noisy - funcSqrt(T_noisy, *popt_sqrt)) ** 2)
    RSS_piece = np.sum((d_noisy - funcPiece(T_noisy, *popt_piece)) ** 2)

    #R2 for each fit
    R2_hyp = 1 - (RSS_hyp / TSS)
    R2_sqrt = 1 - (RSS_sqrt / TSS)
    R2_piece = 1 - (RSS_piece / TSS)

    #plt.scatter(T_averaged, d_averaged)
    #plt.plot(trialT, d_fit_hyp, 'b-', label="pls fit")
    plt.plot(trialT, d_fit_sqrt, 'g-', label="pls fit")
    plt.plot(T_noisy,mid_temp_tangent)
    plt.plot(T_noisy,low_temp_tangent)
    #plt.plot(trialT, d_fit_piece, 'y-', label="pls fit")
    #plt.plot(trialT, d_fit_sqrt, 'r-',ls='--', label="Sqrt Fit")
    plt.show()
    return trialT, d_fit_hyp, d_fit_sqrt, d_fit_piece, T_glass_hyp, T_g_solution_sqrt[0], T_glass_piece, d_glass_hyp, d_glass_sqrt, d_glass_piece, T_glass_hyp_err, T_glass_sqrt_err, T_glass_piece_err, R2_hyp, R2_sqrt, R2_piece

#curve_fit_op(energy,60,60)