def curve_fit_op(energy, num_points, trial_points, block_size=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import t
    import math

    sorted_indices = np.argsort(energy[:, 1])
    energy_sorted = energy[sorted_indices]

    T_noisy = energy_sorted[:, 1]
    d_noisy = energy_sorted[:, 4]
    trialT = np.linspace(T_noisy[0], T_noisy[-1], trial_points)

    def funcHyp(xdata, w, M, G, Tg, c):
        return (w * ((M - G) / 2)) * np.log(np.cosh((xdata - Tg) / 2)) + \
               ((xdata - Tg) * ((M + G) / 2)) + c

    def funcSqrt(xdata, c, a, T, b, f):
        return c - (a * (xdata - T)) - \
               (b / 2) * (xdata - T + np.sqrt((xdata - T) ** 2 + 4 * np.exp(f)))

    def funcPiece(xdata, x1, x2, x3, y1, y2, y3):
        return np.piecewise(
            xdata,
            [xdata < x3, xdata >= x3],
            [
                lambda xdata: ((y1 * (x3 - xdata)) + (y3 * (xdata - x1))) / (x3 - x1),
                lambda xdata: ((y3 * (x2 - xdata)) + (y2 * (xdata - x3))) / (x2 - x3),
            ],
        )

    def block_averaging(x, y, block_size):
        num_blocks = len(x) // block_size
        Tg_list = []
        for i in range(num_blocks):

            x_block = x[i * block_size:(i + 1) * block_size]
            y_block = y[i * block_size:(i + 1) * block_size]


            try:
                popt_block, _ = curve_fit(
                    funcHyp, x_block, y_block, p0=initial_guesses_hyp, maxfev=1000000,
                    bounds=(lower_bound_hyp, upper_bound_hyp)
                )
                Tg_block = popt_block[3]
                Tg_list.append(Tg_block)
            except:
                continue  

        Tg_mean = np.mean(Tg_list)
        Tg_std = np.std(Tg_list, ddof=1)  # Sample standard deviation
        Tg_err = Tg_std / np.sqrt(len(Tg_list))  # Standard error

        return Tg_err

    initial_guesses_hyp = [0.976, -0.795, -0.111, 300.0, 1056.27]
    initial_guesses_sqrt = [1062.97, 0.246, 300.0, 0.457, 6.70]
    initial_guesses_piece = [100, 450, 300, 1100, 1000, 900]

    lower_bound_hyp = [-np.inf, -np.inf, -np.inf, 200, -np.inf]
    upper_bound_hyp = [np.inf, np.inf, np.inf, 400, np.inf]

    lower_bound_sqrt = [-np.inf, -np.inf, 200, -np.inf, -np.inf]
    upper_bound_sqrt = [np.inf, np.inf, 400, np.inf, np.inf]

    lower_bound_piece = [0, 0, 200, -np.inf, -np.inf, -np.inf]
    upper_bound_piece = [np.inf, np.inf, 400, np.inf, np.inf, np.inf]

    popt_hyp, pcov_hyp = curve_fit(
        funcHyp, T_noisy, d_noisy, p0=initial_guesses_hyp,
        maxfev=1000000, bounds=(lower_bound_hyp, upper_bound_hyp)
    )
    popt_sqrt, pcov_sqrt = curve_fit(
        funcSqrt, T_noisy, d_noisy, p0=initial_guesses_sqrt,
        maxfev=1000000, bounds=(lower_bound_sqrt, upper_bound_sqrt)
    )
    popt_piece, pcov_piece = curve_fit(
        funcPiece, T_noisy, d_noisy, p0=initial_guesses_piece,
        maxfev=1000000, bounds=(lower_bound_piece, upper_bound_piece)
    )

    d_fit_hyp = funcHyp(T_noisy, *popt_hyp)
    d_fit_sqrt = funcSqrt(T_noisy, *popt_sqrt)
    d_fit_piece = funcPiece(T_noisy, *popt_piece)

    T_glass_hyp = popt_hyp[3]
    T_glass_sqrt = popt_sqrt[2]
    T_glass_piece = popt_piece[2]

    d_glass_hyp = funcHyp(T_glass_hyp, *popt_hyp)
    d_glass_sqrt = funcSqrt(T_glass_sqrt, *popt_sqrt)
    d_glass_piece = funcPiece(T_glass_piece, *popt_piece)

    T_glass_hyp_err = block_averaging(T_noisy, d_noisy, block_size)
    T_glass_sqrt_err = T_glass_hyp_err  
    T_glass_piece_err = T_glass_hyp_err  

    residuals_hyp = d_noisy - d_fit_hyp
    residuals_sqrt = d_noisy - d_fit_sqrt
    residuals_piece = d_noisy - d_fit_piece

    chi_square_hyp = np.sum(residuals_hyp ** 2)
    chi_square_sqrt = np.sum(residuals_sqrt ** 2)
    chi_square_piece = np.sum(residuals_piece ** 2)

    degrees_of_freedom = len(T_noisy) - len(popt_hyp)
    reduced_chi_square_hyp = chi_square_hyp / degrees_of_freedom
    reduced_chi_square_sqrt = chi_square_sqrt / degrees_of_freedom
    reduced_chi_square_piece = chi_square_piece / degrees_of_freedom

    TSS = np.sum((d_noisy - np.mean(d_noisy)) ** 2)

    RSS_hyp = np.sum((d_noisy - d_fit_hyp) ** 2)
    RSS_sqrt = np.sum((d_noisy - d_fit_sqrt) ** 2)
    RSS_piece = np.sum((d_noisy - d_fit_piece) ** 2)

    R2_hyp = 1 - (RSS_hyp / TSS)
    R2_sqrt = 1 - (RSS_sqrt / TSS)
    R2_piece = 1 - (RSS_piece / TSS)

    print(f"Chi-Square (Hyperbolic): {chi_square_hyp:.2f}, Reduced Chi-Square: {reduced_chi_square_hyp:.2f}")
    print(f"Chi-Square (Square Root): {chi_square_sqrt:.2f}, Reduced Chi-Square: {reduced_chi_square_sqrt:.2f}")
    print(f"Chi-Square (Piecewise): {chi_square_piece:.2f}, Reduced Chi-Square: {reduced_chi_square_piece:.2f}")

    print(f"R² (Hyperbolic): {R2_hyp:.4f}")
    print(f"R² (Square Root): {R2_sqrt:.4f}")
    print(f"R² (Piecewise): {R2_piece:.4f}")

    return (
        T_noisy, d_fit_hyp, d_fit_sqrt, d_fit_piece,
        T_glass_hyp, T_glass_sqrt, T_glass_piece,
        d_glass_hyp, d_glass_sqrt, d_glass_piece,
        T_glass_hyp_err, T_glass_sqrt_err, T_glass_piece_err,
        R2_hyp, R2_sqrt, R2_piece
    )
