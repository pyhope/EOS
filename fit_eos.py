import numpy as np
import eos
import matplotlib.pyplot as plt


def fit_eos():
    """
    Fit the equation of state (EOS) using the BM4_TH model and plot the EOS.

    Returns:
        None
    """

    # Load data from a file named 'brg.dat'. The data is unpacked into separate variables.
    data = np.loadtxt('brg.dat', unpack=True)

    # Assign temperature data to T_data, pressure data to P_data (converted from Pa to GPa),
    # and volume data to V_data from the loaded data.
    T_data = data[0]
    P_data = data[1] / 1e4  # Conversion factor to change pressure units to GPa
    V_data = data[2]

    # Initial guess values for the volume at zero pressure (V0) and temperature (T0).
    V0, T0 = 3200, 3000

    # Fit the EOS using the BM4_TH model with the temperature, volume, and pressure data,
    # along with initial guesses for V0 and T0. Returns optimized parameters and their covariance.
    popt, pcov = eos.fit_BM4_TH(T_data, V_data, P_data, V0, T0)

    # Prepare data for fitting by pairing temperature and volume data into a list of tuples.
    x = []
    for i in range(len(T_data)):
        x.append((T_data[i], V_data[i]))

    # Calculate the fitted pressure values using the BM4_TH model with the optimized parameters.
    P_fit = [eos.BM4_TH(i, *popt, V0, T0) for i in x]

    # Print the sum of squared differences between the actual pressure data and the fitted pressure data,
    # which represents the fitting error.
    print('Fitting Error =', np.sum((P_data - P_fit)**2))

    # Plotting section starts here.

    # Extract the fitted parameters for plotting and diagnostics.
    K0, Kp, Kdp, a, b, c = popt
    print('K0 =', K0, 'Kp =', Kp, 'Kdp =', Kdp, 'a =', a, 'b =', b, 'c =', c)

    # Create a 3D plot with a specific figure size.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a range of temperature and volume values for plotting the fitted EOS surface.
    T_range = np.linspace(0, 4500, 1000)
    V_range = np.linspace(min(V_data) - 100, max(V_data) + 100, 1000)
    T, V = np.meshgrid(T_range, V_range)

    # Calculate the fitted pressure values over the generated range of temperature and volume.
    P_fit = eos.BM4_TH((T, V), K0, Kp, Kdp, a, b, c, V0, T0)

    # Plot the original data points as red scatter points.
    ax.scatter(T_data, V_data, P_data, c='red', label='Original data points')

    # Plot the fitted EOS surface with a colormap and partial transparency.
    ax.plot_surface(T, V, P_fit, cmap='viridis', alpha=0.5)

    # Set labels for the axes and a legend for the plot.
    ax.set_xlabel('T (K)')
    ax.set_ylabel('V (Å$^3$)')
    ax.set_zlabel('P (GPa)')
    ax.legend(fancybox=False, edgecolor='black')

    # Display the plot.
    plt.show()


# Call the fit_eos function to fit the EOS using the BM4_TH model and plot the results.
fit_eos()
