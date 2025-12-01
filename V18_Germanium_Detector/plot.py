import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import json
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem
from scipy.signal import find_peaks


import numpy as np

def read_spe(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Stelle finden, wo $DATA steht
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            start = i + 2   # Nach $DATA kommt: "first last", danach die Zähler
            break

    counts = []

    for line in lines[start:]:
        line = line.strip()

        # echte Count-Zeilen bestehen nur aus ganzen Zahlen
        if line.isdigit():
            counts.append(int(line))

    return np.array(counts)


data_Eu = read_spe('data/Eu.Spe')
data_background = read_spe('data/Background.Spe')
data_Ba = read_spe('data/Ba.Spe')
data_Cs = read_spe('data/Cs.Spe')
data_unknown = read_spe('data/Unknown_source.Spe')

channels = np.arange(len(data_Eu))

livetime_background = 75671
livetime_Eu = 6266
livetime_Ba = 4455
livetime_Cs = 6382
livetime_unknown = 5111

background_normed = data_background / livetime_background
Eu_normed = data_Eu / livetime_Eu
Ba_normed = data_Ba / livetime_Ba
Cs_normed = data_Cs / livetime_Cs
unknown_normed = data_unknown / livetime_unknown

Eu_bgsub = Eu_normed - background_normed
Ba_bgsub = Ba_normed - background_normed
Cs_bgsub = Cs_normed - background_normed
unknown_bgsub = unknown_normed - background_normed

#Plot Eu Spectrum and Background
plt.figure(figsize=(10,6))
plt.plot(channels, Eu_normed, color='blue', zorder=1)
plt.fill_between(channels, np.zeros_like(Eu_normed), Eu_normed, color='blue', alpha=1.0, label = r'$^{152}$Eu')
plt.plot(channels, background_normed, color='orange', zorder=3)
plt.fill_between(channels, np.zeros_like(background_normed), background_normed, color='orange', alpha=1.0, zorder=2, label='Background')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid()
#plt.savefig('plots/Eu_spectrum.pdf')
plt.clf()

#Peaks of Eu
Peaks_low = find_peaks(Eu_bgsub[:3250], height=0.025, distance=10)
Peaks_high = find_peaks(Eu_bgsub[3250:7000], height = 0.003, distance=50)
Peaks = []
for n in Peaks_low[0]:
    Peaks.append(n)
for n in Peaks_high[0]:
    Peaks.append(n+3250)


plt.plot(channels, Eu_bgsub, color='blue', zorder=1)
plt.fill_between(channels, np.zeros_like(Eu_bgsub), Eu_bgsub, color='blue', alpha=1.0, label = r'$^{152}$Eu')
plt.plot(Peaks, Eu_bgsub[Peaks], "x", color='red', label='Identified Peaks', zorder=2)
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid(True)
#plt.savefig('plots/Eu_peaks.pdf')
plt.clf()

#Peaks
def Gauss(x, A, x0, sigma, B):
    return A  * np.exp(-0.5*(x-x0)**2/sigma**2) + B

print(Peaks)

channel_peaks = []
channel_peaks_err = []
peak_area = []
peak_area_err = []
for i in range(len(Peaks)):
    x_peak = channels[Peaks[i]-30:Peaks[i]+30]
    y_peak = Eu_bgsub[Peaks[i]-30:Peaks[i]+30]
    # peak_area = np.sum(y_peak)
    popt, pcov = curve_fit(Gauss, x_peak, y_peak, p0=[max(y_peak), channels[Peaks[i]], 5, min(y_peak)])
    perr = np.sqrt(np.diag(pcov))
    a_fit, x0_fit, sigma_fit, b_fit = popt

    peak_area_fit = a_fit * sigma_fit * np.sqrt(2 * np.pi)
    peak_area_fit_err = peak_area_fit * np.sqrt((perr[0]/a_fit)**2 + (perr[2]/sigma_fit)**2)
    print(f"Peak {i+1}: Channel = {x0_fit:.2f} ± {perr[1]:.2f}, Area = {peak_area_fit:.2f} ± {peak_area_fit_err:.2f}")
    peak_area_t = peak_area_fit*livetime_Eu
    print(f"          Total Counts in Peak: {peak_area_t:.2f} ± {peak_area_fit_err*livetime_Eu:.2f}")
    channel_peaks.append(x0_fit)
    channel_peaks_err.append(perr[1])
    peak_area.append(peak_area_fit)
    peak_area_err.append(peak_area_fit_err)



    # peak_area_sum = np.sum(y_peak[17:-17])
    # print(f"Peak {i+1}: Channel = {x0_fit:.2f} ± {perr[1]:.2f}, Area (sum) = {peak_area_sum:.2f}")

#Gauss Fits Peak1 and 7
# x_peak = channels[Peaks[6]-30:Peaks[6]+30]
# y_peak = Eu_bgsub[Peaks[6]-30:Peaks[6]+30]
# peak_area = np.sum(y_peak)
# popt, pcov = curve_fit(Gauss, x_peak, y_peak, p0=[max(y_peak), channels[Peaks[6]], 5, min(y_peak)])
# perr = np.sqrt(np.diag(pcov))
# a_fit, x0_fit, sigma_fit, b_fit = popt
# x_fit = np.linspace(channels[Peaks[6]-30], channels[Peaks[6]+30], 100)
# y_fit = Gauss(x_fit, popt[0], popt[1], popt[2], popt[3])

# plt.figure(figsize=(8,5))
# plt.plot(channels[Peaks[6]-30:Peaks[6]+30], Eu_bgsub[Peaks[6]-30:Peaks[6]+30], 'x', color='blue', zorder=1, label='Peak 7')
# plt.plot(x_fit, y_fit, label='Fit')
# plt.xlabel('Channel')
# plt.ylabel(r'Counts per second   [1/s]')
# plt.legend()
# plt.grid()
# plt.savefig('plots/Eu_peak7.pdf')
# plt.clf()




#Linear Regression for Energy Callibration
E_lit = np.array([121.78, 244.70, 344.28, 411.12, 443.97, 778.90])  # in keV

def linear(x, m, b):
    return m * x + b

popt, pcov = curve_fit(linear, channel_peaks, E_lit)
perr = np.sqrt(np.diag(pcov))
m_fit, b_fit = popt
print(f"Energy Calibration: E = {m_fit:.4f} ± {perr[0]:.4f} * channel + {b_fit:.4f} ± {perr[1]:.4f}")
#Plot Energy Calibration
#plt.figure()
#plt.plot(channel_peaks, E_lit, 'o', label='Measured Peaks', color='red')
plt.errorbar(channel_peaks, E_lit, xerr=channel_peaks_err, fmt='o', label='Measured Peaks')
x_fit = np.linspace(0, 8000, 100)
y_fit = linear(x_fit, *popt)
plt.plot(x_fit, y_fit, label='Linear Fit')
plt.xlabel('Channel')
plt.ylabel(r'Energy $E$ [keV]')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('plots/Energy_calibration.pdf')
plt.clf()
#print(perr)
E_m_fit = m_fit
E_b_fit = b_fit

#Vollenergienachweiswahrscheinlichkeit Eu
#Aktivität
activity_0 = 4130
delta_activity_0 = 60
decay_time = 25*365.25*24*3600+40*24*3600  # in seconds
halbwertszeit_Eu = 13*365.25*24*3600+196*24*3600  # in seconds
activity_Eu = activity_0 * np.exp(-np.log(2) * decay_time / halbwertszeit_Eu)
activity_Eu_err = delta_activity_0 * np.exp(-np.log(2) * decay_time / halbwertszeit_Eu)
print(f"Aktuelle Aktivität von Eu-152: {activity_Eu:.2f} ± {activity_Eu_err:.2f} Bq")

#Raumwinkelterm
a = 9.5e-2  # in m
r = 2.25e-2  # in m
omega = 0.5*(1-a/np.sqrt(a**2+r**2))

#Emissionswahrascheinlichkeiten
P_Eu = np.array([0.2841, 0.0755, 0.2660, 0.0224, 0.028, 0.1297])
P_Eu_err = np.array([0.0013, 0.0004, 0.0012, 0.0010, 0.0002, 0.0006])



W = []
delta_W = []
for i in range(len(peak_area)):
    Q = peak_area[i] / (activity_Eu * P_Eu[i] * omega)
    Q_err = np.sqrt( (peak_area_err[i]/(activity_Eu*P_Eu[i]*omega))**2 + (peak_area[i]*activity_Eu_err/(activity_Eu**2 *P_Eu[i]*omega))**2 + (peak_area[i]*P_Eu_err[i]/(activity_Eu * P_Eu[i]**2 * omega))**2 )
    #delta_Q = Q * np.sqrt( (peak_area_err[i]/peak_area[i])**2 + (activity_Eu_err/activity_Eu)**2 + (P_Eu_err[i]/P_Eu[i])**2 )
    W.append(Q)
    delta_W.append(Q_err)
    print(f"Peak {i+1}: Vollenergienachweiswahrscheinlichkeit W = {Q:.4e} ± {Q_err:.4e}") 


#Regression Vollenergienachweiswahrscheinlichkeit
def W_func(E, a, b):
    return a * E**(-b)

popt, pcov = curve_fit(W_func, E_lit, W)
perr = np.sqrt(np.diag(pcov))
a_q_fit, b_q_fit = popt
print(f"Vollenergienachweiswahrscheinlichkeit Fit: W(E) = {a_q_fit:.4e} ± {perr[0]:.4e} * E^(-{b_q_fit:.4f} ± {perr[1]:.4f})")

x_fit = np.linspace(100, 800, 100)
y_fit = W_func(x_fit, *popt)
#Plot Vollenergienachweiswahrscheinlichkeit
E_peaks = linear(uarray(channel_peaks, channel_peaks_err), m_fit, b_fit)
W_u = uarray(W, delta_W)    
plt.errorbar(noms(E_peaks), noms(W_u), xerr=stds(E_peaks), yerr=stds(W_u), fmt='x', label='Data')
plt.plot(x_fit, y_fit, label='Powser Law Fit')
plt.xlabel(r'Energy $E$ [keV]')
plt.ylabel(r'Full Energy detection probability $Q$')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('plots/Q_Eu.pdf')
plt.clf()



#Cesium Peak Analysis
plt.plot(channels, Cs_bgsub, zorder=1)
plt.fill_between(channels, np.zeros_like(Cs_bgsub), Cs_bgsub, alpha=1.0, label = r'$^{137}$Cs')
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid(True)
plt.savefig('plots/Cs_spectrum.pdf')
plt.clf()

def gauss(x, a, mu, omega, b):
    return a/np.sqrt(2*np.pi*omega**2) * np.exp(-0.5 * ((x - mu) / omega) ** 2) + b

find_peak_cs = find_peaks(Cs_bgsub, height=0.03, distance=20)
#x_peak = channels[peak_cs[0]-30:peak_cs[0]+30]
peak_cs = []
for n in find_peak_cs[0]:
    peak_cs.append(n)
print(peak_cs)





x_peak = channels[peak_cs[0]-30:peak_cs[0]+30]
y_peak = Cs_bgsub[peak_cs[0]-30:peak_cs[0]+30]
popt, pcov = curve_fit(gauss, x_peak, y_peak, p0=[max(y_peak), channels[peak_cs[0]], 5, min(y_peak)])
perr = np.sqrt(np.diag(pcov))
a_fit, x0_fit, sigma_fit, b_fit = popt
a_err, x0_err, sigma_err, b_err = perr
x_fit = np.linspace(channels[peak_cs[0]-30], channels[peak_cs[0]+30], 100)
y_fit = gauss(x_fit, popt[0], popt[1], popt[2], popt[3])

area_cs = a_fit * livetime_Cs
print(f"Area under Cs Peak: {area_cs:.2f} ± {area_cs * np.sqrt((a_err/a_fit)**2 + (sigma_err/sigma_fit)**2):.2f} counts")    

# --- Höhen definieren ---
y_half = b_fit + a_fit/(2* np.sqrt(2*np.pi)*sigma_fit)
y_tenth = b_fit + a_fit/(10* np.sqrt(2*np.pi)*sigma_fit)

# --- Funktion für x-Werte bei gegebenem y ---
def x_for_y(y_target, A, x0, sigma, B):
    yp = y_target - B
    dx = sigma * np.sqrt(-2*np.log(yp / A))
    return x0 - dx, x0 + dx

# Halbwertsbreite x-Werte
x_half_L, x_half_R = x_for_y(y_half, a_fit/(np.sqrt(2*np.pi)*sigma_fit), x0_fit, sigma_fit, b_fit)
# Zehntelbreite x-Werte
x_tenth_L, x_tenth_R = x_for_y(y_tenth, a_fit/(np.sqrt(2*np.pi)*sigma_fit), x0_fit, sigma_fit, b_fit)

# --- Plot ---
plt.plot(x_peak, y_peak, 'x', color='blue', label='Daten')
plt.plot(x_fit, y_fit, 'r-', label='Gauss-Fit')

# Halbmaximum horizontal
plt.hlines(a_fit/2, x_half_L, x_half_R, colors='green', linestyles='--', label='Halbmaximum')


# Zehntelmaximum horizontal
plt.hlines(y_tenth, x_tenth_L, x_tenth_R, colors='purple', linestyles=':', label='Zehntelmaximum')

# Zehntelmaximum vertikal
plt.vlines([x_tenth_L, x_tenth_R], ymin=min(y_peak), ymax=y_tenth, colors='purple', linestyles=':')

plt.xlabel('Channel')
plt.ylabel('Counts')
plt.legend()
#plt.show()
plt.clf()

print("FWHM = ", x_half_R - x_half_L)
print("Tenth width = ", x_tenth_R - x_tenth_L)
print('FWHM in kev:', (x_half_R - x_half_L)*E_m_fit)
print('Tenth width in kev:', (x_tenth_R - x_tenth_L)*E_m_fit)
print('Ratio FWTM/FWHM:', (x_tenth_R - x_tenth_L)/(x_half_R - x_half_L))

plt.plot(channels[peak_cs[0]-30:peak_cs[0]+30], Cs_bgsub[peak_cs[0]-30:peak_cs[0]+30], 'x', color='blue', zorder=1, label='Cs Peak')
plt.plot(x_fit, y_fit, label='Fit')
# Halbmaximum horizontal
plt.hlines(y_half, x_half_L, x_half_R, colors='green', linestyles='--', label='FWHM')
# Zehntelmaximum horizontal
plt.hlines(y_tenth, x_tenth_L, x_tenth_R, colors='purple', linestyles='--', label='FWTM')

plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('plots/Cs_peak_gauss.pdf')
plt.clf()


# print(f"A     = {a_fit:.4e} ± {a_err:.4e}")
# print(f"x0    = {x0_fit:.5e} ± {x0_err:.4e}")
# print(f"sigma = {sigma_fit:.4e} ± {sigma_err:.4e}")
# print(f"B     = {b_fit:.4e} ± {b_err:.4e}")

# print(Cs_bgsub[peak_cs[0]]*livetime_Cs)
cs_area_fit = a_fit * livetime_Cs
E_cs = m_fit*peak_cs[0] + b_fit
print(f"Cs Peak Energy: {E_cs:.2f} keV")




epsilon_cs = E_cs/511
Compton_cs = E_cs * (2*epsilon_cs/(1 + 2*epsilon_cs))
print(f"Compton Edge Energy (theory): {Compton_cs:.5f} keV")
Backscatter_cs = E_cs / (1 + 2*epsilon_cs)
print(f"Backscatter Peak Energy (theory): {Backscatter_cs:.5f} keV")
channel1_cs = (Compton_cs - b_fit)/m_fit
channel2_cs = (Backscatter_cs - b_fit)/m_fit

backscatter_cs = 1618*m_fit + b_fit
compton_cs = 3912*m_fit + b_fit
print(f"Backscatter Peak Channel (measured): {backscatter_cs:.2f}")
print(f"Compton Edge Channel (measured): {compton_cs:.2f}")
# E_backscatter_meas = m_fit*(channel_backscatter) + b_fit
# E_compton_meas = m_fit*(channel_compton) + b_fit
# print(f"Backscatter Peak Energy (measured): {E_backscatter_meas:.2f} keV")
# print(f"Compton Edge Energy (measured): {E_compton_meas:.2f} keV")

plt.plot(channels, Cs_bgsub, zorder=1)
plt.fill_between(channels, np.zeros_like(Cs_bgsub), Cs_bgsub, alpha=1.0, label = r'$^{137}$Cs')
plt.vlines(channel1_cs, ymin=0, ymax=max(Cs_bgsub), colors='red', linestyles='--', label='Compton Edge')
plt.vlines(channel2_cs, ymin=0, ymax=max(Cs_bgsub), colors='orange', linestyles='--', label='Backscatter Peak')
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.yscale('log')
plt.ylim(1e-4, 3e-1)
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('plots/Cs_compton.pdf')
plt.clf()


def linear(x, m, b):
    return m * x + b
x_axe = channels[int(channel2_cs):int(channel1_cs)]
y_axe = Cs_bgsub[int(channel2_cs):int(channel1_cs)]
popt, pcov = curve_fit(linear, x_axe, y_axe)
perr = np.sqrt(np.diag(pcov))
m_fit, b_fit = popt
print(f"Compton Edge Linear Fit: m = {m_fit:.4e} ± {perr[0]:.4e}, b = {b_fit:.4e} ± {perr[1]:.4e}")

I = 0.5*m_fit*(channel1_cs**2 - channel2_cs**2) + b_fit*(channel1_cs - channel2_cs)
print(f"Integral over Compton Continuum: {I:.4e} counts/s")
I_total = I* livetime_Cs
print(f"Total Counts in Compton Continuum: {I_total:.2f} counts")

# def E_fit(channel):
#     return m_fit * channel + b_fit

# def wirkungsquerschnitt(E_gamma, channel, k):
#     epsilon = E_gamma / 511  # E_gamma in keV, 511 keV is the electron rest mass energy
#     return k*(2+(E_fit(channel)/(E_gamma - E_fit(channel)))**2 * ((1/epsilon)**2 + (E_gamma - E_fit(channel))/E_gamma -2*(E_gamma - E_fit(channel))/(epsilon*E_fit(channel))))
# x_cs = channels[int(channel2_cs):int(channel1_cs)]
# y_cs = Cs_bgsub[int(channel2_cs):int(channel1_cs)]
# popt, pcov = curve_fit(wirkungsquerschnitt, x_cs, y_cs)
# perr = np.sqrt(np.diag(pcov))
# k_fit = popt[0]
# print(f"Wirkungsquerschnitt Fit: k = {k_fit:.4e} ± {perr[0]:.4e}")
#plt.plot([channel1_cs, channel2_cs], [Compton_cs, Backscatter_cs], 'o', label='Calculated Points')
x_fit = np.linspace(channel2_cs, channel1_cs, 100)
y_fit = linear(x_fit, *popt)
plt.plot(x_fit, y_fit, label='Linear Fit', alpha = 1, color = 'black', zorder = 2, lw = 2)
plt.plot(x_axe, y_axe, '.', label='Data Points', color = 'blue', zorder = 1)
plt.fill_between(x_axe, np.zeros_like(y_axe), y_axe, alpha=1.0, color = 'blue', zorder = 1)
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('plots/Cs_linear.pdf')
plt.clf()

print(f"Slope of Compton Edge Fit: {m_fit:.4e} ± {perr[0]:.4e} counts/s per channel")
print(f"Intercept of Compton Edge Fit: {b_fit:.4e} ± {perr[1]:.4e} counts/s")




#Barium Peak Analysis
Ba_peaks = find_peaks(Ba_bgsub[:], height=0.013, distance=20)
peak_ba = []
E_ba = []
Q_ba = []
A_ba = [] 
for n in Ba_peaks[0]:
    E = E_m_fit * n + E_b_fit
    Q = a_q_fit * (E)**(b_q_fit)
    E_ba.append(E)
    Q_ba.append(Q)
    peak_ba.append(n)
# print(peak_ba)
#print(E_ba)
#print(Ba_peaks)
print(Q_ba)
plt.plot(channels, Ba_bgsub, zorder=1)
plt.fill_between(channels, np.zeros_like(Ba_bgsub), Ba_bgsub, alpha=1.0, label = r'$^{133}$Ba')
plt.plot(peak_ba, Ba_bgsub[peak_ba], "x", color='red', label='Identified Peaks', zorder=2)
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('plots/Ba_spectrum.pdf')
plt.clf()




#Unknown Peak Analysis
Peaks_low_unknown = find_peaks(unknown_bgsub[:5100], height=0.07, distance=50)
Peaks_high_unknown = find_peaks(unknown_bgsub[5100:7900], height = 0.007, distance=50)
Peaks_unknown = []
for n in Peaks_low_unknown[0]:
    Peaks_unknown.append(n)
for n in Peaks_high_unknown[0]:
    Peaks_unknown.append(n+5100)

plt.plot(channels, unknown_bgsub, zorder=1)
plt.fill_between(channels, np.zeros_like(unknown_bgsub), unknown_bgsub, alpha=1.0, label = 'Unknown Source')
plt.plot(Peaks_unknown[1:], unknown_bgsub[Peaks_unknown[1:]], "x", color='red', label='Identified Peaks', zorder=2)
plt.xlabel('Channel')
plt.ylabel(r'Counts per second   [1/s]')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('plots/Unknown_spectrum.pdf')
plt.clf()