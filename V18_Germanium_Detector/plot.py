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
plt.savefig('plots/Eu_spectrum.pdf')
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
plt.savefig('plots/Eu_peaks.pdf')
plt.clf()

#Peaks
def Gauss(x, A, x0, sigma, B):
    return A * np.exp(-0.5*(x-x0)**2/sigma**2) + B

print(Peaks)

channel_peaks = []
channel_peaks_err = []
for i in range(len(Peaks)):
    x_peak = channels[Peaks[i]-30:Peaks[i]+30]
    y_peak = Eu_bgsub[Peaks[i]-30:Peaks[i]+30]
    # peak_area = np.sum(y_peak)
    popt, pcov = curve_fit(Gauss, x_peak, y_peak, p0=[max(y_peak), channels[Peaks[i]], 5, min(y_peak)])
    perr = np.sqrt(np.diag(pcov))
    a_fit, x0_fit, sigma_fit, b_fit = popt

    peak_area_fit = a_fit * sigma_fit * np.sqrt(2 * np.pi)
    peak_area_err = peak_area_fit * np.sqrt((perr[0]/a_fit)**2 + (perr[2]/sigma_fit)**2)
    print(f"Peak {i+1}: Channel = {x0_fit:.2f} ± {perr[1]:.2f}, Area = {peak_area_fit:.2f} ± {peak_area_err:.2f}")
    channel_peaks.append(x0_fit)
    channel_peaks_err.append(perr[1])


    # peak_area_sum = np.sum(y_peak[17:-17])
    # print(f"Peak {i+1}: Channel = {x0_fit:.2f} ± {perr[1]:.2f}, Area (sum) = {peak_area_sum:.2f}")


# plt.plot(channels[Peaks[5]-30:Peaks[5]+30], Eu_bgsub[Peaks[5]-30:Peaks[5]+30], 'x', color='blue', zorder=1)
# plt.grid()
# plt.show()



#Linear Regression for Energy Callibration
E_lit = np.array([121.78, 244.70, 344.28, 411.12, 443.97, 778.90])  # in keV

def linear(x, m, b):
    return m * x + b

popt, pcov = curve_fit(linear, channel_peaks, E_lit)
perr = np.sqrt(np.diag(pcov))
m_fit, b_fit = popt
print(f"Energy Calibration: E = {m_fit:.4f} ± {perr[0]:.4f} * channel + {b_fit:.4f} ± {perr[1]:.4f}")
#Plot Energy Calibration
plt.figure(figsize=(10,6))
plt.errorbar(channel_peaks, E_lit, xerr=channel_peaks_err, fmt='o', label='Measured Peaks', color='blue')
x_fit = np.linspace(0, 8000, 100)
y_fit = linear(x_fit, *popt)
plt.plot(x_fit, y_fit, label='Linear Fit', color='red')
plt.xlabel('Channel')
plt.ylabel('Energy [keV]')
plt.legend()
plt.grid()
plt.show()
# plt.savefig('plots/Energy_calibration.pdf')
# plt.clf()