import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import pandas as pd
import scipy.constants as const
import math
from scipy.optimize import curve_fit



# Diese Funktion liest Daten aus einer CSV-Datei ein und gibt sie als Numpy-Array zurück.
def Data(path):
    df = pd.read_csv(
        path,
        sep=",",          # <- wichtig
        header=1,      # falls keine Header-Zeile
        comment="#",      # falls Kommentarzeilen existieren
        engine="python"
    )
    # Falls es mehr Spalten gibt, hier ggf. einschränken:
    df = df.iloc[:, :2]
    return df.to_numpy(dtype=np.float64)

#Part1: AM mit Trägerunterdrückung
t_1, u_1 = Data("data/scope_1.csv").T

plt.figure(figsize=(10, 6))
plt.plot(t_1 * 1e6, u_1, label="Signal: AM", color="blue")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"$U$ [V]")
plt.grid()
plt.legend()
plt.savefig("build/AM_Trägerunterdrückung.pdf")
plt.clf()

# peak1_1 = 2.1
# peak2_1 = 1.9
# f_T_1 = (peak1_1 + peak2_1) / 2
# f_M_1 = (peak1_1 - peak2_1) / 2
# print("AM mit Trägerunterdrückung:")
# print("Trägerfrequenz f_T_1: ", f_T_1, "MHz")
# print("Modulationsfrequenz f_M_1: ", f_M_1, "MHz")




eps = 1e-3  # anpassen
for i in range(len(t_1)):
    if abs(u_1[i]) < eps:
        print("Nahe Nulldurchgang bei t =", t_1[i], "s")

T1_1 = 9.89*10**(-6)
T2_1 = 9.39*10**(-6)
delta_T_1 = 0.01*10**(-6)
f_T_1 = 1 / (T1_1 - T2_1)
delta_f_T_1 = f_T_1**2 *np.sqrt(2*delta_T_1**2)
print("Trägerfrequenz f_T_1: ", f_T_1 / 1e6, "+/-", delta_f_T_1 / 1e6, "MHz")

f_M_1 = 1/(10*10**(-6))
delta_f_M_1 = f_M_1**2 * 0.1*10**(-6)
print("Modulationsfrequenz f_M_1: ", f_M_1 / 1e3, "+/-", delta_f_M_1 / 1e3, "kHz")




#Part 2: AM mit Trägerabstrahlung
t_2, u_2 = Data("data/scope_2.csv").T

plt.figure(figsize=(10, 6))
plt.plot(t_2 * 1e6, u_2, label="Signal: AM", color="blue")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"$U$ [V]")
plt.grid()
plt.legend()
plt.savefig("build/AM_Trägerabstrahlung.pdf")
plt.clf()


#Modulationsgrad
u_max_max = [17.66, 17.66, 17.61]
u_max_min = [6.80, 6.80, 6.75]
u_min_max = [12.09, 11.72, 12.04, 12.12]
u_min_min = [6.46, 6.46, 6.49, 6.46]

U_max_max = np.mean(u_max_max)
U_max_min = np.mean(u_max_min)
U_min_max = np.mean(u_min_max)
U_min_min = np.mean(u_min_min)

s_max_max = np.std(u_max_max, ddof=1)
s_max_min = np.std(u_max_min, ddof=1)
s_min_max = np.std(u_min_max, ddof=1)
s_min_min = np.std(u_min_min, ddof=1)   

delta_U_max_max = s_max_max / np.sqrt(len(u_max_max))
delta_U_max_min = s_max_min / np.sqrt(len(u_max_min))
delta_U_min_max = s_min_max / np.sqrt(len(u_min_max))
delta_U_min_min = s_min_min / np.sqrt(len(u_min_min))

U_max = ufloat(U_max_max, delta_U_max_max) - ufloat(U_max_min, delta_U_max_min)
U_min = ufloat(U_min_max, delta_U_min_max) - ufloat(U_min_min, delta_U_min_min)

m = (U_max - U_min) / (U_max + U_min)
print("Modulationsgrad m: ", m)
print("Maximalspannung:")
print("U_max_max: ", U_max_max, "+/-", delta_U_max_max)
print("U_max_min: ", U_max_min, "+/-", delta_U_max_min)
print("U_max: ", U_max)
print("Minimalspannung:")
print("U_min_max: ", U_min_max, "+/-", delta_U_min_max)
print("U_min_min: ", U_min_min, "+/-", delta_U_min_min)
print("U_min: ", U_min)


#Modulationsgrad anders
L_t = -60
L_sb = -70
R = 50  # Ohm
P_t = 10**(L_t / 10)  # mW
P_sb = 10**(L_sb / 10)  # mW
U_t = np.sqrt(P_t * R)  # mV
U_sb = np.sqrt(P_sb * R)  # mV
m_2 =  U_sb / U_t
print("Modulationsgrad m (über Seitenbänder): ", m_2)


def U_rms_from_dBm(L_dBm, R=50.0):
    return np.sqrt(R*1e-3) * 10**(L_dBm/20)

def uU_from_uL(U, uL_dB):
    return U * (np.log(10)/20) * uL_dB

L = -60      # dBm
uL = 1       # dB (Beispiel)
U = U_rms_from_dBm(L, R=50.0)
uU = uU_from_uL(U, uL)
U_with_uncertainty = ufloat(U, uU)
print("U mit Unsicherheit: ", U_with_uncertainty)

U_t = U_rms_from_dBm(L_t, R=50.0)
uU_t = uU_from_uL(U_t, 1)
U_t_with_uncertainty = ufloat(U_t, uU_t)    
U_sb = U_rms_from_dBm(L_sb, R=50.0)
uU_sb = uU_from_uL(U_sb, 1)
U_sb_with_uncertainty = ufloat(U_sb, uU_sb)
m_3 = U_sb_with_uncertainty / U_t_with_uncertainty
print("Modulationsgrad m (mit Unsicherheiten): ", m_3)
print("U_t mit Unsicherheit: ", U_t_with_uncertainty)
print("U_sb mit Unsicherheit: ", U_sb_with_uncertainty)



#Frequenzmodulation
def Data_3(path):
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=2,          # <-- die beiden Header-Zeilen überspringen
        header=None,
        engine="python"
    )
    # df hat jetzt 3 Spalten: t, u1, u2
    return df.to_numpy(dtype=np.float64)
t_3, u_3, U_3 = Data_3("data/scope_5.csv").T

plt.figure(figsize=(10, 6))
plt.plot(t_3 * 1e6, u_3, label="Signal: FM", color="blue")
plt.plot(t_3 * 1e6, U_3, label="Signal: FM", color="red")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"$U$ [V]")
plt.grid()
plt.legend()
# plt.savefig("build/FM.pdf")
# plt.clf()


T_1 = [-220, -326, -420.5]
T_2 = [ -244, -350, -441.5]
f_1 =[]
f_2 =[]
delta_F = []
fehler = []
for i in range(len(T_1)):
    f_1.append(10**9 /T_1[i])
    f_2.append(10**9 /T_2[i])
    delta_F.append(abs(f_1[i] - f_2[i]) / (2*10**6))

f_1_mean = np.mean(f_1)
f_2_mean = np.mean(f_2)
s_f_1 = np.std(f_1, ddof=1)
s_f_2 = np.std(f_2, ddof=1)
delta_f_1 = s_f_1 / np.sqrt(len(f_1))
delta_f_2 = s_f_2 / np.sqrt(len(f_2))
print("f_1: ", f_1_mean, "+/-", delta_f_1)
print("f_2: ", f_2_mean, "+/-", delta_f_2)

delta_f = abs(ufloat(f_1_mean, delta_f_1) - ufloat(f_2_mean, delta_f_2))
print("delta_f: ", delta_f)

F_mean = np.mean(delta_F)
s_F = np.std(delta_F, ddof=1)
delta_F_mean = s_F / np.sqrt(len(delta_F))
print("delta_F: ", F_mean, "+/-", delta_F_mean)
m_fm = F_mean / 0.2
delta_m_fm = delta_F_mean / 0.2
print("Modulationsgrad: ", m_fm, "+/-", delta_m_fm)


#Demodulation
delta_t = [10, 16, 20, 26, 32, 38, 44, 50, 56, 62, 70, 78, 86, 96, 100]
delta_y = [1.625, 10.375, 19.875, 33.125, 45.875, 57.375, 69.875, 81.875, 92.375, 102.375, 112.500, 117.125, 118.625, 117.125, 115.125]
phi = []
for i in range(len(delta_y)):
    phi. append(2 * np.pi * 5e6 * delta_t[i] * 1e-9)
plt.figure(figsize=(10, 6))
plt.plot(phi, delta_y, 'rx', label="Messwerte") 
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\Delta U$ [V]")
plt.grid()
plt.legend()
plt.savefig("build/Demodulation.pdf")
plt.clf()

print(phi)