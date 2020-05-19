# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:46:50 2020

@author: jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import numpy as np
import seaborn as sns

NANOGravN = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/NANOGrav.txt',sep='\t')
NANOGravN.columns=['PSRJ']

df1 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Hobbs/Hobbs2004.txt',skiprows=3,sep='\t')
df1.columns=['PSRJ', 'DM', 'dDM/dt']

df2 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Lam/Lam2016.txt',sep='\t')
df2.columns=['PSRJ', 'DM', 'dDM/dt']

df3 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/PSRCAT/PSRCAT.txt',sep='\t')
df3.columns=['PSRJ', 'DM', 'dDM/dt']

df4 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/You/You2007.txt',sep='\t')
df4.columns=['PSRJ', 'DM', 'dDM/dt']

df5 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Petroff/Petroff2013_1.txt',sep='\t')
df5.columns=['PSRJ', 'DM', 'dDM/dt']

df6 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Petroff/Petroff2013_2.txt',sep='\t')
df6.columns=['PSRJ', 'DM', 'dDM/dt']

df7 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Donner/Donner2014.txt',sep='\t')
df7.columns=['PSRJ', 'DM', 'dDM/dt']


frames = [df1]
frames = [df1, df2, df3, df4, df5, df6, df7]
result = pd.concat(frames, ignore_index=True)

er1 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Hobbs/HobbsError.txt',sep='\t', header=None)
er1.columns=['PSRJ', 'Error']

er2 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Lam/LamError.txt',sep='\t')
er2.columns=['PSRJ', 'Error']

er3 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/PSRCAT/PSRCATError.txt',sep='\t')
er3.columns=['PSRJ', 'Error']

er4 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/You/YouError.txt',sep='\t')
er4.columns=['PSRJ', 'Error']

er5 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Petroff/Petroff2013_1wunc.txt',sep='\t')
er5.columns=['PSRJ', 'Error']

er6 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Petroff/Petroff2013_2wunc.txt',sep='\t')
er6.columns=['PSRJ', 'Error']

er7 = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/Donner/Donner2014_wunc.txt',sep='\t')
er7.columns=['PSRJ', 'Error']

frames2 = [er1, er2, er3, er4, er5, er6, er7]
error = pd.concat(frames2, ignore_index=True)

duplicates = pd.concat(g for _, g in result.groupby("PSRJ") if len(g) > 1)
#for index, row in duplicates.iterrows():
    #print(row['PSRJ'], row['DM'], row['dDM/dt'])

result.drop_duplicates(subset ='PSRJ', keep = 'first', inplace = True)
result =result.reset_index(drop=True)

duplicates = pd.concat(g for _, g in error.groupby("PSRJ") if len(g) > 1)
#for index, row in duplicates.iterrows():
    #print(row['PSRJ'], row['Error'])

error.drop_duplicates(subset ='PSRJ', keep = 'first', inplace = True)
error =error.reset_index(drop=True)



for i in range(len(result)):
    result.iat[i,0] = str(result.iat[i,0]).strip()
    result.iat[i,1] = float(result.iat[i,1])
    if isinstance(result.iat[i,2], str):
        s=result.iat[i,2]
        s = re.sub(r'[^\x00-\x7F]+','-', s)
        result.iat[i,2] = float(s)


for i in range(len(error)):
    error.iat[i,0] = str(error.iat[i,0]).strip()
    error.iat[i,1] = float(error.iat[i,1])


NANOGrav = pd.DataFrame()
NANOerror = pd.DataFrame()
for i in range(len(NANOGravN)):
    psrj = NANOGravN.iat[i,0]
    NANOGrav = NANOGrav.append(result[result['PSRJ']==psrj])
    NANOerror = NANOerror.append(error[error['PSRJ']==psrj])
    result = result[result.PSRJ != psrj]
    error = error[error.PSRJ != psrj]
NANOGrav = NANOGrav.reset_index(drop=True)
result = result.reset_index(drop=True)
error = error.reset_index(drop=True)
NANOerror = NANOerror.reset_index(drop=True)

reserror = []
#for i in range(len(error)):
#   if result.iat[i,2] != 0:
#       reserror.append((math.log(math.e)/result.iat[i,2])*error.iat[i,1])
#   else:
#       reserror.append(0)
    
#for i in range(len(error)):
#    if error.iat[i,1] != 0:
#        error.iat[i,1] = math.exp(error.iat[i,1])

from scipy.optimize import curve_fit

def best_fit(x, a, b):
    return a+b*x

def square_root(x):
    return np.sqrt(x)

y1=result['DM'].astype('float64')
y2=result['dDM/dt'].astype('float64')
valid = ~(np.isnan(y1) | np.isnan(y2))

pars, cov = curve_fit(f=best_fit, xdata=y1[valid], ydata=y2[valid], bounds=(-np.inf, np.inf))

plt.plot(y1[valid], y2[valid], '.', label = 'original data')
plt.plot(y1[valid], best_fit(y1[valid],*pars), label = 'best fit',)
plt.legend(loc='lower right')
plt.xlabel('DM')
plt.ylabel('dDM/dt')
plt.title('DM vs dDM/dt')
#plt.xscale('log')
#plt.yscale('log')


result=result.sort_values(by="DM", ascending=True)

#x=np.linspace(1,10**3,100)
f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="log")
ax.plot(result['DM'], result['dDM/dt'], '.')
plt.legend(loc='lower right')
plt.xlabel('DM')
plt.ylabel('dDM/dt')
plt.title('Logarithmic Graph of DM vs dDM/dt')
#ax.plot(x, np.sqrt(x))
#ax.errorbar(result['DM'], result['dDM/dt'], yerr=error['Error'], fmt='.k');
ax.plot(result['DM'][::5], np.sqrt(result['DM'][::5].astype('float64')*0.00001))



f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="log")
plt.legend(loc='lower right')
plt.xlabel('DM')
plt.ylabel('dDM/dt')
plt.title('Logarithmic Graph of DM vs dDM/dt')
#ax.set_ylim([-25,25])
#ax.errorbar(result['DM'], result['dDM/dt'], yerr=error['Error'], fmt='.k');
ax.plot(result['DM'], result['dDM/dt'], '.')
ax.plot(np.sqrt(result['DM'].astype('float64')), np.sqrt(result['dDM/dt'].astype('float64')), 'g')
#sns.regplot(x='DM', y='dDM/dt', data=result, ax=ax, truncate=True)
#ax.errorbar(NANOGrav['DM'], NANOGrav['dDM/dt'], yerr=NANOerror['Error'], ecolor='red', fmt='.k');
ax.plot(NANOGrav['DM'], NANOGrav['dDM/dt'], '.',color='red')
#ax.plot(np.sqrt(NANOGrav['DM'].astype('float64')), np.sqrt(NANOGrav['dDM/dt'].astype('float64')), '.', color='orange')
#sns.lmplot(x='DM', y='dDM/dt', data=NANOGrav)


tau = pd.read_table('C:/Users/jacob/Desktop/SURP Research/CITA Project/Pulsar Tables/PSRCAT/Tau_sc.txt',sep='\t')
tau.columns=['PSRJ', 'Tau_sc']
#tau = tau.reset_index(drop=True)

result2 = pd.DataFrame()
for i in range(len(tau)):
    psrj = tau.iat[i,0]
    if len(result[result['PSRJ']==psrj]) != 0:
        result2 = result2.append(result[result['PSRJ']==psrj])

tau = tau[tau.PSRJ.isin(result2.PSRJ)]

result2 =result2.reset_index(drop=True)
result2.drop_duplicates(subset ='PSRJ', keep = 'first', inplace = True)
print(result2)   


f, ax = plt.subplots(figsize=(7, 7))
plt.legend(loc='lower right')
plt.xlabel('Scattering Timescale')
plt.ylabel('dDM/dt')
plt.title('dDM/dt vs Scattering Timescale')
plt.plot(tau['Tau_sc'], result2['dDM/dt'], '.')











