#!/usr/bin/env python
"""
Sample script that uses the Novoptel package created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

# In Spyder console type:
# >>> pip install ftd2xx
# >>> pip install pypiwin32
# After a restart of Spyder both packets are listed in pip, and you can access the FTDI devices:
# >>> pip list
# Package                            Version  
# ---------------------------------- ---------
# ...
# ftd2xx                             1.1.2    
# ... 
# pypiwin32                          223      
# ...
# With another Python installation, pypiwin32 was automatically replaced/superseded by pywin32.

# To connect ftd3xxx devices:
# See https://ftdichip.com/software-examples/ft600-601-software-examples/
# Download https://ftdichip.com/wp-content/uploads/2020/07/D3XXPython_Release1.0.zip
# Unpack it. In open Anaconda3 (64-bit), Anaconda Powershell Prompt (Anaconda3), navigate
# by "cd D3XXPython_Release1.0" etc and enter: python setup.py install
# From https://ftdichip.com/software-examples/ft600-601-software-examples/ download D3XX.
# After unpacking, copy FTD3XX.dll into the directory C:\ProgramData\Anaconda3.
# Now you are read to use ftd3xx.

# For the package Python_USB you need Matlab Runtime R2021a (9.10). Download and install it,
# starting from https://de.mathworks.com/products/compiler/matlab-runtime.html

from __future__ import print_function
import os                                 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Deprecated! See https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
# import ftd2xx as ftd
# import ftd3xx as ftd3
# import sys

# How to work without Matlab: It is possible to run Matlab .m files on Octave.
# Octave can be run from Python, using oct2py. 
# In addition, a Python package Python_USB has been compiled from the underlying Matlab .m files.
# initepc, initeps, initpm, initlu are run to connect EPC1000, EPS1000, PM1000, LU1000. 
# readepc, readeps, readpm, readlu read from these devices. 
# writeepc, writeeps, writepm, writelu write into these devices. 
# closeepc, closeeps, closepm, closelu write into these devices. 
# Without argument, initepc, initeps, initpm, initlu load
# LastDevEPC.mat, LastDevEPS.mat, LastDevPM.mat, LastDevLU.mat,
# where the last connected devices are stored. The file format is Matlab .mat with
# option "-v6", which is compatible with Octave. 
# Initially and when devices are changed these files must be created.
# We create LastDevEPC.mat, LastDevEPS.mat, LastDevPM.mat, LastDevLU.mat
# by running initepc(0), initeps(0), initpm(0), initlu(0) under oct2py. 
# Then closeepc, closeeps, closepm, closelu are run under oct2py.
# This frees EPC1000, EPS1000, PM1000, LU1000 for subsequent connection under my_Novoptel.
# Before this connection, device data is loaded from the 4 files, under oct2py.
# To call initepc, initeps, initpm, initlu under Python, the type argument must be 2
# and a second argument is needed, the device descritptor string.

# USB test program for Novoptel EPC1000, EPS1000, PM1000, LU1000

# ATTENTION: This or equivalent seems to be needed in some cases:
#pathToExecutable = ('C:\\Program Files\\GNU Octave\\Octave-6.1.0\\mingw64\\bin\\octave-cli.exe')
#os.environ['OCTAVE_EXECUTABLE'] = pathToExecutable from oct2py import Oct2Py oc = Oct2Py()
# pathToExecutable = (
#   'C:\\Program Files\\GNU Octave\\Octave-6.1.0\\mingw64\\bin\\octave-cli.exe'
#   # 'C:\Program Files\GNU Octave\Octave-6.1.0\mingw64\bin\octave-cli.exe'
# )
# os.environ['OCTAVE_EXECUTABLE'] = pathToExecutable

from oct2py import Oct2Py
oc = Oct2Py()
# If needed, modify the following command in order to point to the directory 
# where the utility files for GNU Octave are located.
# These can be downloaded from https://www.novoptel.de/Home/Downloads_en.php
# or https://www.novoptel.eu/Home/Downloads_en.php , currently
# https://www.novoptel.de/Home/Downloads/Octave_USB_Support_Files.zip or
# https://www.novoptel.eu/Home/Downloads/Octave_USB_Support_Files.zip .
oc.addpath('..\\OCT');  # for GNU Octave, version 5.2.0 
# oc.addpath('..\\OCT6'); # for GNU Octave, version 6.1.0
oc.addpath('..\\COM');


#############################################################################
#############################################################################

# The following demonstrates the selection of EPC1000, EPS1000, PM1000, LU1000 under oct2py.
# 
# None of this is needed unless one subsequently wants to use the Python package Python_USB

stri = input('Select EPC1000 device (0) or look it up in LastDevEPC.mat (1 or "enter"):')
if stri=='0':
    oc.initepc(0)
else:
    oc.initepc()
# oc.load('LastDevEPC.mat', 'LastDevDescr') is a dictionary. We read the element 'LastDevDescr'.
DevDescrEPC = oc.load('LastDevEPC.mat', 'LastDevDescr')['LastDevDescr']
print(DevDescrEPC) 
    
stri = input('Select EPS1000 device (0) or look it up in LastDevEPS.mat (1 or "enter"):')
if stri=='0':
    oc.initeps(0)
else:
    oc.initeps()
# oc.load('LastDevEPS.mat', 'LastDevDescr') is a dictionary. We read the element 'LastDevDescr'.
DevDescrEPS = oc.load('LastDevEPS.mat', 'LastDevDescr')['LastDevDescr']
print(DevDescrEPS) 
    
stri = input('Select PM1000 device (0) or look it up in LastDevPM.mat (1 or "enter"):')
if stri=='0':
    oc.initpm(0)
else:
    oc.initpm()
# dummy is a dictionary. We read its elements.
dummy       = oc.load('LastDevPM.mat',  'LastDevDescr', 'LastDevType')
DevDescrPM  = dummy['LastDevDescr']
typePM      = dummy['LastDevType' ]
print(DevDescrPM, typePM, sep='\n')

stri = input('Select LU1000 device (0) or look it up in LastDevLU.mat (1 or "enter"):')
if stri=='0':
    oc.initlu(0)
else:
    oc.initlu()
# oc.load('LastDevLU.mat', 'LastDevDescr') is a dictionary. We read the element 'LastDevDescr'.
DevDescrLU  = oc.load('LastDevLU.mat',  'LastDevDescr')['LastDevDescr']
print(DevDescrEPS) 
    
print()

# If you wish you can now read and write under oct2py:
    
# Read whether EPC1000 control is on or off:
res, ok = oc.readepc(1, nout=2)
print(res, ok, sep='\n')
# Set control to on, regardless of ONOFF line status
ok = oc.writeepc(1, 3)
print(ok, sep='\n')

# Read HWP rotation status of EPS1000:
res, ok = oc.readeps(0, nout=2)
print(res, ok, sep='\n')
# Set HWP to forward rotation:
ok = oc.writeeps(0, 1)
print(ok, sep='\n')

# Read averaging time exponent ATE of PM1000:
res, ok = oc.readpm(512+1, nout=2)
print(res, ok, sep='\n')
# Stay in the normal input power mode of PM1000, don't switch to the high-sensitivity mode
ok = oc.writepm(512+196, 1)
print(ok, sep='\n')
# Disable the auto-averaging at low input power
ok = oc.writepm(512+220, 1)
print(ok, sep='\n')

# Read power of laser 1 in LU1000:
res, ok = oc.readlu(128+49, nout=2)
print(res, ok, sep='\n')
# Set power of laser 1 in LU1000 to 12.34 dBm:
ok = oc.writelu(128+49, 1234)
print(ok, sep='\n')

# Before you can use the instruments by any other software you must close them in oc:
    
ok = oc.closeepc()
print(ok, sep='\n')
ok = oc.closeeps()
print(ok, sep='\n')
ok = oc.closepm()
print(ok, sep='\n')
ok = oc.closelu()
print(ok, sep='\n')

# Now we can use the instruments under the Python package Python_USB.
# We could have done so right away, without oct2py. But oct2py allows init*(0) to be called,
# which yields a list of devices among which you can select.
# And anyway, unless the desired device descriptors and type PM are known you have to read them
# from the LastDev*.mat files, which has been done above.

import Python_USB
import matlab                             # Needed for support of package compiled from Matlab
my_Novoptel = Python_USB.initialize()

# Initialize the instruments under my_Novoptel. Other than before under oc,
# we collect more return parameters.

print('initepc:  ')
typeEPC     = matlab.uint16([2])
#DevDescrEPC = "EPC1000-100-Test"
okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut = my_Novoptel.initepc(typeEPC, DevDescrEPC, nargout=5)
print(okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut, sep='\n')
print()

print('initeps:  ')
typeEPS     = matlab.uint16([2])
#DevDescrEPS = "EPS1000-20M-XL-S-FP-LN-D 999"
okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut = my_Novoptel.initeps(typeEPS, DevDescrEPS, nargout=5)
print(okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut, sep='\n')
print()

print('initpm:   ')
#typePM      = matlab.uint16([2]) # activate for USB2.0 device
#DevDescrPM  = "PM1000-100M-XL-FA-NN-D 44"
#typePM      = matlab.uint16([3]) # activate for USB3.0 device
#DevDescrPM  = "PM1000-100M_speed_test"
okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut = my_Novoptel.initpm(typePM,  DevDescrPM,  nargout=5)
print(okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut, sep='\n')
print()

print('initlu:   ')
typeLU     = matlab.uint16([2])
#DevDescrLU = "LU1000-CC-FP-L-D 00020"
okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut = my_Novoptel.initlu(typeLU,  DevDescrLU,  nargout=5)
print(okOut, diagnosisOut, handleoutOut, LastDevDescr_inOut, LastDevDescr_outOut, sep='\n')
print()

# Read/write as needed

# Read serial number of EPC1000: 
res, okOut = my_Novoptel.readepc(matlab.double([23]), nargout=2)
print(res, okOut, sep='\n')
# Set averaging time exponent ATE of EPC1000 to 4:
okOut = my_Novoptel.writeepc(matlab.double([32]),matlab.double([4])); 
print(okOut, sep='\n')

# Read QWP0 position index of EPS1000: 
res, okOut = my_Novoptel.readeps(matlab.double([41]), nargout=2)
print(res, okOut, sep='\n')
# Set QWP5 to backward rotation:
okOut = my_Novoptel.writeeps(matlab.double([6]),matlab.double([3])); 
print(okOut, sep='\n')

# Read S3 sign setting of PM1000: 
S3sign, okOut = my_Novoptel.readpm(matlab.double([512+45]), nargout=2)
print(S3sign, okOut, sep='\n')
# Debouncing of PM1000 BNC trigger input is set to minimum:
okOut = my_Novoptel.writepm(matlab.double([512+226]),matlab.double([0])); # Delay/Debouncing 0*10ns
print(okOut, sep='\n')

# Read channel number of laser 1 in LU1000:
res, okOut = my_Novoptel.readlu(matlab.double([128+48]), nargout=2)
print(res, okOut, sep='\n')
# Set channel number of laser 1 in LU1000 to 17:
okOut = my_Novoptel.writelu(matlab.double([128+48]),matlab.double([17])); 
print(okOut, sep='\n')

# Before you can use the instruments by any other software you must close them in my_Novoptel:
    
print('closeepc: ')
okOut = my_Novoptel.closeeps()
print(okOut, sep='\n')

print('closeeps: ')
okOut = my_Novoptel.closeeps()
print(okOut, sep='\n')

print('closeepm: ')
okOut = my_Novoptel.closepm()
print(okOut, sep='\n')

print('closeelu: ')
okOut = my_Novoptel.closelu()
print(okOut, sep='\n')

# stop operation of package:

my_Novoptel.terminate()
