import os
import numpy as np
from numpy import sin, cos, pi
from sympy import Matrix
import csv

# Consider 6 states drawn from data: 
# X_distance, Y_distance (GPS)
# Course, speed, yaw_rate, acceleration (IMU + ODO)

dt = 1.0/30.0  # Time variation between measurements
dtGPS = 1.0/10.0  # Time variation between GPS readings

# Initial state covariance 
Pt = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

# Process variances (assumed)
pGPS = 0.5*7.0*dt**2 # 7.0 m/s^2 maximum acceleration pushing the car
pCourse = 0.2*dt # 0.2 rad/s maximum turn rate
pVelocity = 7.0*dt # maximum speed increase factor (7.0 m/s^2)
pYaw = 0.2*dt # maximum turn acceleration (0.2 m/s^2)
pAcc = 0.5

# Gausian Noise of covariance Qt
Qt = np.diag([pGPS**2, pGPS**2, pCourse**2, pVelocity**2, pYaw**2, pAcc**2])

# Variances of measurement devices
varGPS=5.0
varSpeed=3.0
varYaw=0.1
varAcc=1.0

# Gaussian Noise of covariance Rt
Rt = np.diag([varGPS**2, varGPS**2, varSpeed**2, varYaw**2, varAcc**2])

# ----- CSV Data ---- #
# GPS Data
timestamp, num_satellites, latitude, longitude, altitude = np.loadtxt('GPS_data.csv', delimiter=',', unpack=True, skiprows=1)  

# IMU Data
course, ax, speed, yaw, yaw_rate = np.loadtxt('IMU_data.csv', delimiter=',', unpack=True, skiprows=1)

# Equirectangular aproximation of distance traveled
# Computational efficent / less accurate
# x = R * delta(longitude) * cos(latitude)
# y = R * delta(latitude)
# d(x, y) = R * sqrt(x^2 + y^2) ; R = earth's radius

radiusEarth = 6378388.0 # in m
realRadius = altitude + radiusEarth  # in m

x = realRadius * cos(latitude*pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
y = realRadius * np.hstack((0.0, np.diff(latitude))) # in m
 
# Prox distance traveled vectors
mx = np.cumsum(x)
my = np.cumsum(y)

ds = np.sqrt(x**2+y**2)

# GPS Trigger for Kalman Filter
# Check if the robot has moved(!! GPS Data !!) 
GPS=(ds!=0.0).astype('bool') 

# Initial State
Xt = np.matrix([[mx[0], my[0], course[0]/180.0*pi, speed[0], yaw_rate[0]/180.0*pi, ax[0]]]).transpose()
 
measurements = np.vstack((mx, my, speed, yaw_rate/180.0*pi, ax))
# Lenth of the measurement
m = measurements.shape[1]

from EKF_prediction import PredictionProcess
from EKF_update import UpdateProcess

pred = PredictionProcess()
updt = UpdateProcess()
 
for i in range(measurements.shape[1]):
    print("\n")
    print("Stage:",i)

    # Check if the car is steering 
    if  np.abs(yaw_rate[i])<0.0001:
        isSteering=False
    else:
        isSteering=True

    # Predict state and covariance
    Xt, Pt = pred.predict(Xt,Pt,Qt,isSteering,dt)
    print("Predicted values:")
    print("Xt:\n",Xt,"\nPt:\n",Pt)
   
    Zt=measurements[:,i]
    if GPS[i]:
        GPS_Data=True
    else:
        GPS_Data=False
    
    # Update state and covariance
    Xt_f,Pt_f = updt.update(Xt,Pt,Zt,Rt,GPS_Data)
    print("Final updated values:")
    print("Xt_f:\n",Xt_f,"\nPt_f:\n",Pt_f)

    Xt = Xt_f
    Pt = Pt_f
