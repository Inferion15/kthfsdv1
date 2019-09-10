import numpy as np
from numpy import sin, cos, pi
from sympy import Matrix

class UpdateProcess():

	# updates state and covariance with additional data given
	def update(self, Xt, Pt, Zt, Rt, GPS_Data):
		Xt_shape = Xt.shape[0]

		# start state function 
		h = np.matrix([[float(Xt[0])],
						[float(Xt[1])],
						[float(Xt[3])],
						[float(Xt[4])],
						[float(Xt[5])]])

		# check if recive GPS data
		if GPS_Data: #update with dtGPS rate for every (xGPS, yGPS, speed, yaw_rate, acc)
			# JACOBIAN H
			H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        	[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        	[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        	[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        	[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
		else:
			H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        	[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        	[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        	[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

		# innovation covariance
		S = H*Pt*H.transpose() + Rt

		# Kalman gain
		K = (Pt*H.transpose()) * np.linalg.inv(S)

		# Update the state estimate
		Z = Zt.reshape(H.shape[0],1)
		y = Z - h # innovation matrix
		Xt_f = Xt + (K*y) # final state

		# Update the error covariance 
		I = np.eye(Xt.shape[0])
		Pt_f = (I - (K*H))*Pt

		return Xt_f, Pt_f