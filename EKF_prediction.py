import numpy as np
from numpy import sin, cos, pi
from sympy import Matrix

class PredictionProcess():

	def predict(self, Xt_1, Pt_1, Qt, isSteering, dt):
		Xt = Xt_1

		# Check if it's steering
		if isSteering: # motion functions not linear
			Xt[0] = Xt_1[0] + (Xt_1[3]/Xt_1[4]) * (sin(Xt_1[4]*dt+Xt_1[2]) - sin(Xt_1[2]))
			Xt[1] = Xt_1[1] + (Xt_1[3]/Xt_1[4]) * (-cos(Xt_1[4]*dt+Xt_1[2]) + cos(Xt_1[2]))
			Xt[2] = (Xt_1[2] + Xt_1[4]*dt + pi) % (2.0*pi) - pi # avoid out of range values 
			Xt[3] = Xt_1[3] + Xt_1[5]*dt
			Xt[4] = Xt_1[4]
			Xt[5] = Xt_1[5]
		else: # driving straight - motion functions are linear
			Xt[0] = Xt_1[0] + Xt_1[3]*dt*cos(Xt_1[2])
			Xt[1] = Xt_1[1] + Xt_1[3]*dt*sin(Xt_1[2])
			Xt[2] = Xt_1[2]
			Xt[3] = Xt_1[3] + Xt_1[5]*dt
			Xt[4] = 0.0000001
			Xt[5] = Xt_1[5]

		# Compute the Jacobian matrix(states x states) (partial differential equations in course, speed and yaw_rate)
		# differentiate
		j13 = float((Xt[3]/Xt[4]) * (cos(Xt[4]*dt+Xt[2]) - cos(Xt[2])))
		j14 = float((1.0/Xt[4]) * (sin(Xt[4]*dt+Xt[2]) - sin(Xt[2])))
		j15 = float((dt*Xt[3]/Xt[4])*cos(Xt[4]*dt+Xt[2]) - (Xt[3]/Xt[4]**2)*(sin(Xt[4]*dt+Xt[2]) - sin(Xt[2])))
		j23 = float((Xt[3]/Xt[4]) * (sin(Xt[4]*dt+Xt[2]) - sin(Xt[2])))
		j24 = float((1.0/Xt[4]) * (-cos(Xt[4]*dt+Xt[2]) + cos(Xt[2])))
		j25 = float((dt*Xt[3]/Xt[4])*sin(Xt[4]*dt+Xt[2]) - (Xt[3]/Xt[4]**2)*(-cos(Xt[4]*dt+Xt[2]) + cos(Xt[2])))

		#Jacobian Matrix
		J=np.matrix([[1.0, 0.0, j13, j14, j15, 0.0],
					[0.0, 1.0, j23, j24, j25, 0.0],
					[0.0, 0.0, 1.0, 0.0, dt, 0.0],
					[0.0, 0.0, 0.0, 1.0, 0.0, dt],
					[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
					[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

		#Covariance matrix
		Pt_1=J*Pt_1*J.transpose()+Qt

		return Xt, Pt_1

