
import numpy as np
import math
import matplotlib.pyplot as plt
import quaternion

class quatern_update(object):

    def __init__(self, Time_T):
        self.quat = quaternion.quaternion(1, 0, 0, 0)
        self.rot = np.matrix([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.T = Time_T
        self.T_0 = 0
        self.gx = 0
        self.gy = 0
        self.gz = 0
        self.aRoll = 0
        self.aPitch = 0
        self.Dtheta = 0
        self.DPhi = 0
        self.DPsi = 0


    def update_quaternion(self, Gy):
        g_x = math.radians(Gy[0])
        g_y = math.radians(Gy[1])
        g_z = math.radians(Gy[2])
        Sw = quaternion.quaternion(0, g_x,g_y,g_z)
        qdot = np.multiply(np.multiply(0.5,self.quat), Sw)
        quat = np.add(self.quat, np.multiply(qdot, self.T))
        quat_arr = quaternion.as_float_array(quat)
        quat_arr = np.divide(quat_arr, math.sqrt((quat_arr[0]**2+quat_arr[1]**2+quat_arr[2]**2+quat_arr[3]**2)))
        self.quat = quaternion.quaternion(quat_arr[0],quat_arr[1],quat_arr[2],quat_arr[3])

    def quaternion_to_rotation(self):
        self.rot = quaternion.as_rotation_matrix(self.quat)



    def quaternion_to_rpy(self,Acc,Gy):
        q = quaternion.as_float_array(self.quat)
        self.yaw =  math.degrees(math.atan2(2.0 * (q[1] *q[2] + q[0] * q[3]),
                                                        -1+2*(q[0] * q[0] + q[1] * q[1])))
        self.pitch = math.degrees(math.asin(-2.0 * (q[1] * q[3] - q[0] * q[2])))
        self.roll = math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                  -1+2*(q[0] * q[0] + q[1] * q[1])))
        #self.acc_roll_picth(Acc,Gy)
        self.acc_roll_pitch(Acc,Gy)
        #self.acc_inclination(Acc)
        return np.array([self.roll, self.pitch,self.yaw])

    def acc_roll_pitch(self, Acc, Gy):
        tmpaRoll = math.atan2(float(Acc[1]),math.sqrt(float(Acc[0])**2+float(Acc[2])**2))
        tmpaPitch = math.atan2(-float(Acc[0]), math.sqrt(float(Acc[1])**2+float(Acc[2])**2))
        self.aRoll = 0.5 * (float(Gy[0])*self.T+self.aRoll)+(1-0.5)*math.degrees(tmpaRoll)
        self.aPitch = 0.5 * (float(Gy[1]) * self.T + self.aPitch) + (1 - 0.5) * math.degrees(tmpaPitch)




    def time_update(self, time_T):
        if self.T_0 == 0:
            self.T_0 = time_T
        else:
            self.T = (time_T - self.T_0)/1000
            self.T_0 = time_T







#aa = quatern_update()
#f = open('Acc_experiment.txt', 'r')
#line = f.readline()

#while line:  # The file is finite
#    line = f.readline()
#    splitted = line.split(",")
#    aa.time_update(float(splitted[0]))
#    aa.update_quaternion(np.array([float(splitted[4]), float(splitted[5]), float(splitted[6])]), aa.T)
#    aa.quaternion_to_rpy(np.array([splitted[1], splitted[2], splitted[3]]),np.array([splitted[4], splitted[5], splitted[6]]))
#    print([aa.aRoll,  aa.aPitch, aa.yaw] )



