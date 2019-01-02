"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from own_trial import quatern_update
# Estimation parameter of EKF
Q = np.diag([1.1, 1.1])**2  # Observation x,y position covariance
R = np.diag([0.4, 0.4, np.deg2rad(10.0), 0.4])**2  # predict state covariance





show_animation = True


def calc_input(acc, yawrate1):
    #acc = acc  # [m/s^2]
    yawrate = math.radians(yawrate1) # [rad/s]
    u = np.array([[acc, yawrate]]).T
    return u


def observation(xd, u):
    # add noise to input
    ud1 = u[0, 0]
    ud2 = u[1, 0]
    ud = np.array([[ud1, ud2]]).T

    xd = motion_model(xd, ud)

    return xd, ud


def motion_model(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 0.0, 0],
                  [0, 0, 0, 0.0]])

    B = np.array([[(0.5*DT*DT * math.cos(x[3, 0])+DT * math.cos(x[3, 0])), 0],
                  [(0.5 * DT*DT * math.sin(x[3, 0])+DT * math.sin(x[3, 0])), 0],
                  [1.0, 0.0],
                  [0.0, 1.0]])

    x = F.dot(x) + B.dot(u)
    return x

def motion_model_KF(x, u):

    F1 = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 0.0, 0],
                  [0, 0, 0, 0.0]])

    B1 = np.array([[(0.5*DT*DT +DT) , 0],
                  [(0.5 * DT*DT +DT), 0],
                  [1.0, 0.0],
                  [0.0, 1.0]])

    x1 = F1.dot(x) + B1.dot(u)
    return x1


def observation_model(x):
    #  Observation Model
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H.dot(x)

    return z


def jacobF(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+acc*dt*cos(yaw)+0.5*acc*cos(yaw)*dt^2
    y_{t+1} = y_t+acc*dt*sin(yaw)+0.5*acc*sin(yaw)*dt^2
    v_{t+1} = v{t}
    yaw_{t+1} = yaw_t+omega*dt

    so
    dx/dacc = dt*cos(yaw)+0.5*cos(yaw)*dt^2
    dx/dyaw = -acc*dt*sin(yaw)-0.5*acc*sin(yaw)*dt^2
    dy/dacc = sin(yaw)*dt+0.5*sin(yaw)*dt^2
    dy/dyaw = acc*dt*cos(yaw)+0.5*acc*cos(yaw)*dt^2


    """

    acc = u[0]
    yaw = u[1]
    jF = np.array([
        [1.0, 0.0, 0.5*DT*DT * math.cos(yaw)+DT*math.cos(yaw), -acc*DT* math.sin(yaw)-0.5*acc*math.sin(yaw)*DT**2],
        [0.0, 1.0, 0.5*DT*DT * math.sin(yaw)+DT*math.sin(yaw),acc*DT* math.cos(yaw)+0.5*acc*math.cos(yaw)*DT**2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacobF_KF(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+acc*dt*cos(yaw)+0.5*acc*cos(yaw)*dt^2
    y_{t+1} = y_t+acc*dt*sin(yaw)+0.5*acc*sin(yaw)*dt^2
    v_{t+1} = v{t}
    yaw_{t+1} = yaw_t+omega*dt

    so
    dx/dacc = dt*cos(yaw)+0.5*cos(yaw)*dt^2
    dx/dyaw = -acc*dt*sin(yaw)-0.5*acc*sin(yaw)*dt^2
    dy/dacc = sin(yaw)*dt+0.5*sin(yaw)*dt^2
    dy/dyaw = acc*dt*cos(yaw)+0.5*acc*cos(yaw)*dt^2


    """

    acc = u[0]
    yaw = u[1]
    jF1 = np.array([
        [1.0, 0.0, 0.5*DT*DT , acc*DT+0.5*DT**2],
        [0.0, 1.0, 0.5*DT*DT +DT,acc*DT+0.5*acc*DT**2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF1



def jacobH(x):
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH


def ekf_estimation(xEst,xEst_KF,PEst, PEst_KF, z, u):

    #  Predict
    xPred = motion_model(xEst, u)
    xPred_KF  = motion_model_KF(xEst_KF, u)
    jF = jacobF(xPred, u)
    jF_KF = jacobF_KF(xPred_KF, u)
    PPred = jF.dot(PEst).dot(jF.T) + R
    PPred_KF = jF_KF.dot(PEst_KF).dot(jF_KF.T) + R
    #  Update
    jH = jacobH(xPred)
    jH_KF = jacobH(xPred_KF)
    zPred = observation_model(xPred)
    zPred_KF = observation_model(xPred_KF)
    y = z.T - zPred
    y_KF = z.T-zPred_KF
    S = jH.dot(PPred).dot(jH.T) + Q
    S_KF = jH_KF.dot(PPred_KF).dot(jH_KF.T)+Q
    K = PPred.dot(jH.T).dot(np.linalg.inv(S))
    K_KF = PPred_KF.dot(jH_KF.T).dot(np.linalg.inv(S_KF))
    xEst = xPred + K.dot(y)
    xEst_KF = xPred_KF + K_KF.dot(y_KF)
    PEst = (np.eye(len(xEst)) - K.dot(jH)).dot(PPred)
    PEst_KF = (np.eye(len(xEst_KF))) - K_KF.dot(jH_KF).dot(PPred_KF)
    return xEst, PEst, xEst_KF, PEst_KF


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([[math.cos(angle), math.sin(angle)],
                  [-math.sin(angle), math.cos(angle)]])
    fx = R.dot(np.array([[x, y]]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")



def main():
    print(__file__ + " start!!")
    f = open('Parsed_Acc_pgh_133.txt', 'r')
    #time = 0.0
    # State Vector [x y acc yaw]'
    xEst = np.array([[0, 0, 0, 0]]).T
    xEst_KF = np.array([[0, 0, 0, 0]]).T
    xTrue = np.array([[0.0, 0.0, 0, 0]]).T
    #xTrue.shape = (4,1)
    PEst = np.eye(4)
    PEst_KF = np.eye(4)

    xDR = np.array([[0, 0, 0, 0]]).T  # Dead reckoning

    # history
    hxEst = xEst
    hxEst_KF = xEst_KF
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((1, 2))
    DT_hist = 0.0
    line = f.readline()
    yaw_calculate = quatern_update(float(line.split(",")[0]))
    yaw_calculate.time_update(float(line.split(",")[0]))

    while line:  # The file is finite
        line = f.readline()
        splitted = line.split(",")
        yaw_calculate.time_update(float(splitted[0]))
        yaw_calculate.update_quaternion(np.array([float(splitted[4]),float(splitted[5]),float(splitted[6])]))
        rpy = yaw_calculate.quaternion_to_rpy(np.array([splitted[1], splitted[2], splitted[3]]),np.array([splitted[4], splitted[5], splitted[6]]))
        #print(np.array([yaw_calculate.aPitch,yaw_calculate.aRoll,rpy[2]]))
        #print(rpy)
        u = calc_input(-float(splitted[3]),rpy[1])

        global DT
        DT  = yaw_calculate.T
        print(DT)
        if (DT_hist != 0):
            xDR, ud = observation(xDR, u)
            z = np.array([[float(splitted[10]), float(splitted[11])]])
            xEst, PEst, xEst_KF, PEst_KF = ekf_estimation(xEst,xEst_KF, PEst,PEst_KF, z, ud) # z -> insert our xy values here
            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hxEst_KF = np.hstack((hxEst_KF, xEst_KF))
            hxDR = np.hstack((hxDR, xDR))
            #hxTrue = np.hstack((hxTrue, xTrue))
            hz = np.vstack((hz, z))

            if show_animation:
                plt.cla()

                plt.plot(hxTrue[0, :].flatten(),
                        hxTrue[1, :].flatten(), "-b")
                #plt.plot(hxDR[0, :].flatten(),
                #         hxDR[1, :].flatten(), "-k")
                plt.plot(hxEst[0, :].flatten(),
                         hxEst[1, :].flatten(), "-r")
                plt.plot(hxEst_KF[0, :].flatten(),
                         hxEst_KF[1, :].flatten(), "-y")
                plt.plot(hz[:, 0], hz[:, 1], "-g")
                plot_covariance_ellipse(xEst, PEst)
                #plot_covariance_ellipse(xEst_KF,PEst_KF)
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.1)
        #print(np.asarray(hxEst,float).T)
        DT_hist = float(splitted[0])
    f.close()

if __name__ == '__main__':
    main()
