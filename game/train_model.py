#!/usr/bin/env python
# encoding: utf-8
"""
@File   : model
@author : wuziheng
@Date   : 3/20/18 
@license: 
"""

def train_model(t, v_t, u_t):
    """
    fitting curve by Train csv data, describe the relationship between the next sample point time with this point
    velocity and traction force. this is called dynamic system description
    :param t: Train sample time point
    :param v_t: t sample point velocity
    :param u_t: t sample point traction force
    :return: t+1 sample point velocity
    """
    if t < 500:
        v_t1 = -0.0002 * (v_t ** 2) + 1.0005 * v_t + 0.0054 * u_t - 0.0035
    elif t < 1000:
        v_t1 = -0.00007 * (v_t ** 2) + 1.0007 * v_t + 0.0050 * u_t - 0.0026
    elif t < 1500:
        v_t1 = 0.00006 * (v_t ** 2) + 0.9987 * v_t + 0.0061 * u_t - 0.0030
    elif t < 2000:
        v_t1 = 0.0002 * (v_t ** 2) + 0.9987 * v_t + 0.0066 * u_t - 0.0041
    else:
        v_t1 = 0.0004 * (v_t ** 2) + 0.9977 * v_t + 0.0061 * u_t - 0.0030
    return v_t1


def utrain_model(t, v_t1, v_t):
    """
    :param t: Train sample time point
    :param v_t1: t+1 sample point velocity
    :param v_t: t sample point velocity
    :return: need traction force according to the Train model
    """
    if t < 500:
        u_t = (v_t1 + 0.0002 * (v_t ** 2) - 1.0005 * v_t + 0.0035) / 0.0054
    elif t < 1000:
        u_t = (-0.00007 * (v_t ** 2) + 1.0007 * v_t - 0.0026 - v_t1) / (- 0.0050)
    elif t < 1500:
        u_t = (0.00006 * (v_t ** 2) + 0.9987 * v_t - 0.0030 - v_t1) / (-0.0061)
    elif t < 2000:
        u_t = (0.0002 * (v_t ** 2) + 0.9987 * v_t - 0.0041 - v_t1) / (-0.0066)
    else:
        u_t = (0.0004 * (v_t ** 2) + 0.9977 * v_t - 0.0030 - v_t1) / (-0.0061)
    return u_t


def curve(t):
    """
    target curve for the Train during the sample fragment
    :param t: time point
    :return: target velocity at t point
    """
    if 100 > t >= 0:
        velocity = (0.8 * (t / 20) ** 2)
    elif 200 > t >= 100:
        velocity = 40 - 0.8 * (t / 20 - 10) ** 2
    elif 400 > t >= 200:
        velocity = 40
    elif 500 > t >= 400:
        velocity = 0.6 * (t / 20 - 20) ** 2 + 40
    elif 600 > t >= 500:
        velocity = 70 - 0.5 * (t / 20 - 30) ** 2
    elif 1800 > t >= 600:
        velocity = 70
    elif 1900 > t >= 1800:
        velocity = 70 - 0.6 * (t / 20 - 90) ** 2
    elif 2000 > t >= 1900:
        velocity = 40 + 0.6 * (t / 20 - 100) ** 2
    elif 2200 > t >= 2000:
        velocity = 40
    elif 2300 > t >= 2200:
        velocity = 40 - 0.8 * (t / 20 - 110) ** 2
    elif 2400 > t >= 2300:
        velocity = 0.8 * (t / 20 - 120) ** 2
    else:
        velocity = 0
    return velocity


def target_v(t):
    return sum([curve(i) for i in range(t-10, t+10)])/20


def target_u(t):
    return utrain_model(t,target_v(t+1),target_v(t))

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    t_v = [target_v(i) for i in range(2500)]
    t_u = [target_u(i) for i in range(2500)]

    plt.figure(1)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.subplot(212);plt.title('target u');plt.plot(t_u);plt.ylabel("u")
    plt.subplot(211);plt.title('target v');plt.plot(t_v);plt.ylabel("v(km/h)");plt.ylim(0,80)
    fig.savefig('../fig/target_v_u.png', dpi=100)


    import csv
    import numpy as np

    s_u = []
    s_v = []
    with open('../csv/1.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            s_v.append(int(row[0]))
            s_u.append(int(row[1]))

    sample_start = 50
    sample_end = 2400
    s_u = np.array(s_u)[sample_start:sample_end]
    s_v = np.array(s_v)[sample_start:sample_end]

    plt.figure(2)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.subplot(212);plt.title('target u');plt.plot(s_u);plt.ylabel("u")
    plt.subplot(211);plt.title('target v');plt.plot(s_v/10.0);plt.ylabel("v(km/h)");plt.ylim(0,80)
    fig.savefig('../fig/sample_v_u.png', dpi=100)