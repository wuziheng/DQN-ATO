#!/usr/bin/env python
# encoding: utf-8
"""
@File   : game
@author : wuziheng
@Date   : 3/20/18 
@license: 
"""
import random
import numpy as np
from train_model import target_v, target_u, train_model

VERSION_NUM = 1
CRASH_LIMIT = 0.3
N_ACTIONS = 5


class Game(object):
    """
    game render for Train_model: to transform train sample point state to a 2D picture.
    @ start_point: train_model start point
    @ end_point: train_model last point
    @ input_shape: width and height of render output
    @ actions: input control state list (fuse control model)
    @ pixel_t: reder parameter, time point interval per pixel
    @ pixel_v: render parameter, velocity interval per pixel
    """
    start_point = 500
    end_point = 600
    input_shape = [80, 80]
    actions = [-10, -5, 0, 5, 10]
    pixel_t = 0.05
    pixel_v = 0.05

    def __init__(self, v_range=0.1, u_range=5):
        """
        game initial every beginning or after crashed, random choose a start point from [start_point,end_point], set
        initial velocity according to the train_model and needed speed curve(not exactly) with in v_range and u_range
        deviation. reset terminal flag
        :param v_range: initial velocity deviation range
        :param u_range: initial force deviation range
        """
        self.t = random.randrange(self.start_point,
                                  self.end_point)
        self.init = self.t
        self.v = target_v(self.t) + random.uniform(-1, 1) * v_range
        self.u = target_u(self.t) + random.uniform(-1, 1) * u_range

        self.d_v = self.v - target_v(self.t)
        self.dd_v = 0
        self.d_u = self.u - target_u(self.t)

        self.observation = self._get_observation()
        self.reward = 0
        self.terminal = False

    def step(self, d_u):
        """
        simulation of train control, to transform train_state from t to t+1 according the train_model with input delta_u
        also get a reward of this step according to our designed game reward system
        :param d_u: one-hot code form delta_u(delta traction force)
        :return: next sample time point train state(image), this step reward, terminal or not
        """
        d_u = np.sum(d_u * self.actions)
        self.u += d_u

        " calculate next state velocity according to train_model with changed u(force) and now velocity"
        self.v = train_model(self.t, self.v, self.u)

        self.t += 1
        self.dd_v = abs(self.v - target_v(self.t)) - abs(self.d_v)
        self.d_v = self.v - target_v(self.t)
        self.d_u = self.u - target_u(self.t)

        " get render state(image)"
        self.observation = self._get_observation()
        self.terminal = abs(self.d_v) > CRASH_LIMIT or self.t >= self.end_point
        reward = self._get_reward()
        return self.observation, reward, self.terminal

    def _get_observation(self):
        """
        generate state (self.input_shape) image. render background, feasible domain, train(may not in image)
        :return: render state(image)
        """
        pixel_t = self.pixel_t
        pixel_v = self.pixel_v
        self.input_shape = [int(int(self.input_shape[0] * pixel_v) / pixel_v),
                            int(int(self.input_shape[0] * pixel_v) / pixel_v)]

        v_target = [target_v(i) for i in range(self.t, self.t + int(self.input_shape[1] * pixel_t))]

        bottom = int(min(v_target)) - 1
        top = bottom + self.input_shape[0] * pixel_v

        self.bottom = bottom
        self.top = top

        img = np.zeros(self.input_shape, dtype='uint8')

        " render the background "
        for i in range(int(self.input_shape[1] * pixel_t)):
            img[int(i / pixel_v):int((i + 1) / pixel_v), :] = (top - i) * 10

        " render the mid feasible domain "
        for i in range(len(v_target)):
            mid = int((v_target[i] - bottom) / pixel_v)
            img[self.input_shape[0] - mid - int(CRASH_LIMIT / pixel_v):self.input_shape[0] - mid + int(
                CRASH_LIMIT / pixel_v),
            int(i / pixel_t):int((i + 1) / pixel_t)] = 0

        " make sure train is now in image and render it"
        train_mid = int((self.v - bottom) / pixel_v)
        if 80 > train_mid > 0:
            img[80 - train_mid - 1:80 - train_mid + 1, 4:6] = 255

        " stack 4 image to one"
        img = np.array(img)
        img = np.stack((img, img, img, img), axis=2)
        self.observation = img

        return self.observation

    def _get_reward(self):
        """
        designed loss function of every step according to the train step after step.
        :return: float
        """
        if abs(self.d_v) > 0.3:
            reward = - 10
        elif abs(self.d_v) < 0.1:
            reward = 1.5 - 10 * abs(self.d_v) - 5 * abs(self.dd_v)
        else:
            reward = 0.2

        self.reward += reward
        return reward

    def render(self):
        fig = plt.gcf()
        fig.set_size_inches(8, 6)

        plt.imshow(game.observation[:, :, 0])

        xgroup_labels = ["t=%4d" % self.t, "t=%4d" % (self.t + 1), "t=%4d" % (self.t + 2), "t=%4d" % (self.t + 3),
                         "t=%4d" % (self.t + 4)]
        ygroup_labels = ["v = %2d" % (self.bottom + 4), "v = %2d" % (self.bottom + 3), "v = %2d" % (self.bottom + 2),
                         "v = %2d" % (self.bottom + 1), "v = %2d" % (self.bottom)]

        x = [0, 20, 40, 60, 79]

        plt.yticks(x, ygroup_labels, rotation=0)
        plt.xticks(x, xgroup_labels, rotation=0)
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    game = Game()
    # one-hot code form of input
    du = [[0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0],
          [1, 0, 0, 0, 0]]
    du = np.array(du)
    for i in range(3):
        game.step(d_u=du[i])
        game.render()
