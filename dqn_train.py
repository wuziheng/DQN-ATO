#!/usr/bin/env python
# encoding: utf-8
"""
@File   : dqn_train
@author : wuziheng
@Date   : 3/21/18 
@license: 
"""
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from collections import deque
import os
import shutil
import matplotlib.pyplot as plt

from model.cnn import inference,dueling
from game.pic_game import Game, N_ACTIONS, VERSION_NUM

GAMENAME = 'train'  # the name of the game being played for log files
ACTIONS = N_ACTIONS  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1


def create_network():
    s = tf.placeholder("float",[None,80,80,4])
    readout = inference(s,ACTIONS)
    return s, readout


def train_network(s, readout, sess):
    # def loss function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open simulator
    game = Game()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[ACTIONS/2] = 1
    observation, reward, terminal = game.step(do_nothing)

    # store memory pool
    D = deque()


    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state('saved_networks/beta%d_version'%VERSION_NUM)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")


    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [observation]})[0]
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[ACTIONS/2] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        _observation, r_t, terminal = game.step(a_t)
        if terminal:
            game = Game()

        D.append((observation, a_t, r_t, _observation, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        observation = _observation
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            if not os.path.exists('saved_networks/beta%d_version' % VERSION_NUM):
                os.mkdir('saved_networks/beta%d_version' % VERSION_NUM)
                shutil.copy('game/pic_game.py', 'saved_networks/beta%d_version/' % VERSION_NUM)

            saver.save(sess, 'saved_networks/beta%d_version/' % VERSION_NUM + 'toy_train' + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("version_num", VERSION_NUM, "TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON %.6f"%epsilon, "/ ACTION", action_index, "/ REWARD %4d" % r_t, \
              "/ Q_MAX %e" % np.max(readout_t), "game t: %4d" % game.t, "game.reward: %4d" % game.reward,
              "delta_v: %.4f" % (game.d_v))


def run_network(s, readout, sess):
    game = Game()
    if not os.path.exists("test_%s" % GAMENAME):
        os.mkdir("test_%s" % GAMENAME)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/beta%d_version" % VERSION_NUM)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    v = []
    target = []

    # game start with do nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[ACTIONS/2] = 1
    observation, r_0, terminal = game.step(do_nothing)
    begin_point = game.t

    v.append(game.v)
    target.append(game.v - game.d_v)

    show = 0

    if not show:
        tlen = []
        for i in range(100):
            if terminal:
                print("train crashed at begin point: %4d!" % begin_point)
            else:
                while not terminal:
                    readout_t1 = readout.eval(feed_dict={s: [observation]})[0]
                    a_t1 = np.zeros([ACTIONS])
                    a_t1[np.argmax(readout_t1)] = 1
                    observation, r_0, terminal = game.step(a_t1)

                    v.append(game.v)
                    target.append(game.v - game.d_v)

                tlen.append(game.t - begin_point)
                print("train travels from %4d to %4d" % (begin_point, game.t))

                # init again
                game.__init__()
                observation, r_0, terminal = game.step(do_nothing)
                begin_point = game.t

        print("train average len :%.4f " % (sum(tlen) / len(tlen)))

    if show:
        if terminal:
            print("train crashed at begin point: %4d!" % begin_point)
        else:
            while not terminal:
                readout_t1 = readout.eval(feed_dict={s: [observation]})[0]
                a_t1 = np.zeros([ACTIONS])
                a_t1[np.argmax(readout_t1)] = 1
                observation, r_0, terminal = game.step(a_t1)

                v.append(game.v)
                target.append(game.v - game.d_v)

            print("train travels from %4d to %4d" % (begin_point, game.t))

        plt.plot(v)
        plt.plot(target)
        plt.show()


def testGame():
    sess = tf.InteractiveSession()
    s, readout = create_network()
    run_network(s, readout, sess)


def playGame():
    sess = tf.InteractiveSession()
    s, readout = create_network()
    train_network(s, readout, sess)


def main():
    playGame()
    #testGame()

if __name__ == "__main__":
    main()