# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:35:52 2018

@author: k
"""
import numpy as np
import tensorflow as tf

import gym
env= gym.make('CartPole-v0')

#네트워크의 상수
learning_rate = 1e-1
input_size = env.observation_space.shape[0] #4개의 실수인풋
output_size = env.action_space.n #2개의 액션(왼쪽,오른쪽)

#placeholder로 선언
X = tf.placeholder(tf.float32,[None, input_size],name = "input_x")
#신경망의 가중치
W1 = tf.get_variable("W1",shape=[input_size,output_size],
                     initializer = tf.contrib.layers.xavier_initializer()) #4개의 입력을 받아 2개의 출력을 준다
Qpred = tf.matmul(X,W1) #예측값

#placeholder로 선언
Y = tf.placeholder(shape = [None,output_size],dtype=tf.float32)
#Loss Function
loss = tf.reduce_sum(tf.square(Y-Qpred))
#학습
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#q_learning에 필요한 값
num_episodes = 2000
dis = 0.9
rList =[]

#환경설정
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1. / ((i/10)+1)
    rAll = 0
    step_count =0
    s = env.reset()
    done = False
    #Q-network training
    while not done:
        step_count += 1
        x= np.reshape(s,[1,input_size]) #현재상태를 적절한 state로 가공시킨다
        #Qpred값
        Qs = sess.run(Qpred, feed_dict={X: x})  #Qpred에 필요한 X값을 x로 넣고 돌리겠다
        #액션 선택
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done: #마지막 상태
            Qs[0,a] = - 100 #끝나면 -100점 (FrozenLake와 달리 마지막에 보상이 있는게 아니라 끝나면 무조건 벌받아야한다)
        else:
            x1 = np.reshape(s1,[1,input_size])
            Qs1 = sess.run(Qpred, feed_dict={X: x1}) #다음상태에서 취할 수 있는 Q값 #feed_dict: 변수넣는거인듯
            Qs[0,a] = reward + dis*np.max(Qs1) #정답,label,Q의 target에 해당

        sess.run(train, feed_dict= {X: x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i,step_count))

    if len(rList) > 10 and np.mean(rList[-10:]) > 500: #10번동안 500번연속 버티면 잘했다고 판단하고 끝낸다
        break

#학습된 네트워크를 살펴본다
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1,input_size])
    Qs = sess.run(Qpred, feed_dict = {X: x})
    a = np.argmax(Qs)

    observation,reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score:{}".format(reward_sum))
        break

#성능이 별로 안좋다 : 왜? network이 적어서, Qpred가 수렴하지 않고 발산해서
"""
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes <10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done ,_ = env.step(action)
    print(observation,reward,done)
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
"""
