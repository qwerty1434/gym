# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:25:01 2018

@author: k
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
input_size = env.observation_space.n #우리는 16
output_size = env.action_space.n #우리는 4
learning_rate = 0.1

#인풋데이터와 네트워크 가중치
X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)#placeholder는 값을 넘겨주는데 사용되는 함수 //1x16의 모양으로 넘기겠다  #[[0,1,2,3...,15]]의 1x16 형태
W = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01)) #입력사이즈,출력사이즈에 맞는 weight를 만들어 주세요 // Variable은 학습이 가능한 값 #[[0,1,2,3],[3,1,2,3],[0,5,2,3],...]의 16x4 형태

#실제 Q값(Y)과 네트워크로 예상한 Q값(Qpred)
Qpred = tf.matmul(X,W)#입력 x 가중치 (X와 W의 행렬곱을 실행한다) #[[a1,a2,a3,a4]]의 1x4 형태 #그렇기 때문에 a1에 접근할려면 [0,a]라고 밑에서 쓴다
Y = tf.placeholder(shape=[1, output_size] , dtype= tf.float32)#Y label 받는 값

#loss 계산식
loss = tf.reduce_sum(tf.square(Y-Qpred))#loss 계산 cost(W) = (Ws -y)^2

#loss를 최소화 시키는 과정
train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss) #loss를 최소화 시키게 학습시켜주세요

dis = .99
num_episodes = 2000
rList = []

#one-hot
def one_hot(x):
    return np.identity(16)[x:x+1]

init = tf.global_variables_initializer() #변수 초기화 시키기 위해 필요한거
#학습시작
with tf.Session() as sess:
    sess.run(init) #초기화 실행
    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []
        while not done:
            Qs = sess.run(Qpred, feed_dict = {X: one_hot(s)})#원핫 인코딩한 x를 X값으로 위에서 구한 Qs를 Y레이블로 넘겨서 트레인을 시킨다
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
                
            s1, reward, done, _ = env.step(a)
            #학습하기(y label과 loss function만들기)
            if done: #terminal(마지막 단계이면)
                Qs[0,a] = reward #Q 업데이트 #이게 y값 #각 action에 대한 reward값을 Qs에 저장한다
            else: #not terminal (중간단계면)
                Qs1 = sess.run(Qpred , feed_dict = {X: one_hot(s1)})
                Qs[0,a] = reward + dis * np.max(Qs1) #Q 업데이트 #보상 + discount_rate *다음 상태에서 받을 수 있는 최대의 값
            #학습
            sess.run(train, feed_dict={X:one_hot(s),Y:Qs}) #업데이트 된 Q를 바탕으로 a만 학습시킨다#s1에서 나오는 Q값(Qs1) 중 가장 큰 값을 discount_rate를 곱하고 reward를 더해 Qs[0,a]로 넣는다
            
            rAll += reward
            s = s1
        rList.append(rAll)
#결과확인
print("Percent of succesful episodes : " + str(sum(rList)/num_episodes)+ "%")
plt.bar(range(len(rList)),rList, color = "blue")
plt.show()
            

    
    
    
    
    
    
    
    
    
    