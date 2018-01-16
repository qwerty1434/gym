# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:58:42 2018

@author: k
"""
#원래있는 deque라는 함수를 이용해 버퍼를 담는다
#우리가 정한 메모리보다 데이터가 많아지면 앞에꺼 지우고 넣는다
 #버퍼에서 10개씩 가져와 학습한다
#np.vstack : 쌓는다
#dqn.update를 이용해 쌓은걸 한번에 학습시킨다

#1. 네트워크 빌드 and initializer
#2. 환경만들기
#3. 루프돌면서 액션가져오기
#4. 액션 취하기(env-step)
#5. 상태나 리워드를 버퍼에 저장(모아두기)
#6. 버퍼에 일정값 모으면 거기서 랜덤샘플 뽑아서 네트워크 학습시키기

import numpy as np
import tensorflow as tf

class DQN:
    
    def __init__(self , session, input_size, output_size, name="main"): #네트워크 initializer 할 때 session 받아오는거 잊지말자
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()
    #1. Go Deep    
    def _build_network(self, h_size = 10, l_rate = 1e-1): #네트워크 만들기
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x")
            W1 = tf.get_variable("W1", shape = [self.input_size, h_size],initializer = tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X,W1))
            W2 = tf.get_variable("W2", shape = [h_size, self.output_size],initializer = tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(layer1,W2) 
        
        self._Y = tf.placeholder(shape = [None, self.output_size], dtype = tf.float32)
    
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
    
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss) 

    def predict(self,state): #직접 Qpred를 부르는게 아니라 predict함수로 호출하겠다
        x = np.reshape(state, [1,self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X:x})

    def update(self,x_stack,y_stack): #x,y를 주면 내가 network의 loss를 minimize하게 train해줄께
        return self.session.run([self._loss, self.train],feed_dict = {self.X:x_stack,self.Y:y_stack})        
    