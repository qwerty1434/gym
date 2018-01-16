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
import random
from collections import deque

import gym
env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000


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
        return self.session.run([self._loss, self._train],feed_dict = {self._X:x_stack,self._Y:y_stack})        
    
    #train from replay buffer 버퍼를 통해 학습
def simple_replay_train(DQN,train_batch):
    x_stack = np.empty(0).reshape(0,DQN.input_size)
    y_stack = np.empty(0).reshape(0,DQN.output_size)
        
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
            
        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + dis* np.max(DQN.predict(next_state))
            
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])
            
    return DQN.update(x_stack, y_stack)
    
    #bot play 학습된거를 실행해 보는거    
def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s , reward , done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
def main():
    max_episodes = 5000
        
    replay_buffer = deque()
        
    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size) #네트워크 만들기 #dqn.DQN이였는데 DQN클래스를 다른파일로 정의안하고 여기서 정의한 후 DQN으로 사용
        tf.global_variables_initializer().run() #초기화
            
        for episode in range(max_episodes):
            e = 1. /((episode/10)+1)
            done = False
            step_count = 0
                
            state = env.reset()
                
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample() #random한 액션
                else:
                    action = np.argmax(mainDQN.predict(state)) #우리 네트워크에서 가져온 액선
                    
                next_state, reward, done, _ = env.step(action)  
                if done: #끝나면 -100점
                    reward = -100
                #버퍼에 정보저장    
                replay_buffer.append((state,action,reward,next_state,done))
                if len(replay_buffer) > REPLAY_MEMORY: #너무 많으면 앞에꺼 지우고 넣기
                    replay_buffer.popleft()
                
                state = next_state
                step_count += 1 
                if step_count >10000:
                    break
                
            print("Episode :{} steps: {}".format(episode, step_count))
            if step_count > 10000: #10000번만 해라 (무한히 하면 결과를 못보니깐)
                pass
                break
            if episode % 10 == 1 : #10번에 한번씩 
                for _ in range(50): #50번씩 돌면서 모아둔 버퍼에서 랜덤한 10개를 뽑아 그걸 학습시켜라 
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch) #네트워크를 미니배치로 학습시켜라
                print("Loss: ",loss)
        bot_play(mainDQN)
        
if __name__ == "__main__":
    main()
            
