# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:09:02 2018

@author: k
"""
#최대 200번까지 /벌 -50점 : 146
#최대 1001번까지 / 벌 -150점 : 67
#최대 200번까지 / 벌 -50점 / random한 숫자를 더하는 action 진행 : 30
#최대 200번까지 / 벌 -10점 / 2000번 반복 : 22
#최대 1001번까지 / 벌 -10점 / 2000번 반복 : 52
#최대 9998번까지 / 벌 -100점 / 2000번 반복 : 956
#최대 9998번까지 / 벌 -100점 / 2000번 반복 / random한 숫자를 더하는 action 진행 :  너무잘됨 오래걸려서 껐음 
import numpy as np
import tensorflow as tf
import random
from collections import deque

import gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 9998

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
def replay_train(mainDQN,targetDQN,train_batch): #targetDQN추가됨
    x_stack = np.empty(0).reshape(0,input_size) #DQN.input_size에서 input_size로 수정
    y_stack = np.empty(0).reshape(0,output_size) #DQN.output_size에서 output_size로 수정
        
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
            
        if done:
            Q[0,action] = reward
        else: # a = argmax(mainDQN(s'))
            #Q[0,action] = reward + dis* np.max(mainDQN.predict(next_state))
            Q[0,action] = reward + dis* targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])
            
    return mainDQN.update(x_stack, y_stack) #mainDQN만 학습
    #Network(varialbe) copy 일정 시간이 지날때마다 targetDQN을 mainDQN과 같게 한다
def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name) #src_scope_name에 있는 trainable_variables 변수(루프 돌때마다 변화하는 weight값)를 가져온다
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)
    
    for src_var , dest_var in zip(src_vars,dest_vars):
        op_holder.append(dest_var.assign(src_var.value())) #dest_var에 src_var의 value값을 assign한다 #desc_var == src_var이 된다

    return op_holder


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
    max_episodes = 2000
    
    replay_buffer = deque()
        
    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name = "main") #네트워크 만들기 #dqn.DQN이였는데 DQN클래스를 다른파일로 정의안하고 여기서 정의한 후 DQN으로 사용
        targetDQN = DQN(sess, input_size ,output_size, name = "target") #네트워크가 두개
        tf.global_variables_initializer().run() #초기화
        
        copy_ops = get_copy_var_ops(dest_scope_name = "target" , src_scope_name = "main") #네트워크를 복사
        sess.run(copy_ops)
            
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
                
           #     action = np.argmax(mainDQN.predict(state) + np.random.randn(1,env.action_space.n)/(episode+1))
      
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
                    loss, _ = replay_train(mainDQN,targetDQN, minibatch) #네트워크를 미니배치로 학습시켜라
                print("Loss: ",loss)
                sess.run(copy_ops) #10개로 실행할 때마다 target네트워크를 업데이트 시켜라
        bot_play(mainDQN)
        
if __name__ == "__main__":
    main()
            
