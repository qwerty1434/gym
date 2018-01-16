# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:29:06 2018

@author: k
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
        id = 'FrozenLake-v3', #내가 원하는 이름
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery' : False}
         )

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])

dis = 0.99 #람다, 감소율
num_episodes = 2000

rList = [] #결과저장용 리스트
for i in range(num_episodes):
    state = env.reset() # 환경 초기화
    rAll = 0
    done = False
    
    while not done: #게임이 끝나지 않을때까지 무한반복해라 
        action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)/(i+1)) #노이즈 추가방식 

        new_state, reward, done,_ = env.step(action) #env.step()함수로 action을 실행해 그 결과를 변수에 담아라
        #Q 업데이트
        Q[state,action] = reward + dis*np.max(Q[new_state,:]) # 미래에 받을 reward는 discount 시켜 받자
        #reward를 합한다
        rAll += reward
        #다음 상태로 이동
        state = new_state 

    rList.append(rAll)

#결과 프린트
print("Success rate: " + str(sum(rList)/num_episodes))
print( "Final Q-Table Values")
print( "LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList,color="blue")
plt.show()


