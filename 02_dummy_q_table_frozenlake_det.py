# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): #가장 큰 값을 고르는데 가장 큰 값이 없으면 아무값이나 고른다
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
        id = 'FrozenLake-v3', #내가 원하는 이름
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery' : False}
         )

env = gym.make('FrozenLake-v3')


#초기값이 0인 이름이 Q인 array를 만든다
Q = np.zeros([env.observation_space.n,env.action_space.n])#여기 예에서는 16(가능한 s의 개수),4(가능한 a의 개수)

#시행횟수 2000번
num_episodes = 2000

#
rList = [] #결과저장용 리스트
for i in range(num_episodes):
    state = env.reset() # 환경 초기화
    rAll = 0
    done = False
    
    while not done: #게임이 끝나지 않을때까지 무한반복해라 
        action = rargmax(Q[state, :])#Q값을 보고 가장 큰 액션을 선택해라 //Q값이 똑같다면 random하게 선택해라 // rargmax = random argmax
        new_state, reward, done,_ = env.step(action) #env.step()함수로 action을 실행해 그 결과를 변수에 담아라
        #Q 업데이트
        Q[state,action] = reward + np.max(Q[new_state,:]) # : 은 모든경우 즉 위,아래,오른쪽,왼쪽 4가지를 말한다
        #reward를 합한다
        rAll += reward
        #다음 상태로 이동
        state = new_state 
#각각의 성공실패를 하나의 list에 담는다
    rList.append(rAll)

#결과 프린트
print("Success rate: " + str(sum(rList)/num_episodes)) #성공하면 1,실패하면0이기 때문에 전체중 1의 개수
print( "Final Q-Table Values")
print( "LEFT DOWN RIGHT UP")
print(Q)
#언제 성공하고 언제 실패했는지
plt.bar(range(len(rList)),rList,color="blue")
plt.show()

