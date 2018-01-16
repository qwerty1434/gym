# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:54:19 2018

@author: k
"""


import gym
    
class _GetchWindows: #키보드 입력 받아들이는 모듈(윈도우용)
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getwch()



inkey = _GetchWindows()

#MACROS -숫자를 다른값을 주면 에러발생(왜?)
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#Key mapping
arrow_keys = {
        'w' : UP,
        's' : DOWN,
        'd' : RIGHT,
        'a' : LEFT} #방향키 할줄 몰라서 wasd로 했다
    
env = gym.make('FrozenLake-v0')
env.render()
state = env.reset()

while True:
    key=inkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    
    action=arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)
    
    if done:
        print("Finished with reawrd", reward)
        break

