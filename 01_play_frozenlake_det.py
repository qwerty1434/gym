         # -*- coding: utf-8 -*-


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
    
import gym
from gym.envs.registration import register
import sys #tty, termios 도 기존코드에 있었는데 Getch를 GetchWindows로 바꾸면 필요없는거인듯

register(
        id = 'FrozenLake-v3', #내가 원하는 이름
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery' : False} #4x4맵, 미끄럽지 않게
         ) #환경을 우리에 맞게 약간 수정해 준다
 
env = gym.make('FrozenLake-v3') #register id에 등록한 이름 
env.render()

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

#로컬로 할 수 없을까?
#cmd에서는 왜 빨간색 표시가 안될까    
    

    
    
    
    
    
    
    
    
    
    
    
    
    