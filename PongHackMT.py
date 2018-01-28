# import

import pygame, sys, random, math
from pygame.locals import *

# import
import random
import gym
import numpy as np
import pygame, sys, random
from pygame.locals import *
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import datetime
import time

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.999 # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
        optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
        if not done:
            target = (reward + self.gamma *
            np.amax(self.model.predict(next_state)[0]))
        #print(state)
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# initialise the pygame module
pygame.init()

# setting windows detail
window_width = 500
window_height = 500
ScoreBarHeight = 30

# colors
white = (255, 255, 255)
black = (0, 0, 0)


# set up display size
windowDisplay = pygame.display.set_mode((window_width,window_height), HWSURFACE | DOUBLEBUF | RESIZABLE)

# title
pygame.display.set_caption("PongHackMT")

# 
ball_h = 9
ball_w = 9
paddleP_h = 45
paddleP_w = 15
paddleC_h = 45
paddleC_w = 15

clock = pygame.time.Clock()

ball_img = pygame.image.load('ball.png')
paddle1_img = pygame.image.load('paddle.png')
paddle2_img = pygame.image.load('paddle.png')


def paddle1(paddleP_x,paddleP_y):
        windowDisplay.blit(paddle1_img,(paddleP_x,paddleP_y))
def paddle2(paddleC_x,paddleC_y):
        windowDisplay.blit(paddle2_img,(paddleC_x,paddleC_y))
def ball(ball_x,ball_y):
        windowDisplay.blit(ball_img, (ball_x,ball_y))
def angleCalc(paddle_y, ball_y):
        y =  5* ( (ball_y - (paddle_y + (paddleC_h / 2 ))) / paddleC_h*.5 )
        return y

## hard exits the game
def quit(agent):
    # save the model
    fn = 'weights/pong_weights-' + str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) \
         + '.h5'
    agent.save(fn)
    print('Saved pong weights as',fn)
    print('Exiting..')
    pygame.quit()
    sys.exit()

paddleC_x = window_width - 10 - paddleC_w
paddleP_x = 10
paddleP_y = (0.5*(window_height-ScoreBarHeight))+ScoreBarHeight
paddleC_y = paddleP_y
paddleP_change = 0
paddleC_change = 0
paddle_speed = window_height/105
ball_x = 0.5 * window_width
ball_y = (0.5 * (window_height-ScoreBarHeight))+ScoreBarHeight
ball_xspeed = window_width/160
ball_yspeed = random.uniform(-3,3)*window_height/210
playerScore = 0
cpuScore = 0
paddle_shift=0
paddle_shift_rate=0.6


myFont = pygame.font.SysFont("Courier New", 20, bold=True)


# instantiate the Deep Q Neural Agent
state_size = 8
action_size = 3
agent = DQNAgent(state_size, action_size)

# kinda large
batch_size = 1000

# total rewards throughout the lifetime of the game
total_reward = 0

# how many clocks until exit
epoch = 0
TOTAL_TICKS = 300000

# flag for training mode
TRAINING = False

# deque for the mean of the rewards measured in the matches
mean = deque(maxlen=10000)

print('hackmt pong ai: Training Mode', TRAINING)

# game loop
while epoch < TOTAL_TICKS:

    # current reward for the match
    curr_reward = 0

    if epoch != 0 and epoch % 1000 == 0:
       print ('epoch:', epoch, 'mean: ', np.mean(mean),'e:', agent.epsilon)

    if not TRAINING:
        scoresLine = pygame.draw.rect(windowDisplay, white, (0, ScoreBarHeight-1, window_width, 2), 0)

    while ball_yspeed == 0:
            ball_yspeed = random.uniform(-3,3)


    state = np.array([paddleP_y,paddleP_change, paddleC_y, paddleC_change,
                        ball_x, ball_y, ball_xspeed, ball_yspeed])
    state = np.reshape(state, [1, state_size])
    action = agent.act(state)

    if action == 0:
        paddleC_change = - (paddle_speed)
    # down
    if action == 2:
        paddleC_change = (paddle_speed)

    done = ball_x<0
    if done:
        print('computer scored')


    #Paddle Movement
    for event in pygame.event.get():
        if event.type == QUIT:
            quit(agent)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit(agent)
            if event.key == pygame.K_UP:
                paddleP_change = - (paddle_speed)
            if event.key == pygame.K_DOWN:
                paddleP_change = (paddle_speed)
            if event.key == pygame.K_w:
                paddleC_change = - (paddle_speed)
            if event.key == pygame.K_s:
                paddleC_change = (paddle_speed)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                paddleP_change = 0
            if event.key == pygame.K_w or event.key == pygame.K_s:
                paddleC_change = 0
    
    #randomize left side level
    if paddle_shift_rate >= 1:
        paddle_shift_rate -= 0.7
    if paddle_shift > paddleP_h:
        paddle_shift_rate -= paddleP_h

    if paddleP_y + paddle_shift > ball_y + 0.5 * ball_h:
        paddleP_change = -paddle_shift_rate * (paddle_speed)
    else:
        paddleP_change = paddle_shift_rate * (paddle_speed)


    # bounding box for the paddles
    if paddleP_y + (paddleP_change+paddleP_h) >= window_height+paddle_speed or paddleP_y + (paddleP_change) <= ScoreBarHeight:
        paddleP_change = 0
    if paddleC_y + (paddleC_change+paddleC_h) >= window_height+paddle_speed or paddleC_y + (paddleC_change) <= ScoreBarHeight:
        paddleC_change = 0
    #END Paddle Movement


    #Ball Movement
    paddleP_y += paddleP_change
    paddleC_y += paddleC_change
    ball_x += ball_xspeed

    if not TRAINING:
        pygame.display.update()
        windowDisplay.fill(black)

    paddle1(paddleP_x,paddleP_y)
    paddle2(paddleC_x,paddleC_y)
    #END Ball Movement


    # Ball/Paddle Collision
    #Player Paddle
    if ball_x + ball_xspeed <= paddleP_x + paddleP_w - 1 and ball_x + ball_w - 1 + ball_xspeed >= paddleP_x:
        if ball_y + ball_yspeed <= paddleP_y + paddleP_h - 1 and ball_y + ball_h - 1 + ball_yspeed >= paddleP_y:
            ball_x +=1
            ball_xspeed *= -1
            angle = angleCalc(paddleP_y, ball_y)
            ball_yspeed = ball_xspeed * math.sin(angle)*2
            player_hit = 1
            
            if TRAINING:
                paddle_shift += 3
                paddle_shift_rate += 0.08

    # CPU paddle
    if ball_x + ball_xspeed <= paddleC_x + paddleC_w - 1 and ball_x + ball_w - 1 + ball_xspeed >= paddleC_x:
        if ball_y + ball_yspeed <= paddleC_y + paddleC_h - 1 and ball_y + ball_h - 1 + ball_yspeed >= paddleC_y:
            ball_x -= 1
            ball_xspeed *= -1
            angle = angleCalc(paddleC_y, ball_y)
            ball_yspeed = ball_xspeed * math.sin(angle) *-2
            computer_hit = 1    

            # advance the current reward to 1
            #curr_reward = 1        
    # END Ball/Paddle Collision


    # Ball Out of Bounds
    # If Player Loses
    if (ball_x<0):

        # reset the position of the player paddle
        ball_x = 0.5 * window_width
        ball_y = (0.5 * (window_height-ScoreBarHeight))+ScoreBarHeight
        ball_yspeed = random.uniform(-3,3)
        cpuScore += 1   # increase the scoreboard

    # If CPU Loses
    if (ball_x>window_width):

        # reset the position of the cpu paddle
        ball_x = 0.5 * window_width
        ball_y = (0.5 * (window_height-ScoreBarHeight))+ScoreBarHeight
        ball_yspeed = random.uniform(-3,3)
        playerScore += 1    # increase the scoreboard

    # exit the match
    #if not TRAINING and playerScore == 20:
    #    pygame.quit()
    #    sys.exit()
    # END Ball Out of Bounds



    # Ball Vertical Limit
    if ball_y  + ball_yspeed <= ScoreBarHeight - 1:
        ball_y += (ScoreBarHeight-ball_y)-ball_yspeed
        ball_yspeed = -1* ball_yspeed
    elif ball_y + (ball_h-1) +ball_yspeed >= window_height:
        ball_y += (window_height-(ball_y+ball_h-1))-ball_yspeed
        ball_yspeed = -1* ball_yspeed
    else:
        ball_y += ball_yspeed
    #END Ball Vertical Limit

    # set the coordinated of the ball
    ball(ball_x,ball_y)

    # Update and Display Score
    if not TRAINING:
        cpuScoreDisplay = myFont.render(str(cpuScore), 1, white)
        playerScoreDisplay = myFont.render(str(playerScore), 1, white)
        windowDisplay.blit(cpuScoreDisplay, (window_width*3/4, ScoreBarHeight/2 - 10))
        windowDisplay.blit(playerScoreDisplay, (window_width/4, ScoreBarHeight/2 - 10))
    # END Update and Display Score

    if not TRAINING:
        clock.tick(30)

    if (np.abs(ball_y - paddleC_y) > 25):
        curr_reward = -1 

    next_state = np.array([paddleP_y,paddleP_change, paddleC_y, paddleC_change,
                        ball_x, ball_y, ball_xspeed, ball_yspeed])
    next_state = np.reshape(state, [1, state_size])
    agent.remember(state, action, curr_reward, next_state, done)

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # append the current reward value to the mean deque
    mean.append(curr_reward)
    epoch+=1     # reduce the game ticker down one

quit(agent)