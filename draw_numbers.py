import numpy as np
import pandas as pd
import pygame
from neural_network import *
import pygame
from pygame.locals import QUIT

pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)

clock = pygame.time.Clock()
framerate = 200

font = pygame.font.SysFont("Arial Black", 30)

def redraw(screen):
    screen.fill((70, 70, 70))
    # draw white rectangle (canvas background)
    for i in range(28):
        for j in range(28):
            value = canvas[i][j]
            value = 255 - 255*value
            value = max(0,value)
            pygame.draw.rect(screen, (value,value,value), (i*gridsize*28 + 50, j*gridsize*28 + 50, 28*gridsize,28*gridsize))
    pygame.display.update()

def draw(gridsize, mousex, mousey, canvas, mid_increment, edge_increment):
    for i in range(28):
        for j in range(28):
            if ((mousex >= j*gridsize*28 + 50 and mousex <= j*gridsize*28 + 50 + gridsize*28) and
                        (mousey >= i*gridsize*28 + 50 and mousey <= i*gridsize*28 + 50 + gridsize*28)):
                        canvas[j][i] += mid_increment
                        canvas[j-1][i] += edge_increment
                        canvas[j+1][i] += edge_increment
                        canvas[j][i-1] += edge_increment
                        canvas[j][i+1] += edge_increment


width = 800
height = 500
screen = pygame.display.set_mode((width, height))
gridsize = 0.5

inPlay = True
clicking = False

canvas = [[0] * 28 for i in range(28)]


while inPlay:
    
    if clicking:
        for i in range(200):
            draw(gridsize, mousex, mousey, canvas, 0.0002, 0.0001)
    
    if clicking:
        for i in range(28):
                for j in range(28):
                    if ((mousex >= j*gridsize*28 + 50 and mousex <= j*gridsize*28 + 50 + gridsize*28) and
                        (mousey >= i*gridsize*28 + 50 and mousey <= i*gridsize*28 + 50 + gridsize*28)):
                        canvas[j][i] += 0.01

    redraw(screen)
    clock.tick(framerate)

    mousex,mousey = pygame.mouse.get_pos()
    for event in pygame.event.get():
        pressed = pygame.key.get_pressed()

        if event.type == QUIT:
            pygame.quit()
            inPlay = False

        # used to check if buttons are pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
            clicking = True

        if clicking:
            pass
                        # canvas[j-1][i] += 0.1
                        # canvas[j+1][i] += 0.1
                        # canvas[j][i-1] += 0.1
                        # canvas[j][i+1] += 0.1


        # if release mouse button
        if event.type == pygame.MOUSEBUTTONUP:
            clicking = False

        if pressed[pygame.K_BACKSPACE]:
            for i in range(len(canvas)):
                for j in range(len(canvas[0])):
                    canvas[i][j] = 0

        if pressed[pygame.K_RETURN]:
            inp = []
            for i in range(len(canvas)):
                inp += canvas[i]
            inp = np.matrix(inp).reshape(784,1)

            # Forward propagation input -> hidden
            h_pre = bih + wih @ inp
            h = ReLU(h_pre)
            # Forward propagation hidden -> output
            o_pre = bho + who @ h
            o = softmax(o_pre)

            print("Prediction:", o.argmax())
            print(o)
        