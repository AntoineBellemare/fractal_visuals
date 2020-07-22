import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import PIL
from PIL import Image
import scipy.misc
import imageio
import itertools

import glob
import cv2
from threading import Thread, RLock
import random


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise



def fractal_generator(persistence, size):
    perlin = generate_fractal_noise_2d((size,size), (2,2), octaves=8, persistence=persistence)
    #perlin = (perlin)/np.max(np.max(abs(perlin)))
    #for i in range(len(perlin)):
    #    for j in range(len(perlin)):
    #        if perlin[i][j] >= 0:
    #            perlin[i][j] = 1
    #        elif perlin[i][j] < 0:
    #            perlin[i][j] = -1
    return perlin

def increments(steps, direction = 0):
    steps = int(100/steps)
    a = range(0,100,steps)
    b = [x for x in a]

    if direction == 1:
        c = b + [100]
        b.reverse()
        d = c + b
    else:
        c = b
        d = c
    e = [x/100 for x in d]
    return e

def Interpol(imageA, imageB, steps, direction = 0):
    inter_matrix = []
    steps = increments(steps, direction)
    imageA = Image.fromarray(imageA).convert('L')
    imageB = Image.fromarray(imageB).convert('L')
    for k in steps:
        PIL.Image.blend(imageA, imageB, k)
        outImage = np.asarray(imageA) * (1.0 - k) + np.asarray(imageB) * k
        inter_matrix.append(outImage)
        #final_matrix = np.array(inter_matrix).T
    return inter_matrix

def contrast(image):
    for i in range(len(image)):
        for j in range(len(image)):
            if image[i][j] >= 0:
                image[i][j] = 1
            elif image[i][j] < 0:
                image[i][j] = -1
    return image

def InterpolMulti(size, list_image, steps, direction = 0):
    size = size
    InterpolTemp = np.empty((0, size, size))

    for first, second in zip(list_image, list_image[1:]):
        InterpolMatrix = Interpol(first, second, steps, direction)
        InterpolMatrix = np.array(InterpolMatrix)
        InterpolTemp = np.vstack((InterpolTemp, InterpolMatrix))
    InterpolMulti = np.array(InterpolTemp)
    return InterpolMulti

def Generate_list_images(list_persistence, size):
    list_temp = []
    for x in list_persistence:
        image = contrast(fractal_generator(x, size))
        list_temp.append(image)
    list_image = np.array(list_temp)
    return list_image

def create_anim(img_arr):
    img = []
    for i in range(img_arr.shape[2]):
        im = plt.imshow(img_arr[:,:,i], animated=True, cmap='gray')
        img.append([im])
    return img

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



class Generator(Thread):
    # Generates a sequence of images that interpolates between a base image and a new created fractal
    # Options are the base image, end image FD, interpolation_params)
    def __init__(self, interp_steps):
        Thread.__init__(self)
        self.interp_steps = interp_steps

    def run(self):
        global to_compute, next_imgs
        while True:
            if to_compute==True:
                if 'last_image' not in locals():
                    last_image = contrast(fractal_generator(0.1, size=1024))
                    print('created first image')
                to_compute = False
                FD_control = random.randint(0,9)/10
                end_image = contrast(fractal_generator(FD_control, last_image.shape[0]))
                print('start')
                next_imgs = Interpol(last_image, end_image, self.interp_steps, direction = 0)
                last_image = end_image
                print('stop')
                print(last_image)

class Display(Thread):
    # cycle_duration : duration in secs of one pass through next_imgs
    def __init__(self, interp_steps, cycle_duration):
        Thread.__init__(self)
        self.waitTime = (int((cycle_duration/interp_steps)*1000))

    def run(self):
        global next_imgs, to_compute
        while True:
            if next_imgs != []:
                print('start disp loop')
                current_imgs = next_imgs
                #pdb.set_trace()
                to_compute = True
                for img in current_imgs:
                    cv2.imshow('mahMovie', img)
                    cv2.waitKey(self.waitTime)

        #highgui.DestroyWindow('mahMovie)





if __name__ == '__main__':
    global next_imgs, last_image, to_compute

    cycle_duration = 4
    interp_steps = 24*cycle_duration

    to_compute = True
    next_imgs = []
    gen = Generator(interp_steps=interp_steps)
    disp = Display(interp_steps=interp_steps, cycle_duration=cycle_duration)
    gen.start()
    disp.start()

#size = 1024
#list_persistence = [0.1, 0.2, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
