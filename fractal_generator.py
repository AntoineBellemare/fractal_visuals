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
from utils import contrast, Interpol, fractal_generator
from pythonosc import dispatcher, osc_server
import sys
import time
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio


class Generator(Thread):
    '''
    Generates a sequence of images that interpolates between a base image and a new created fractal
    This thread updates the next_imgs list when to_compute is set to True by the Display thread
    '''
    def __init__(self, interp_steps):
        Thread.__init__(self)
        self.interp_steps = interp_steps

    def run(self):
        global to_compute, next_imgs, FD_control
        while True:
            if to_compute==True:
                if 'last_image' not in locals():
                    last_image = contrast(fractal_generator(0.1, size=size))
                to_compute = False
                FD_control = np.random.random()
                end_image = contrast(fractal_generator(FD_control, last_image.shape[0]))
                next_imgs = Interpol(last_image, end_image, self.interp_steps, direction = 0)
                last_image = end_image


class Display(Thread):
    '''
    cycle_duration : duration in secs of one pass through next_imgs
    This thread displays the images in the next_imgs list, and sets to_compute to True
    to start the generation of the following batch of images
    '''

    def __init__(self, interp_steps, cycle_duration):
        Thread.__init__(self)
        self.waitTime = (int((cycle_duration/interp_steps)*1000)) # REPASSER CA A 1000

    def run(self):
        global next_imgs, to_compute
        while True:
            if next_imgs != []:
                current_imgs = next_imgs
                to_compute = True
                for img in current_imgs:
                    cv2.imshow('Fractal generator', img)
                    cv2.waitKey(self.waitTime)
        #highgui.DestroyWindow('mahMovie)

def renormalize(n, range1, range2):
    '''
    Takes a float with range range1 and rescales it into range range2
    Input :
    n : float
    range1 : tuple
    range2 : tuple
    Output :
    float
    '''
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def complexity2FD(unused_addr, complexity):
    global FD_control
    FD_control = renormalize(complexity, (-2,2), (0.3,0.9))
    print(FD_control)
    return FD_control


class OSCreceive(Thread):
    def __init__(self, addr, port, message):
        Thread.__init__(self)
        self.addr = addr
        self.port = port
        self.message = message


    def run(self):
        global FD_control
        from pythonosc import dispatcher
        dispatcher = dispatcher.Dispatcher()
        dispatcher.map('/svd', complexity2FD)
        server = osc_server.BlockingOSCUDPServer((self.addr, self.port), dispatcher)
        print("Serving on {}".format(server.server_address))
        server.serve_forever()




if __name__ == '__main__':
    global next_imgs, last_image, to_compute, FD_control, size
    size=1024
    cycle_duration = 8
    interp_steps = 12*cycle_duration
    FD_control = 0.3

    to_compute = True
    next_imgs = []

    '''
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map('/svd', complexity2FD)
    server = osc_server.ThreadingOSCUDPServer(('localhost', 5055), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
    '''

    #osc = OSCreceive(addr='localhost', port=5055, message='/svd')
    #osc.start()

    gen = Generator(interp_steps=interp_steps)
    disp = Display(interp_steps=interp_steps, cycle_duration=cycle_duration)
    gen.start()
    disp.start()

#size = 1024
#list_persistence = [0.1, 0.2, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
