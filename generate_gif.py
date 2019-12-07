import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import scipy.misc
import imageio

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


def fractal_generator(persistence):
    perlin = generate_fractal_noise_2d((256,256), (2,2), octaves=8, persistence=persistence)
    img = Image.fromarray(perlin)
    np.median(perlin)
    for i in range(len(perlin)):
        for j in range(len(perlin)):
            if perlin[i][j] >= 0:
                perlin[i][j] = 1
            elif perlin[i][j] < 0:
                perlin[i][j] = -1
    return perlin

def increments (steps):
    steps = int(100/steps)
    a = range(0,100,steps)
    b = [x for x in a]
    c = b + [100]
    b.reverse()
    d = c + b
    e = [x/100 for x in d]
    return e

def FractalMorphing(imageA, imageB, steps):
    inter_matrix = []
    steps = increments(steps)
    imageA = Image.fromarray(imageA).convert('L')
    imageB = Image.fromarray(imageB).convert('L')
    for k in steps:
        PIL.Image.blend(imageA, imageB, k)
        outImage = np.asarray(imageA) * (1.0 - k) + np.asarray(imageB) * k
        inter_matrix.append(outImage)
        final_matrix = np.array(inter_matrix).T
    return final_matrix


def create_anim(img_arr):
    img = []
    for i in range(img_arr.shape[2]):
        im = plt.imshow(img_arr[:,:,i], animated=True, cmap='gray')
        img.append([im])
    return img



if __name__ == '__main__':
    #np.random.seed(0)
    noise = generate_perlin_noise_2d((256, 256), (8, 8))
    #plt.imshow(noise, cmap='gray', interpolation='lanczos')
    #plt.colorbar()

    #np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (16, 16), 4)
    #plt.figure()
    #plt.imshow(noise, cmap='gray', interpolation='lanczos')
    #plt.colorbar()
    #plt.show()


    image1 = fractal_generator(0.7)
    image2 = fractal_generator(0.8)
    InterpolMatrix = FractalMorphing(image1, image2, 50)

    img_arr = InterpolMatrix



    img = create_anim(img_arr)
    print(len(img))
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, img, interval=50, blit=True,
                                    repeat_delay=None, repeat=True)

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("movie.mp4", writer=writer)

    plt.show()





'''
arr1 = np.zeros((512, 512), dtype=np.uint8)
arr2 = np.ones((512, 512), dtype=np.uint8)
img_arr = np.stack((arr1,arr2), axis=2)
'''
