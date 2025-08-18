import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import PIL
from PIL import Image
import numpy as np
import sklearn.preprocessing as sk
import PIL
import cv2
from numpy.linalg import inv
import os
import random



def rect_contains(rect, point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True

def measure_triangle(image, points):
    rect = (0, 0, image.shape[1], image.shape[0])
    sub_div = cv2.Subdiv2D(rect)

    for p in points:
        sub_div.insert(p)

    triangle_list = sub_div.getTriangleList()

    triangle = []
    pt = []

    for t in triangle_list:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                triangle.append((ind[0], ind[1], ind[2]))

        pt = []

    return triangle

def apply_affine_transform(src, src_tri, target_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(target_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha, colorspace = 'BW'):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    #print(r)
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    if colorspace == 'BW':
        mask = np.zeros((r[3], r[2]), dtype=np.float32)
    if colorspace == 'RGB':
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    #print(plt.imshow(mask))
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    size = (r[2], r[3])

    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    #print(warp_image1)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2
    #print(plt.imshow(warp_image1))
    #print(0*(1 - mask) + img_rect * mask)
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask
    return img

def get_morph(del_triangles, src_img, src_points, target_img, target_points, alpha = 0.5, colorspace = 'BW'):

    weighted_pts = []
    for i in range(len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * target_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * target_points[i][1]
        weighted_pts.append((x, y))
    img_morph = np.zeros(src_img.shape, dtype=src_img.dtype)

    img_stack = []
    for triangle in del_triangles:
        x, y, z = triangle
        t1 = [src_points[x], src_points[y], src_points[z]]
        #print(t1)
        t2 = [target_points[x], target_points[y], target_points[z]]
        #print(t2)
        t = [weighted_pts[x], weighted_pts[y], weighted_pts[z]]
        #print(t)
        img_stack.append(morph_triangle(src_img, target_img, img_morph, t1, t2, t, alpha, colorspace = colorspace))
    return img_stack



### Wraped-up function ###

def delaunay_morphing (im_in, im_out, src_points=None, target_points=None, steps = 25, colorspace = 'BW'):
    im_in = (im_in - np.min(im_in))/np.ptp(im_in)
    im_out = (im_out - np.min(im_out))/np.ptp(im_out)
    if src_points == None:
        src_points = [(0, 0), (0, len(im_in)-1), (len(im_in)-1, 0), (len(im_in)-1, len(im_in)-1),
                      (len(im_in)/2, len(im_in)/3), (len(im_in)/4, len(im_in)/4)]
    if target_points == None:
        target_points = [(0, 0), (0, len(im_in)-1), (len(im_in)-1, 0), (len(im_in)-1, len(im_in)-1),
                      (len(im_in)/3, len(im_in)/2), (len(im_in)/4, len(im_in)/4)]
    avg_points = []
    for i in range(len(src_points)):
        x = 0.5 * src_points[i][0] + 0.5 * target_points[i][0]
        y = 0.5 * src_points[i][1] + 0.5 * target_points[i][1]
        avg_points.append((int(x), int(y)))
    triangles = measure_triangle(im_in, avg_points)
    del_morph = []
    for percent in np.linspace(0, 1, num=steps):
        del_morph.append(get_morph(triangles, im_in, src_points, im_out, target_points, alpha=percent, colorspace = colorspace)[:][0]) ### multiples images are created in get_morph OPTIMIZATION TO BE DONE
    return del_morph
