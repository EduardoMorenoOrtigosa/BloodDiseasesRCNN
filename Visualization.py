#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os,cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# Reading images
def _readIMG(name):
    return cv2.imread(name)

def _readIMG_RGB(name):
    return cv2.cvtColor(_readIMG(name), cv2.COLOR_BGR2RGB)  

def _readIMG_HSV(name):
    return cv2.cvtColor(_readIMG_RGB(name), cv2.COLOR_RGB2HSV)
    
    
# Showing images
def show_image(name):
    imag = _readIMG(name)
    plt.imshow(imag)
    plt.show()
    
def show_imageRGB(name):
    imag = _readIMG_RGB(name)
    plt.imshow(imag)
    plt.show()
    
def show_image_3d(name):
    
    imag = _readIMG_RGB(name)
    
    r, g, b = cv2.split(imag)
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = imag.reshape((np.shape(imag)[0]*np.shape(imag)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    
def show_imageHSV_3d(name):

    hsv_imag = _readIMG_HSV(name)
    imag = _readIMG_RGB(name)

    pixel_colors = imag.reshape((np.shape(imag)[0]*np.shape(imag)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    h, s, v = cv2.split(hsv_imag)
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def show_imageHSV_3d_axis(name):
    
    hsv_imag = _readIMG_HSV(name)
    imag = _readIMG_RGB(name)

    pixel_colors = imag.reshape((np.shape(imag)[0]*np.shape(imag)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    
    h, s, v = cv2.split(hsv_imag)
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(180, 180)
    plt.show()

    ##### 
    h, s, v = cv2.split(hsv_imag)
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(180, 90)
    plt.show()

    #####
    h, s, v = cv2.split(hsv_imag)
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(90, -90)
    plt.show()
    
def show_mask(name, lo, up):
    
    hsv_imag = _readIMG_HSV(name)
    imag = _readIMG_RGB(name)
    
    mask = cv2.inRange(hsv_imag, lo, up)
    result = cv2.bitwise_and(imag,imag,mask=mask)
    
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

