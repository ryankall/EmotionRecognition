from sklearn import datasets
import cv2
import pygame
import json

# Load the faces datasets
data = datasets.fetch_olivetti_faces()
targets = data.target

data = data.images

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

table = {}

# users label data happy or not
for x in range(1):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', data[x])
    cv2.waitKey(15000)
    cv2.destroyWindow('image')
    emotion = input("Enter 1 for happy or 0 for not happy:")
    if emotion == '1':
        table[x] = True
    else:
        table[x] = False

# write to json file
with open("label_data.xml", 'w') as output:
    json.dump(table, output)



