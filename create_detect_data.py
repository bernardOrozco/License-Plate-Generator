# -*-coding: UTF-8 -*-
import numpy as np
from genplate_advanced import *
import os
import pandas as pd
import pickle
import cv2
import time
import csv

index = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
         "6": 6, "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17, "I":18,
         "J": 19, "K": 20, "L": 21, "M": 22, "N": 23, "P": 24, "Q": 25, "R": 26, "S": 27, "T": 28, "U": 29, "V": 30,
         "W": 31, "X": 32, "Y": 33, "Z": 34}

chars = [u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"I", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z"]


def rand_range(lo, hi):
    return lo + r(hi - lo)


def gen_rand():
    name = ""
    label = []
    for i in range(3):
        label.append(rand_range(10, 35))
    for i in range(4):
        label.append(rand_range(0, 10))

    for i in range(7):
        name += chars[label[i]]
    return name, label

def recalculate_bbox(bbox, new_size, prev_size):
    x, y, w, h = bbox
    nw, nh = new_size
    pw, ph = prev_size
    rx, ry = (nw/pw), (nh/ph)

    bbox = (x*rx),(y*ry),(w*rx),(h*ry)
    bbox = tuple([round(x) for x in bbox])
    return bbox

def gen_sample(genplate_advanced, width, height):
    name, label = gen_rand()
    img, bbox = genplate_advanced.generate_detect(name)

    prev_h, prev_w, _ = img.shape
    img = cv2.resize(img, (width, height))

    bbox = recalculate_bbox(bbox, (width, height), (prev_w, prev_h))

    # # x,y,w,h = bbox
    # # cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    # cv2.imshow("tmp", img)
    # cv2.waitKey(1)
    # time.sleep(3)

    return label, name, img, bbox


def genBatch(batchSize, outputPath, annotationPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    label_store = []
    with open(annotationPath, 'w', newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(["filename","label","x","y","w","h"])
        for i in range(batchSize):
            print('create num:' + str(i))
            label, name, img, bbox = gen_sample(genplate_advanced, 300, 300)
            label_store.append(label)
            filename = str(i).zfill(4) + ".jpg"
            filepath = os.path.join(outputPath, filename)

            # cv2.imshow("tmp", img)
            # cv2.waitKey(1)
            # time.sleep(1)
            x,y,w,h = bbox
            cv2.imwrite(filepath, img)
            csv_out.writerow((filename, name, x, y, w, h))


font_en = './font/LicensePlate.ttf'
bg_dir = './backgrounds/SUN2012pascalformat/JPEGImages'
genplate_advanced = GenPlate(font_en, bg_dir, scale=2)

batchSize = 50000
path = './data/train_data'
annotations_path = './data/train_data/labels.csv'
genBatch(batchSize=batchSize, outputPath=path, annotationPath=annotations_path)

batchSize = 10000
path = './data/val_data'
annotations_path = './data/val_data/labels.csv'
genBatch(batchSize=batchSize, outputPath=path, annotationPath=annotations_path)
