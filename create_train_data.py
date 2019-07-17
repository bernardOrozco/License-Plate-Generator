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


def r(val):
    return int(np.random.random() * val)


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


def gen_sample(genplate_advanced, width, height):
    name, label = gen_rand()
    img = genplate_advanced.generate(name)
    img = cv2.resize(img, (width, height))
    # img = np.multiply(img, 1 / 255.0)
    # img = img.transpose(2, 0, 1)
    return label, name, img


def genBatch(batchSize, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    label_store = []
    with open('labels.csv','w', newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(["filename","label"])
        for i in range(batchSize):
            print('create num:' + str(i))
            label, name, img = gen_sample(genplate_advanced, 320, 157)
            label_store.append(label)
            filename = str(i).zfill(4) + ".jpg"
            filepath = os.path.join(outputPath, filename)
            # print(filename, name)
            # filepath = os.path.join(outputPath, label + ".jpg")
            # filepath = outputPath + '/' + str(label) + ".jpg"
            # print(filepath)
            # cv2.imshow("tmp", img)
            # cv2.waitKey(1)
            # time.sleep(1)
            cv2.imwrite(filepath, img)
            csv_out.writerow((filename, name))
    # label_store = pd.DataFrame(label_store)
    np.savetxt('label.txt', label_store)
    # label_store.to_csv('label.txt')


batchSize = 10000
path = './data/val_data'
font_en = './font/LicensePlate.ttf'
bg_dir = './NoPlates'
genplate_advanced = GenPlate(font_en, bg_dir)
genBatch(batchSize=batchSize, outputPath=path)
