#coding=utf-8
import os
import argparse
from math import *
import numpy as np
import cv2
# import PIL
from PIL import Image, ImageFont, ImageDraw
from PlateCommon import *
import time

class GenPlate:
    def __init__(self, fontEng, NoPlates, scale=1):
        self.scale = scale
        self.fontE = ImageFont.truetype(fontEng, 35*self.scale, 0)
        self.img = np.array(Image.new("RGB", (160*self.scale, 70*self.scale), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread("./images/nl_template.bmp"), (160*self.scale, 70*self.scale))
        self.smu = cv2.imread("./images/smu2.jpg")
        self.padding = 10
        self.resize_opt = [.2, .3, .4, .5, .6, .7, .8, .9]
        self.noplates_path = []
        for parent, parent_folder, filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent + "/" + filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset = 5 * self.scale
        for i in range(3):
            base = offset + i * 9 * self.scale + i * 7 * self.scale
            self.img[0: 70*self.scale, base: base + 23*self.scale] = GenCh(self.fontE, val[i], self.scale)
        base = offset + 3 * 9 * self.scale + 3 * 7 * self.scale
        self.img[0: 70*self.scale, base: base + 23*self.scale] = GenCh(self.fontE, "-", self.scale)
        for i in range(4,6):
            base = offset + i * 9 * self.scale + i * 7 * self.scale
            self.img[0: 70*self.scale, base: base + 23*self.scale] = GenCh(self.fontE, val[i-1], self.scale)
        base = offset + 6 * 9 * self.scale + 6 * 7 * self.scale
        self.img[0: 70*self.scale, base: base + 23*self.scale] = GenCh(self.fontE, "-", self.scale)
        for i in range(7,9):
            base = offset + i * 9 * self.scale + i * 7 * self.scale
            self.img[0: 70*self.scale, base: base + 23*self.scale] = GenCh(self.fontE, val[i-2], self.scale)

        return self.img
    
    def gen_random_position(self, outer, inner, bbox):
        x1,y1,w,h = bbox
        ymax, xmax, _ = outer
        ymin, xmin, _ = inner

        x = r(xmax-xmin-1)
        y = r(ymax-ymin-1)

        x1 += x
        y1 += y

        return ((x, y),(x1,y1,w,h))

    def move_com(self, com, size, position):
        w,h=size
        blank_image = np.zeros((h,w,3), np.uint8)
        row, col, _ = com.shape
        x, y = position
        blank_image[y:(y+row), x:(x+col)] = com

        return blank_image

    def generate(self, text):
        if len(text) == 7:
            # fg = self.draw(text.decode(encoding="utf-8"))
            fg = self.draw(text)
            
            # # white letters
            # fg = cv2.bitwise_not(fg)
            # com = cv2.bitwise_or(fg, self.bg)

            # black letters
            fg = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
            _, fg = cv2.threshold(fg, 10, 255, cv2.THRESH_BINARY)
            com = cv2.bitwise_and(self.bg,self.bg,mask = fg)

            # com = rot(com,r(60)-30,com.shape,30);
            com = rot(com, r(40) - 20, com.shape, 20)
            com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))
            com = AddSmudginess(com, self.smu)

            # com = tfactor(com)
            com = random_envirment(com, self.noplates_path)
            com = AddGauss(com, 1 + r(2))
            com = addNoise(com)
            return com

    def generate_detect(self, text):
        if len(text) == 7:
            # fg = self.draw(text.decode(encoding="utf-8"))
            fg = self.draw(text)
            
            # # white letters
            # fg = cv2.bitwise_not(fg)
            # com = cv2.bitwise_or(fg, self.bg)

            # black letters
            fg = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
            _, fg = cv2.threshold(fg, 10, 255, cv2.THRESH_BINARY)
            com = cv2.bitwise_and(self.bg,self.bg,mask = fg)

            # add noise to clear letters
            com = AddGauss(com, 1)

            # give padding for transform not cropping
            com = cv2.copyMakeBorder(com,self.padding,self.padding,self.padding,self.padding,cv2.BORDER_CONSTANT,value=(0,0,0))

            # shear image sideways
            com, bbox = rot(com, r(40) - 20, com.shape, 20, bbox=True, padding=self.padding)

            # rotate vertical
            com, bbox = rotRandrom(com, 10, (com.shape[1], com.shape[0]), bbox=bbox)

            # random resize
            ratio = self.resize_opt[r(len(self.resize_opt))]
            com, bbox = scaleRandom(com, ratio, bbox=bbox)

            # get the bbox of transformations
            bbox = cv2.boundingRect(bbox)

            # select background
            env_bg = random_background(com, self.noplates_path, (600*self.scale, 600*self.scale))
            position, bbox = self.gen_random_position(env_bg.shape, com.shape, bbox)


            # generate new mask
            com = self.move_com(com, (600*self.scale, 600*self.scale), position)


            # combine background
            com = combine_images(env_bg, com)

            # add noise and blured image
            com = AddSmudginess(com, self.smu)
            com = AddGauss(com, 1 + r(2))
            com = addNoise(com)

            return com, bbox


    def genPlateString(self, pos, val):
        plateStr = ""
        box = [0, 0, 0, 0, 0, 0, 0]
        if(pos != -1):
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]
                elif cpos == 1:
                    plateStr += chars[41 + r(24)]
                else:
                    plateStr += chars[31 + r(34)]
        return plateStr

    def genBatch(self, batchSize, pos, charRange, outputPath, size):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        for i in range(batchSize):
            plateStr = self.genPlateString(-1, -1)
            print(plateStr)
            img = self.generate(plateStr)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, size)
            # filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr + ".jpg")
            filename = os.path.join(outputPath, str(i).zfill(5) + '_' + plateStr + ".jpg")
            cv2.imwrite(filename, img)
            print(filename, plateStr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_ch', default='./font/platech.ttf')
    parser.add_argument('--font_en', default='./font/platechar.ttf')
    parser.add_argument('--bg_dir', default='./NoPlates')
    parser.add_argument('--out_dir', default='./data/plate_train', help='output dir')
    parser.add_argument('--make_num', default=10, type=int, help='num')
    parser.add_argument('--img_w', default=120, type=int, help='num')
    parser.add_argument('--img_h', default=32, type=int, help='num')
    return parser.parse_args()


def main(args):
    G = GenPlate(args.font_ch, args.font_en, args.bg_dir)
    G.genBatch(args.make_num, 2, range(31, 65), args.out_dir, (args.img_w, args.img_h))


if __name__ == '__main__':
    main(parse_args())
