import math
import numpy as np
import cv2 as cv

class Gabor:

    def __init__(self, n_filters=9, k_size=37, sigma=4.35, lamb=8.25, psi=0.5, gamma=0.9):
        self.n_filters = n_filters
        k = int(k_size/2)
        self.x_max = k
        self.x_min = -k
        self.y_max = k
        self.y_min = -k
        self.lamb = lamb
        self.psi = psi
        self.sigma = sigma
        self.gamma = gamma
        self.filters = []

    def x_mark(self, x, y, t):
        ct = math.cos(t)
        st = math.sin(t)
        xm = (x * ct) + (y * st)
        return xm

    def y_mark(self, x, y, t):
        ct = math.cos(t)
        st = math.sin(t)
        ym = (-x * st) + (y * ct)
        return ym

    def pixel_value(self, x, y, t):
        xm = self.x_mark(x, y, t)
        ym = self.y_mark(x, y, t)
        lhu = -(xm ** 2 + (self.gamma ** 2 * ym ** 2))
        lhl = 2 * self.sigma ** 2
        lh = math.exp(lhu / lhl)
        rhi = (2 * math.pi * (xm / self.lamb)) + self.psi
        rh = math.cos(rhi)
        p = lh * rh
        return p

    def create_filters(self):
        d = 180 / self.n_filters
        for i in range(0, self.n_filters):
            t = i * d
            filt = []
            for j in range(self.x_min, self.x_max):
                row = []
                for c in range(self.y_min, self.y_max):
                    pixel = self.pixel_value(c, j, t)
                    row.append(pixel)
                filt.append(row)
            n_filter = np.array(filt)
            self.filters.append(n_filter)
        return self.filters

    def apply_filters(self, img):
        new_img = np.zeros_like(img)
        for filter1 in self.filters:
            f_img = cv.filter2D(img, cv.CV_8UC3, filter1)
            np.maximum(new_img, f_img, new_img)
        return new_img


import gabor
import cv2 as cv

# Read in the test image
img = cv.imread("15_test.tif")
img = cv.resize(img, (500, 500))

# Initialise the gabor class and create filters
g = gabor.Gabor(n_filters=20)
result = g.create_filters()


# Apply filters to image and view
g_img = g.apply_filters(img)
# cv.imshow("Phantom", g_img)

# cv.waitKey(0)
