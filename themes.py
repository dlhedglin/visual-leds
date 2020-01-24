from rpi_ws281x import PixelStrip, Color
import numpy as np
import colorsys
import random
from time import sleep
from config import *
from util import *
import pyaudio
from scipy.ndimage import gaussian_filter1d

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)

def hueToRgb(hues):
    cols = []
    for i in range(300):
        cols.append(hsv2rgb(hues[i]/360, 1, 1))
    return cols
def hsv2rgb(h,s,v):
    return list(int(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

class Off:
    def __init__(self):
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
    def start(self):
        for i in range(300):
            self.strip.setPixelColor(i, Color(0,0,0))
            self.strip.show()

class Quicksort:
    def partition(self, arr, start, end):
        pivot = arr[end]
        i = start
        for j in range(start, end):
            if arr[j] <= arr[end]: # if arr[j] is less than arr[i] we swap the values
                arr[i], arr[j] = arr[j], arr[i]
                # cols = hueToRgb(arr)
                col = hsv2rgb(arr[j]/360, 1, 1)
                col2 = hsv2rgb(arr[i]/360, 1, 1)
                self.strip.setPixelColor(j, Color(col[0], col[1], col[2]))
                self.strip.show()
                self.strip.setPixelColor(i, Color(col2[0], col2[1], col2[2]))
                i = i + 1
                self.strip.show()
        arr[i],arr[end] = arr[end], arr[i]
        col = hsv2rgb(arr[i]/360, 1, 1)
        col2 = hsv2rgb(arr[end]/360, 1, 1)
        self.strip.setPixelColor(i, Color(col[0], col[1], col[2]))
        self.strip.show()
        self.strip.setPixelColor(end, Color(col2[0], col2[1], col2[2]))
        self.strip.show()
        return i
    def quicksort(self, arr, start, end):
        if start < end:
            index = self.partition(arr, start, end)
            self.quicksort(arr, start, index-1)
            self.quicksort(arr, index + 1, end)
    def __init__(self):
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
    def start(self):
        while True:
            hues = np.arange(300)
            random.shuffle(hues)
            cols = hueToRgb(hues)
            for k in range(300):
                color = Color(cols[k][0], cols[k][1], cols[k][2])
                self.strip.setPixelColor(k, color)
            self.strip.show()
            self.quicksort(hues, 0, len(hues) - 1)

class Solid:
    def __init__(self):
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
    def start(self):
        for i in range(300):
            self.strip.setPixelColor(i, Color(255,255,255))
        self.strip.show()


class Slide:
    def __init__(self):
        self.gammaTable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
        11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28,
        29, 29, 30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40,
        40, 41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89,
        90, 91, 93, 94, 95, 96, 98, 99,100,102,103,104,106,107,109,110,
        111,113,114,116,117,119,120,121,123,124,126,128,129,131,132,134,
        135,137,138,140,142,143,145,146,148,150,151,153,155,157,158,160,
        162,163,165,167,169,170,172,174,176,178,179,181,183,185,187,189,
        191,193,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
        222,224,227,229,231,233,235,237,239,241,244,246,248,250,252,255]
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()

        self.brightnesses = np.full(LED_COUNT, 0)
        self.cols = []
        self.hues = np.arange(300, step=5)
        for i in range(60):
            self.cols.append(hsv2rgb(self.hues[i]/360, 1, 1))
        self.centerFreqs = getCenterFreqs()
        self.lower, self.upper = getBoundingFreqs(self.centerFreqs)
        self.filterBank = []
        for i in range(len(self.centerFreqs)):
            self.sos = butter_bandpass(self.lower[i], self.upper[i], RATE, order=1)
            self.filterBank.append(self.sos)

        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK, input_device_index=2)
        self.octaveBands = np.full(60, 0.0)

    def start(self):
        while True:
            for col in self.cols:
                data = self.stream.read(CHUNK, False)
                data = np.frombuffer(data ,dtype=np.int16)
                data = data * np.hamming(len(data))
                # t1 = time.time()
                # print(t1-t0)
                for i in range(5):
                    y = sosfilt(self.filterBank[i], data)
                    y = np.sqrt(np.mean(np.power(y, 2)))
                    self.octaveBands[i] = y
                s = (np.mean(self.octaveBands[:5]) // 1000).astype(int)
                idx = np.random.randint(0, LED_COUNT, size = s + 1)
                self.brightnesses[idx] = 1.0
                
                for i in range(LED_COUNT):
                    b = self.brightnesses[i]       
                    color = Color(self.gammaTable[int(col[0] * b)], self.gammaTable[int(col[1] * b)], self.gammaTable[int(col[2] * b)])
                    self.strip.setPixelColor(i, color)
                self.strip.show()
                self.brightnesses[:-1] = self.brightnesses[1:]; self.brightnesses[-1] = 0

                self.brightnesses = self.brightnesses - 0.10
                self.brightnesses[self.brightnesses  < 0] = 0
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class Spectrum:
    def __init__(self):
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()

        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK, input_device_index=2)
        self.gammaTable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
        11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28,
        29, 29, 30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40,
        40, 41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89,
        90, 91, 93, 94, 95, 96, 98, 99,100,102,103,104,106,107,109,110,
        111,113,114,116,117,119,120,121,123,124,126,128,129,131,132,134,
        135,137,138,140,142,143,145,146,148,150,151,153,155,157,158,160,
        162,163,165,167,169,170,172,174,176,178,179,181,183,185,187,189,
        191,193,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
        222,224,227,229,231,233,235,237,239,241,244,246,248,250,252,255]
        self.cols = []
        self.hues = np.arange(300, step=5)
        for i in range(60):
            self.cols.append(hsv2rgb(self.hues[i]/360, 1, 1))
        self.cols = self.cols[::-1]
        self.scale = 1.0/4000
        self.decay = 0.85
        self.centerFreqs = getCenterFreqs()
        self.lower, self.upper = getBoundingFreqs(self.centerFreqs)
        self.filterBank = []
        for i in range(len(self.centerFreqs)):
            sos = butter_bandpass(self.lower[i], self.upper[i], RATE, order=1)
            self.filterBank.append(sos)
        self.octaveBands = np.full(60, 0.0)
    def start(self):
        while True:
            data = self.stream.read(CHUNK, False)
            data = np.frombuffer(data ,dtype=np.int16)
            data = data * np.hanning(len(data))
            for i in range(len(self.centerFreqs)):
                y = sosfilt(self.filterBank[i], data)
                y = np.sqrt(np.mean(np.power(y, 2)))
                if y > self.octaveBands[i]:
                    self.octaveBands[i] = y
            self.octaveBands = gaussian_filter1d(self.octaveBands, 0.5).astype(int)
            for i in range(len(self.octaveBands)):
                brightness = self.octaveBands[i] * self.scale
                if brightness > 1:
                    brightness = 1
                
                color = Color(self.gammaTable[int(self.cols[i][0] * brightness)], self.gammaTable[int(self.cols[i][1] * brightness)], self.gammaTable[int(self.cols[i][2] * brightness)])
                self.strip.setPixelColor(i, color)
                self.strip.setPixelColor(119-i, color)
                self.strip.setPixelColor(i + 120, color)
                self.strip.setPixelColor(240-i, color)
                self.strip.setPixelColor(i + 240, color)
            self.octaveBands = self.octaveBands - 500
            self.strip.show()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class Sparkle:
    def __init__(self):
        self.gammaTable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
        11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28,
        29, 29, 30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40,
        40, 41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89,
        90, 91, 93, 94, 95, 96, 98, 99,100,102,103,104,106,107,109,110,
        111,113,114,116,117,119,120,121,123,124,126,128,129,131,132,134,
        135,137,138,140,142,143,145,146,148,150,151,153,155,157,158,160,
        162,163,165,167,169,170,172,174,176,178,179,181,183,185,187,189,
        191,193,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
        222,224,227,229,231,233,235,237,239,241,244,246,248,250,252,255]
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
        self.cols = []
        self.hues = np.arange(LED_COUNT)
        for i in range(LED_COUNT):
            self.cols.append(hsv2rgb(self.hues[i]/360, 1, 1))
        self.brightnesses = np.full(LED_COUNT, 0)
    def start(self):
        while True:
            for col in self.cols:
                idx = np.random.randint(0, 1000, size = 1)
                if idx < 300:
                    self.brightnesses[idx] = 1.0
                for i in range(LED_COUNT):
                    b = self.brightnesses[i]       
                    color = Color(self.gammaTable[int(col[0] * b)], self.gammaTable[int(col[1] * b)], self.gammaTable[int(col[2] * b)])
                    self.strip.setPixelColor(i, color)
                self.strip.show()
                self.brightnesses = self.brightnesses - 0.02
                self.brightnesses[self.brightnesses  < 0] = 0

class Fade:
    def __init__(self):
        self.gammaTable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
        11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28,
        29, 29, 30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40,
        40, 41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89,
        90, 91, 93, 94, 95, 96, 98, 99,100,102,103,104,106,107,109,110,
        111,113,114,116,117,119,120,121,123,124,126,128,129,131,132,134,
        135,137,138,140,142,143,145,146,148,150,151,153,155,157,158,160,
        162,163,165,167,169,170,172,174,176,178,179,181,183,185,187,189,
        191,193,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
        222,224,227,229,231,233,235,237,239,241,244,246,248,250,252,255]
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
        self.cols = []
        self.hues = np.arange(360, step = 10)
        for i in range(36):
            self.cols.append(hsv2rgb(self.hues[i]/360, 1, 1))
    def start(self):
        while True:
            for col in self.cols:
                for i in range(100):
                    b = i * (1/100)
                    color = Color(self.gammaTable[int(col[0] * b)], self.gammaTable[int(col[1] * b)], self.gammaTable[int(col[2] * b)])
                    for j in range(LED_COUNT):
                        self.strip.setPixelColor(j, color)
                    self.strip.show()
                for i in range(100, 0, -1):
                    b = i * (1/100)
                    color = Color(self.gammaTable[int(col[0] * b)], self.gammaTable[int(col[1] * b)], self.gammaTable[int(col[2] * b)])
                    for j in range(LED_COUNT):
                        self.strip.setPixelColor(j, color)
                    self.strip.show()
            
        
        
