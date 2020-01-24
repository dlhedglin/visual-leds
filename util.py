import numpy as np
from scipy.signal import butter, lfilter, freqz, sosfilt
import colorsys
from numba import jit, prange, njit

global gammaTable
gammaTable = GammaCorreciton = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
222,224,227,229,231,233,235,237,239,241,244,246,248,250,252,255]).astype(int)

def calc_fft(sig, rate):
    n = len(sig)
    freq = np.fft.rfftfreq(n, d = 1/rate)
    mags = abs(np.fft.rfft(sig)/n)
    return (mags, freq)
def getHzOfFftBand(bandIndex, fftSize=1024, rate=48000):
    freqs = np.fft.rfftfreq(1024, d = 1/rate)

def getCenterFreqs(center=1000):
    freqs = np.full(60, np.nan)
    freqs[34] = 1000
    for i in range(33, -1, -1):
        freqs[i] = freqs[i+1]/(2**(1/6))
    for i in range(35, len(freqs)):
        freqs[i] = freqs[i-1] * (2**(1/6))
    return freqs

def getBoundingFreqs(centerBands):
    G = 2
    factor = G ** (1/12)
    lowerCutoffFrequency_Hz=centerBands/factor
    upperCutoffFrequency_Hz=centerBands*factor
    return lowerCutoffFrequency_Hz, upperCutoffFrequency_Hz

def getOctiveBands(fft_data, freqs):
    centerBands = getCenterFreqs()
    lowerBounds, upperBounds = getBoundingFreqs(centerBands)
    bins = []
    for i in range(31):
        bins.append([])
    for i in range(len(freqs)):
        binLocation = np.where((freqs[i] > lowerBounds) & (freqs[i] < upperBounds))
        if binLocation[0].size != 0:
            bins[binLocation[0][0]].append(fft_data[i])
    for i in range(len(bins)):
        bins[i] = np.sum(bins[i])
    for i in range(7):
        bins[i] = fft_data[i]
    return np.asarray(bins)


def getCorrectedColor(rgbValue, GammaCorreciton=gammaTable):
    return GammaCorreciton[rgbValue]
@jit
def getCorrectedRGB(RGBArray, GammaCorreciton = gammaTable):
    return np.asarray([GammaCorreciton[RGBArray[0]], GammaCorreciton[RGBArray[1]], GammaCorreciton[RGBArray[2]]])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                        "{0:x}".format(v) for v in RGB])
def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
      colors in RGB and hex form for use in a graphing function
      defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
            "r":[RGB[0] for RGB in gradient],
            "g":[RGB[1] for RGB in gradient],
            "b":[RGB[2] for RGB in gradient]}
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
      two hex colors. start_hex and finish_hex
      should be the full six-digit color string,
      inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def hsv2rgb(h,s,v):
    return list(int(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

@njit(parallel=True)
def setStripColors(pixels, octaveBands, scale, cols):
    for i in prange(len(octaveBands)):
        brightness = octaveBands[i] * scale
        if brightness > 1:
            brightness = 1
        color = (cols[i] * brightness).astype(int)
        color = getCorrectedRGB(color)
        pixels[i] = color
        pixels[119-i] = color
        pixels[i + 120] = color
        pixels[240-i] = color
        pixels[i + 240] = color
@njit(parallel=True)     
def parOctbands(data, filterBank, octaveBands):
    for i in prange(len(octaveBands)):
        y = sosfilt(filterBank[i], data)
        y = np.sqrt(np.mean(np.power(y, 2)))
        if y > octaveBands[i]:
            octaveBands[i] = y