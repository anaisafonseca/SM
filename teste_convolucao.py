import numpy as np
from numpy.fft import fft2
import cv2
from scipy import signal, ndimage
from scipy.signal import fftconvolve
from skimage.feature import match_template
import time
from main import kernel

def convolve(im, omega, fft=False):
    M, N = im.shape
    A, B = omega.shape
    a, b = A // 2, B // 2
    if not fft:
        f = np.array(im, dtype=float)
        g = np.zeros_like(f, dtype=float)
        for x in range(M):
            for y in range(N):
                aux = 0.0
                for dx in range(-a, a + 1):
                    for dy in range(-b, b + 1):
                        if 0 <= x + dx < M and 0 <= y + dy < N:
                            aux += omega[a - dx, b - dy] * f[x + dx, y + dy]
                g[x, y] = aux
        return g
    else:
        im = np.pad(im, ((0, 1), (0, 1)))
        spi = np.fft.fft2(im)
        spf = np.fft.fft2(omega, s=im.shape)
        g = spi * spf
        f = np.fft.ifft2(g)
        return np.real(f)[1:, 1:]


def test_all():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ski_start = time.time()
        match_template(frame, kernel['edge detection'], pad_input=True, mode='constant', constant_values=0.)
        ski_end = time.time()

        nd_start = time.time()
        conv = ndimage.convolve(frame, kernel['edge detection'], mode='constant', cval=0)
        nd_end = time.time()

        scipy_start = time.time()
        fftconvolve(frame, kernel['edge detection'], mode='same')
        scipy_end = time.time()

        peretta_start = time.time()
        np.uint8(np.round(convolve(frame, kernel['edge detection'], fft=True)))
        peretta_end = time.time()

        sig_start = time.time()
        signal.convolve2d(frame, kernel['edge detection'], mode='same')
        sig_end = time.time()
        cv2.imshow('frame', conv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    print('Tempo de cada um dos algoritmos')
    print(f"scipy_signal: {sig_end- sig_start}")
    print(f"peretta: {peretta_end- peretta_start}")
    print(f"scipy_fft: {scipy_end- scipy_start}")
    print(f"ndimage: {nd_end- nd_start}")
    print(f"skimage: {ski_end- ski_start}")

if __name__ == '__main__':
    test_all()








