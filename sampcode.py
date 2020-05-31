#https://github.com/lakshitadodeja/image_watermarking/blob/master/FM.py
#source code base was taken from the above repositry and modified 
#the main focus here is the comparative study done
import numpy as np
import cv2
import pywt
from PIL import Image, ImageFilter
import random
import math
import cmath


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def applyWatermarkDFT(imageMatrix, watermarkMatrix, alpha):
    shiftedDFT = np.fft.fftshift(np.fft.fft2(imageMatrix))
    watermarkedDFT = shiftedDFT + alpha * watermarkMatrix
    watermarkedImage = np.fft.ifft2(np.fft.ifftshift(watermarkedDFT))

    return watermarkedImage


def DFT(host, watermark):
    host = cv2.resize(host, (300, 300))
    cv2.imshow('Cover Image', host)
    watermark = cv2.resize(watermark, (300, 300))
    cv2.imshow('Watermark Image', watermark)

    watermarkedImage = applyWatermarkDFT(host, watermark, 10)
    watermarkedImage = np.uint8(watermarkedImage)
    cv2.imshow('Watermarked Image', watermarkedImage)


def DWT(host, watermark):
    host = cv2.resize(host, (300, 300))
    cv2.imshow('Cover Image', host)
    watermark = cv2.resize(watermark, (150, 150))
    cv2.imshow('Watermark Image', watermark)

    # DWT on cover image
    host = np.float32(host)
    host /= 255;
    coeffC = pywt.dwt2(host, 'haar')
    cA, (cH, cV, cD) = coeffC

    watermark = np.float32(watermark)
    watermark /= 255;

    # Embedding
    coeffW = (0.4 * cA + 0.1 * watermark, (cH, cV, cD))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage)


    # Extraction
    coeffWM = pywt.dwt2(watermarkedImage, 'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA - 0.4 * cA) / 0.1
    extracted *= 255
    extracted = np.uint8(extracted)
    cv2.imshow('Extracted', extracted)
    print(mse(watermark,extracted))


def DCT(host, watermark):
    host = cv2.resize(host, (512, 512))
    cv2.imshow('Cover Image', host)
    watermark = cv2.resize(watermark, (64, 64))
    cv2.imshow('Watermark Image', watermark)

    host = np.float32(host)
    host /= 255;
    coeffC = pywt.dwt2(host, 'haar')
    cA, (cH, cV, cD) = coeffC
    watermark = np.float32(watermark)
    watermark /= 255

    blockSize = 8
    c1 = np.size(host, 0)
    c2 = np.size(host, 1)
    max_message =int( (c1 * c2) / (blockSize * blockSize))

    w1 = np.size(watermark, 0)
    w2 = np.size(watermark, 1)

    watermark = np.round(np.reshape(watermark, (w1 * w2, 1)), 0)

    if w1 * w2 > max_message:
        print("large")

    message_pad = np.ones((int(max_message), 1), np.float32)
    message_pad[0:w1 * w2] = watermark

    watermarkedImage = np.ones((c1, c2), np.float32)

    k = 50
    a = 0
    b = 0

    for kk in range(max_message):
        dct_block = cv2.dct(host[b:b + blockSize, a:a + blockSize])
        if message_pad[kk] == 0:
            if dct_block[4, 1] < dct_block[3, 2]:
                temp = dct_block[3, 2]
                dct_block[3, 2] = dct_block[4, 1]
                dct_block[4, 1] = temp
        else:
            if dct_block[4, 1] >= dct_block[3, 2]:
                temp = dct_block[3, 2]
                dct_block[3, 2] = dct_block[4, 1]
                dct_block[4, 1] = temp

        if dct_block[4, 1] > dct_block[3, 2]:
            if dct_block[4, 1] - dct_block[3, 2] < k:
                dct_block[4, 1] = dct_block[4, 1] + k / 2
                dct_block[3, 2] = dct_block[3, 2] - k / 2
        else:
            if dct_block[3, 2] - dct_block[4, 1] < k:
                dct_block[3, 2] = dct_block[3, 2] + k / 2
                dct_block[4, 1] = dct_block[4, 1] - k / 2

        watermarkedImage[b:b + blockSize, a:a + blockSize] = cv2.idct(dct_block)
        if a + blockSize >= c1 - 1:
            a = 0
            b = b + blockSize
        else:
            a = a + blockSize

    watermarkedImage_8 = np.uint8(watermarkedImage)
    cv2.imshow('watermarked', watermarkedImage_8)
    coeffWM = pywt.dwt2(watermarkedImage_8,'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA - 0.4 * cA) / 0.1
    extracted *= 255
    extracted = np.uint8(extracted)
    cv2.imshow('Extracted', extracted)
   #print(mse(watermark,extracted))



    [m, n] = np.shape(host)
    host = np.double(host)
    cv2.imshow('Watermark Image', watermark)
    watermark = np.double(watermark)

    # SVD of cover image
    ucvr, wcvr, vtcvr = np.linalg.svd(host, full_matrices=1, compute_uv=1)
    Wcvr = np.zeros((m, n), np.uint8)
    Wcvr[:m, :n] = np.diag(wcvr)
    Wcvr = np.double(Wcvr)
    [x, y] = np.shape(watermark)

    # modifying diagonal component
    for i in range(0, x):
        for j in range(0, y):
            Wcvr[i, j] = (Wcvr[i, j] + 0.01 * watermark[i, j]) / 255

    # SVD of wcvr
    u, w, v = np.linalg.svd(Wcvr, full_matrices=1, compute_uv=1)

    # Watermarked Image
    S = np.zeros((225, 225), np.uint8)
    S[:m, :n] = np.diag(w)
    S = np.double(S)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr))
    wimg = np.double(wimg)
    wimg *= 255
    watermarkedImage = np.zeros(wimg.shape, np.double)
    normalized = cv2.normalize(wimg, watermarkedImage, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow('Watermarked Image', watermarkedImage)


def SVD(host, watermark):
    cv2.imshow('Cover Image', host)
    [m, n] = np.shape(host)
    host = np.double(host)
    cv2.imshow('Watermark Image', watermark)
    watermark = np.double(watermark)

    # SVD of cover image
    ucvr, wcvr, vtcvr = np.linalg.svd(host, full_matrices=1, compute_uv=1)
    Wcvr = np.zeros((m, n), np.uint8)
    Wcvr[:m, :n] = np.diag(wcvr)
    Wcvr = np.double(Wcvr)
    [x, y] = np.shape(watermark)

    # modifying diagonal component
    for i in range(0, x):
        for j in range(0, y):
            Wcvr[i, j] = (Wcvr[i, j] + 0.01 * watermark[i, j]) / 255

    # SVD of wcvr
    u, w, v = np.linalg.svd(Wcvr, full_matrices=1, compute_uv=1)

    # Watermarked Image
    S = np.zeros((225, 225), np.uint8)
    S[:m, :n] = np.diag(w)
    S = np.double(S)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr))
    wimg = np.double(wimg)
    wimg *= 255
    watermarkedImage = np.zeros(wimg.shape, np.double)
    normalized = cv2.normalize(wimg, watermarkedImage, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow('Watermarked Image', watermarkedImage)

def DWT_SVD(host, watermark):
    cv2.imshow('Cover Image', host)
    [m, n] = np.shape(host)
    host = np.double(host)
    cv2.imshow('Watermark Image', watermark)
    watermark = np.double(watermark)

    # Applying DWT on cover image and getting four sub-bands
    host = np.float32(host)
    host /= 255;
    coeffC = pywt.dwt2(host, 'haar')
    cA, (cH, cV, cD) = coeffC

    # SVD on cA
    uA, wA, vA = np.linalg.svd(cA, full_matrices=1, compute_uv=1)
    [a1, a2] = np.shape(cA)
    WA = np.zeros((a1, a2), np.uint8)
    WA[:a1, :a2] = np.diag(wA)

    # SVD on cH
    uH, wH, vH = np.linalg.svd(cH, full_matrices=1, compute_uv=1)
    [h1, h2] = np.shape(cH)
    WH = np.zeros((h1, h2), np.uint8)
    WH[:h1, :h2] = np.diag(wH)

    # SVD on cV
    uV, wV, vV = np.linalg.svd(cV, full_matrices=1, compute_uv=1)
    [v1, v2] = np.shape(cV)
    WV = np.zeros((v1, v2), np.uint8)
    WV[:v1, :v2] = np.diag(wV)

    # SVD on cD
    uD, wD, vD = np.linalg.svd(cD, full_matrices=1, compute_uv=1)
    [d1, d2] = np.shape(cV)
    WD = np.zeros((d1, d2), np.uint8)
    WD[:d1, :d2] = np.diag(wD)

    # SVD on watermarked image
    uw, ww, vw = np.linalg.svd(watermark, full_matrices=1, compute_uv=1)
    [x, y] = np.shape(watermark)
    WW = np.zeros((x, y), np.uint8)
    WW[:x, :y] = np.diag(ww)

    # Embedding Process
    for i in range(0, 255):
        for j in range(0, 255):
            WA[i, j] = WA[i, j] + 0.01 * WW[i, j]

    for i in range(0, 255):
        for j in range(0, 255):
            WV[i, j] = WV[i, j] + 0.01 * WW[i, j]

    for i in range(0, 255):
        for j in range(0, 255):
            WH[i, j] = WH[i, j] + 0.01 * WW[i, j]

    for i in range(0, 255):
        for j in range(0, 255):
            WD[i, j] = WD[i, j] + 0.01 * WW[i, j]

    # Inverse of SVD
    cAnew = np.dot(uA, (np.dot(WA, vA)))
    cHnew = np.dot(uH, (np.dot(WH, vH)))
    cVnew = np.dot(uV, (np.dot(WV, vA)))
    cDnew = np.dot(uD, (np.dot(WD, vD)))

    coeff = cAnew, (cHnew, cVnew, cDnew)

    # Inverse DWT to get watermarked image
    watermarkedImage = pywt.idwt2(coeff, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage)


def DWT_DCT_SVD(host, watermark):
    host = cv2.resize(host, (512, 512))
    cv2.imshow('Cover Image', host)
    watermark = cv2.resize(watermark, (256, 256))
    cv2.imshow('Watermark Image', watermark)
    host = np.float32(host)
    host /= 255
    coeff = pywt.dwt2(host, 'haar')
    cA, (cH, cV, cD) = coeff
    watermark = np.float32(watermark)
    watermark_dct = cv2.dct(watermark)

    cA_dct = cv2.dct(cA)

    ua, sa, va = np.linalg.svd(cA_dct, full_matrices=1, compute_uv=1)
    uw, sw, vw = np.linalg.svd(watermark, full_matrices=1, compute_uv=1)

    # Embedding
    alpha = 10
    sA = np.zeros((256, 256), np.uint8)
    sA[:256, :256] = np.diag(sa)
    sW = np.zeros((256, 256), np.uint8)
    sW[:256, :256] = np.diag(sW)
    W = sA + alpha * sW

    u1, w1, v1 = np.linalg.svd(W, full_matrices=1, compute_uv=1)
    ww = np.zeros((256, 256), np.uint8)
    ww[:256, :256] = np.diag(w1)
    Wmodi = np.matmul(ua, np.matmul(ww, va))

    widct = cv2.idct(Wmodi)
    watermarkedImage = pywt.idwt2((widct, (cH, cV, cD)), 'haar')
    cv2.imshow('watermarkedImage', watermarkedImage)


host = cv2.imread('photographer.jpg', 0)
watermark = cv2.imread('lock.jpg', 0)
DWT(host, watermark)
cv2.waitKey(0)
cv2.destroyAllWindows()
