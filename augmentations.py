# A 2x3 HSV image array for reference
# hsv_image = np.array([
#     # --- ROW 1 ---
#     [  # Pixel (0,0): [H, S, V]
#         [ 60, 200, 150],  # Pixel (0,1): [H, S, V]
#         [  0,   0, 255]   # Pixel (0,2): [H, S, V]
#     ],
#     # --- ROW 2 ---
#     [
#         [ 30, 180, 100],  # Pixel (1,0): [H, S, V],  # Pixel (1,1): [H, S, V]
#         [  0,   0,   0]   # Pixel (1,2): [H, S, V]
#     ]
# ])

import cv2
import numpy as np

# flipping the image mirrors the road, so the steering must mirror too
def augment_flip(img, steering):
    return cv2.flip(img, 1), -steering


def augment_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    # select V channel
    hsv[:, :, 2] *= np.random.uniform(0.4, 1.2)  # multiply V channel by number between 0.4 - 1.2
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)  # Clip over/underflow
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)    # Convert back to RGB

# zoom, then crop to original size
def augment_zoom(img):
    h, w = img.shape[:2]
    # choose how much to scale the image by
    scale = np.random.uniform(1.0, 1.3)
    # new_height, new_width
    scaled_up = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    y0 = (scaled_up.shape[0] - h) // 2
    x0 = (scaled_up.shape[1] - w) // 2
    return scaled_up[y0:y0+h, x0:x0+w]



def augment_pan(img, max_shift=50):
    h, w = img.shape[:2]
    # random pan amounts
    # np.random.randint returns random integers from low (inclusive) to high (exclusive); so maxShift+1
    dx = np.random.randint(-max_shift, max_shift + 1)
    # dy = np.random.randint(-20, 21)
    dy = 0
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h))
    return img


def augment_rotate(img,  max_angle=10):
    h, w = img.shape[:2]
    theta = np.random.uniform(-max_angle, max_angle)  # degrees
    rad = np.deg2rad(theta)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)
    # slide rotation matrix about origin*:
    # [[ cosθ, -sinθ, 0],
    #  [ sinθ,  cosθ, 0]]
    # * 0's in the last column define where the rotation point is
    # center point horizontal (cx), vertical (cy)
    cx, cy = w / 2, h / 2
    M = np.float32([
        [ cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
        [ sin_t,  cos_t, cy * (1 - cos_t) - cx * sin_t]
    ])
    img = cv2.warpAffine(img, M, (w, h))
    return img


def random_augment(img, steering):
    # np.random.rand generates a new number each if statement
    if np.random.rand() < 0.5:
        # print("flipped")
        img, steering = augment_flip(img, steering)
    if np.random.rand() < 0.5:
        # print("brightness adjusted")   
        img = augment_brightness(img)
    # if np.random.rand() < 0.5:
    #     # print("zoomed in")
    #     img = augment_zoom(img)
    # if np.random.rand() < 0.5:
    #     # print("panned")
    #     img = augment_pan(img)
    # if np.random.rand() < 0.5:
    #     # print("rotated")
    #     img = augment_rotate(img)
    return img, steering