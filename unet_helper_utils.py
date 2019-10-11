import cv2
import numpy as np
from sklearn.cluster import KMeans


IMG_SIZE = 400 + 200
BATCH_SIZE = 2


def load_image(path):
    img = cv2.imread(path)
    img = img / 255.
    return img

def coords_to_square(coords, shape):
    new = []
    w, h = shape[:2]
    for x, y in coords:
        if h > w:
            y = int(np.round(y * IMG_SIZE / h))
            x = x + (h - w) / 2
            x = int(np.round(x * IMG_SIZE / h))
        else:
            x = int(np.round(x * IMG_SIZE / w))
            y = y + (w - h) / 2
            y = int(np.round(y * IMG_SIZE / w))
        new.append([x, y])
    return np.array(new)

def to_square(img, img_size=IMG_SIZE):
    three_d = len(img.shape) == 3
    if three_d:
        w, h, c = img.shape
    else:
        w, h = img.shape
        c = 1
    if w > h:
        h = int(h * img_size / w)
        w = img_size
    else:
        w = int(w * img_size / h)
        h = img_size
    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST).reshape([w, h, c])
    margin_w = (img_size - w) // 2
    margin_h = (img_size - h) // 2
    new_img = np.zeros((img_size, img_size, c))
    new_img[margin_w: margin_w + w, margin_h: margin_h + h, :] = img
    if not three_d:
        new_img = new_img.reshape([img_size, img_size])
    return new_img.astype('float32')

def unsquare(img, width, height, coords=None):
    if coords is None:
        if width > height:
            w = IMG_SIZE
            h = int(height * IMG_SIZE / width)
        else:
            h = IMG_SIZE
            w = int(width * IMG_SIZE / height)
        margin_w = (IMG_SIZE - w) // 2
        margin_h = (IMG_SIZE - h) // 2
        img = img[margin_w: margin_w + w, margin_h: margin_h + h]
        img = cv2.resize(img, (height, width))
    else:
        [x1, y1], [x2, y2] = coords
        [sx1, sy1], [sx2, sy2] = coords_to_square(coords, [width, height])
        img = cv2.resize(img[sx1: sx2, sy1: sy2], (y2 - y1, x2 - x1))
    return img

def get_mask(img, labels):
    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
    if isinstance(labels, str):
        labels = np.array(labels.split(' ')).reshape(-1, 5)
        for char, x, y, w, h in labels:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x + w >= img.shape[1] or y + h >= img.shape[0]:
                continue
            mask[y: y + h, x: x + w, 0] = 1
            radius = 6
            mask[y + h // 2 - radius: y + h // 2 + radius + 1, x + w // 2 - radius: x + w // 2 + radius + 1, 1] = 1
    return mask

def preprocess(img, width, height):
    skip = 8
    if width > height:
        w = IMG_SIZE
        h = int(height * IMG_SIZE / width)
    else:
        h = IMG_SIZE
        w = int(width * IMG_SIZE / height)
    margin_w = (IMG_SIZE - w) // 2
    margin_h = (IMG_SIZE - h) // 2
    sl_x = slice(margin_w, margin_w + w)
    sl_y = slice(margin_h, margin_h + h)
    stat = img[margin_w:margin_w + w:skip, margin_h:margin_h + h:skip].reshape([-1, 3])
    img[sl_x, sl_y] = img[sl_x, sl_y] - np.median(stat, 0)
    img[sl_x, sl_y] = img[sl_x, sl_y] / np.std(stat, 0)
    return img


def get_centers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cy = M['m10'] / M['m00']
            cx = M['m01'] / M['m00']
        else:
            cy, cx = cnt[0][0]
        cx = int(np.round(cx))
        cy = int(np.round(cy))
        centers.append([cx, cy])
    centers = np.array(centers)
    return centers

def get_labels(centers, shape):
    if len(centers) == 0:
        return
    kmeans = KMeans(len(centers), init=centers)
    kmeans.fit(centers)
    coords = []
    mlt = 2
    for i in range(0, shape[0], mlt):
        coords.append([])
        for j in range(0, shape[1], mlt):
            coords[-1].append([i, j])
    coords = np.array(coords).reshape([-1, 2])
    preds = kmeans.predict(coords)
    preds = preds.reshape([shape[0] // mlt, shape[1] // mlt])
    labels = np.zeros(shape, dtype='int')
    for k in range(mlt):
        labels[k::mlt, k::mlt] = preds
    return labels

def get_voronoi(centers, mask):
    labels = get_labels(centers, mask.shape)
    colors = np.random.uniform(0, 1, size=[len(centers), 3])
    voronoi = colors[labels]
    voronoi *= mask[:, :, None]
    return voronoi

def get_rectangles(centers, mask):
    mask_sq = to_square(mask)
    centers_sq = coords_to_square(centers, mask.shape)
    labels_sq = get_labels(centers_sq, mask_sq.shape)
    rects = [None for _ in centers]
    valid_centers = []
    for i, (xc, yc) in enumerate(centers):
        msk = (labels_sq == i).astype('float') * mask_sq / mask_sq.max()
        # crop msk
        max_size = 400  + 200
        x1 = max(0, int(np.round(xc - max_size // 2)))
        y1 = max(0, int(np.round(yc - max_size // 2)))
        x2 = min(mask.shape[0], int(np.round(xc + max_size // 2)))
        y2 = min(mask.shape[1], int(np.round(yc + max_size // 2)))
        msk = unsquare(msk, mask.shape[0], mask.shape[1], coords=[[x1,y1], [x2, y2]])
        msk = cv2.inRange(msk, 0.5, 10000)
        contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            y, x, h, w = cv2.boundingRect(cnt)
            x += x1
            y += y1
            if xc >= x and xc <= x + w and yc >= y and yc <= y + h:
                rects[i] = [x, y, w, h]
                if cv2.contourArea(cnt) <= h * w * 0.66:
                    rad_x = min(xc - x, x + w - xc)
                    rad_y = min(yc - y, y + h - yc)
                    rects[i] = [int(np.round(xc - rad_x)), y, int(np.round(2 * rad_x)), h]
                break
        if rects[i] is not None:
            valid_centers.append([xc, yc])
    return np.array([r for r in rects if r is not None]), np.array(valid_centers)

def draw_rectangles(img, rects, centers, fill_rect=[1, 0, 0], fill_cent=[1, 0, 0], fill_all=False):
    new = np.array(img)
    for x, y, w, h in rects:
        for shift in range(4):
            try:
                if fill_all:
                    new[x: x + w, y: y + h] = fill_rect
                else:
                    new[x: x + w, y + shift] = fill_rect
                    new[x: x + w, y + h - shift] = fill_rect
                    new[x + shift, y: y + h] = fill_rect
                    new[x + w - shift, y: y + h] = fill_rect
            except:
                pass
    for x, y in centers:
        r = 15
        new[x - r: x + r, y - r: y + r] = fill_cent
    return new


def add_skipped(mask, boxes, centers):
    avg_w = np.mean([b[2] for b in boxes])
    avg_area = np.mean([b[2] * b[3] for b in boxes])
    new_centers, new_boxes = [], []
    mask_c = draw_rectangles(mask, boxes, [], 0, fill_all=True)
    contours, _ = cv2.findContours(mask_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        y, x, h, w = cv2.boundingRect(cnt)
        found = False
        for xc, yc in centers:
            if xc >= x and xc <= x + w and yc >= y and yc <= y + h:
                found = True
                break
        if not found and (w * h > avg_area * 0.66 or w > avg_w * 1.5):
            new_centers.append([x + w // 2, y + h // 2])
            new_boxes.append([x, y, w, h])
    if len(new_centers) > 0:
        boxes = np.concatenate([boxes, new_boxes], 0)
        centers = np.concatenate([centers, new_centers], 0)
    return boxes, centers
