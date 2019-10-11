import pandas as pd
import cv2
import numpy as np
import os
from word_classification import get_model
from tqdm import tqdm
import time
import tensorflow as tf
import torch
import pickle

def show(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def pad_to_square(img, img_size=120):
    h_img, w_img, c = img.shape
    if h_img > w_img:
        h = img_size
        w = int(img_size / h_img * w_img)
    else:
        w = img_size
        h = int(img_size / w_img * h_img)
    w_pad = (img_size - w) // 2
    h_pad = (img_size - h) // 2
    img_pad = np.zeros([img_size, img_size, c])
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    img_pad[h_pad: h_pad + h, w_pad: w_pad + w,:] = img
    return img_pad


def get_test_character(model_path):
    c = 0
    if not os.path.exists('./test_characters'):
        os.mkdir('./test_characters')
    dev_boxes = pd.read_csv('./csv/dev_boxes.csv')
    test_root = r'E:\input\kuzushiji-recognition\test_images'
    with open(r'E:\kuzhushiji-recognition\data_handler\int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    code_x_y = []
    model = get_model()
    model.load_weights(model_path)
    # 将检测到的图像分割出来
    for image_id, labels in tqdm(dev_boxes.values, desc='making submission', ncols=100):
        if isinstance(labels, float):
            code_x_y.append([])
            continue
        s = labels.split()
        num_obj = len(s) // 5
        path = os.path.join(test_root, image_id + '.jpg')
        img = cv2.imread(path)
        img = np.asarray(img)
        batch = []
        for i in range(num_obj):
            x, y, w, h = eval(s[i * 5 + 1]), eval(s[i * 5 + 2]), eval(s[i * 5 + 3]), eval(s[i * 5 + 4])
            if w == 0:
                w = 10
                c += 1
            elif h == 0:
                h = 10
                c += 1
            croped = img[y: y + h, x: x + w, :]
            croped = pad_to_square(croped)
            batch.append(croped)
        batch = np.asarray(batch).astype(np.float32) / 255.0
        preds = model.predict(batch, workers=0)
        preds = np.argmax(preds, axis=1)
        desc = ""
        for i in range(num_obj):
            x, y, w, h = s[i * 5 + 1], s[i * 5 + 2], s[i * 5 + 3], s[i * 5 + 4]
            code = 'U+' + int_to_word[preds[i]]
            desc += " ".join([code, x, y]) + " "
        code_x_y.append(desc)
    dev_boxes['submission'] = code_x_y
    dev_boxes.to_csv('./submission.csv', index=False)
    print(c)


def get_test_character_v2(model_path):
    c = 0
    if not os.path.exists('./test_characters'):
        os.mkdir('./test_characters')
    dev_boxes = pd.read_csv('./csv/dev_boxes.csv')
    test_root = r'E:\input\kuzushiji-recognition\test_images'
    with open(r'E:\kuzhushiji-recognition\data_handler\int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    code_x_y = []
    model = get_model()
    model.load_weights(model_path)
    # 将检测到的图像分割出来
    for image_id, labels in tqdm(dev_boxes.values, desc='making submission', ncols=100):
        if isinstance(labels, float):
            code_x_y.append(None)
            continue
        s = labels.split()
        boxes = []
        num_obj = len(s) // 5
        path = os.path.join(test_root, image_id + '.jpg')
        img = cv2.imread(path)
        img = np.asarray(img)
        batch = []
        for i in range(num_obj):
            x, y, w, h = eval(s[i * 5 + 1]), eval(s[i * 5 + 2]), eval(s[i * 5 + 3]), eval(s[i * 5 + 4])
            if w == 0 or h == 0:
                continue
            boxes.append('{} {} '.format(x, y))
            croped = img[y: y + h, x: x + w, :]
            croped = pad_to_square(croped)
            batch.append(croped)
        batch = np.asarray(batch).astype(np.float32) / 255.0
        preds = model.predict(batch, workers=0)
        preds = np.argmax(preds, axis=1)
        for i, b in enumerate(boxes):
            code = 'U+' + int_to_word[preds[i]]
            boxes[i] = code + " " + b
        desc = "".join(boxes)
        code_x_y.append(desc)
    dev_boxes['submission'] = code_x_y
    dev_boxes.to_csv('./submission.csv', index=False)
    print(c)


def show_result_rec(img, box):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(w // 4, h // 4))
    cv2.imshow('origin', img)

def get_test_character_v3(model_path):
    dev_boxes = pd.read_csv('./csv/dev_boxes.csv')
    test_root = r'E:\input\kuzushiji-recognition\test_images'
    with open('./dictionary/int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    code_x_y = []
    model = get_model()
    model.load_weights(model_path)
    # 将检测到的图像分割出来
    for image_id, labels in tqdm(dev_boxes.values, desc='making submission', ncols=100):
        if isinstance(labels, float):
            code_x_y.append(None)
            continue
        s = labels.split()
        boxes = []
        num_obj = len(s) // 5
        path = os.path.join(test_root, image_id + '.jpg')
        img = cv2.imread(path)
        img = np.asarray(img)
        batch = []
        for i in range(num_obj):
            x, y, w, h = eval(s[i * 5 + 1]), eval(s[i * 5 + 2]), eval(s[i * 5 + 3]), eval(s[i * 5 + 4])
            if w == 0 or h == 0:
                continue
            img_h, img_w, _ = img.shape
            croped = img[max(y - 10, 0): min(y -10 + h + 15, img_h), max(x - 10, 0): min(x -10 + w +15, img_w), :]
            croped = pad_to_square(croped)
            batch.append(croped)
            x = x + w // 2
            y = y + h // 2
            boxes.append('{} {} '.format(x, y))
        batch = np.asarray(batch).astype(np.float32) / 255.0
        preds = model.predict(batch, workers=0)
        preds = np.argmax(preds, axis=1)
        for i, b in enumerate(boxes):
            code = 'U+' + int_to_word[preds[i]]
            boxes[i] = code + " " + b
        desc = "".join(boxes)
        code_x_y.append(desc)
    dev_boxes['submission'] = code_x_y
    t = time.strftime('%d_%H_%M')
    dev_boxes = dev_boxes.drop(labels='labels', axis=1)
    dev_boxes.columns = ['image_id', 'labels']
    dev_boxes.to_csv(f'./submission/submission_{t}.csv', index=False)


def test_model():
    model_path = './model_saved/model_loss_0.084_acc_0.99999_lr_0.00010000000474974513.h5'
    with open(r'E:\kuzhushiji-recognition\data_handler\int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    model = get_model()
    model.load_weights(model_path)
    root = r'E:\input\classification2\3093'
    img = os.path.join(root, '100241706_00005_1_U+3093_157_121_14.jpg')
    img = cv2.imread(img)
    img = cv2.resize(img , dsize=(120, 120))
    cv2.namedWindow('img')
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    preds = np.argmax(preds, axis=1)
    la = int_to_word[preds[0]]
    print(la)


def get_test_character_prob(model_path):
    record = []
    dev_boxes = pd.read_csv('./csv/dev_boxes.csv')
    test_root = r'E:\input\kuzushiji-recognition\test_images'
    with open('./dictionary/int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    model = get_model()
    model.load_weights(model_path)
    # 将检测到的图像分割出来
    bar = tqdm(dev_boxes.values, desc='making submission', ncols=100, postfix={'boxes': 0})
    for image_id, labels in bar:
        result = dict()
        result['image_id'] = image_id
        if isinstance(labels, float):
            result['boxes'] = None
            result['labels'] = None
            record.append(result)
            continue
        s = labels.split()
        boxes = []
        num_obj = len(s) // 5
        bar.set_postfix({'boxes': num_obj})
        path = os.path.join(test_root, image_id + '.jpg')
        img = cv2.imread(path)
        img = np.asarray(img)
        batch = []
        for i in range(num_obj):
            x, y, w, h = eval(s[i * 5 + 1]), eval(s[i * 5 + 2]), eval(s[i * 5 + 3]), eval(s[i * 5 + 4])
            if w == 0 or h == 0:
                continue
            img_h, img_w, _ = img.shape
            croped = img[max(y - 10, 0): min(y - 10 + h + 15, img_h), max(x - 10, 0): min(x - 10 + w + 15, img_w), :]
            croped = pad_to_square(croped)
            batch.append(croped)
            x = x + w // 2
            y = y + h // 2
            boxes.append((x, y))
        batch = np.asarray(batch).astype(np.float32) / 255.0
        preds = model.predict(batch, workers=0)
        val, idx = torch.topk(torch.as_tensor(preds), k=10, dim=-1)
        val = val.tolist()
        idx = idx.tolist()
        result['boxes'] = boxes
        result['labels'] = {'val': val, 'idx': idx}
        record.append(result)
    t = time.strftime('%d_%H_%M')
    fname = './csv/dict_for_top10_prob_with_image_id_boxes_centers_{}.pkl'.format(t)
    with open(fname, 'wb') as f:
        pickle.dump(record, f)




if __name__ == "__main__":
    get_test_character_v3(r'C:\Users\dongzw\Desktop\dongzw\checkpoint\classification\model_flatten_epoch_10_loss_0.790_acc_0.96222.h5')
    # test_model()




