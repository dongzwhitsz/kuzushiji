import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")
from dataset import TestDataset
from utils import *
from unet import MyUNet
from sklearn.model_selection import train_test_split
import gc
from dataset import PapirusDataset
from unet import calc_loss



root = 'E:/input/'


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    gc.collect()
    model = MyUNet(n_classes=2)
    model = model.to(device)
    torch.cuda.empty_cache()
    gc.collect()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    num_epochs = 40
    df_train = pd.read_csv(root + 'kuzushiji-recognition/train.csv')
    xy_train, xy_dev = train_test_split(df_train, test_size=0.01)

    dataloaders = {'train': DataLoader(PapirusDataset(xy_train,root + 'kuzushiji-recognition/train_images/{}.jpg'),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
    #                         num_workers=1
                                      ),
                  'val': DataLoader(PapirusDataset(xy_dev,root + 'kuzushiji-recognition/train_images/{}.jpg'),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
    #                         num_workers=1
                                   )}

    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = {}
            for inputs, labels, shape in tqdm(dataloaders[phase], desc='phase: ' + phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            history[phase].append({k: np.mean(metrics[k]) for k in metrics})
            print(phase, history[phase][-1])
    torch.save(model.state_dict(), f'./unet_model_epochs_{num_epochs}')


def detecteOnTest():
    device = torch.device('cuda')
    model = MyUNet(n_classes=2)
    model.load_state_dict(torch.load('./checkpoint/unet.checkpoint'))
    model.to(device)

    pred_centers = []
    pred_boxes = []
    df = pd.read_csv(os.path.join(root, 'sample_submission.csv'))

    dataloaders = DataLoader(
        TestDataset(df,root + 'kuzushiji-recognition/test_images/{}.jpg'),
        batch_size=BATCH_SIZE,
        shuffle=False)
    for inputs, shape in tqdm(dataloaders):
        inputs = inputs.to(device)
        outputs = model(inputs)
        for i in range(len(inputs.data.cpu().numpy())):
            mask = outputs.data.cpu().numpy()[i]
            shp = shape.data.cpu().numpy()[i]

            binary0 = cv2.inRange(unsquare(mask[0], shp[0], shp[1]), 0.0, 10000)
            binary1 = cv2.inRange(unsquare(mask[1], shp[0], shp[1]), 0.0, 10000)

            centers0 = get_centers(binary1)
            rects, centers = get_rectangles(centers0, binary0)
            rects, centers = add_skipped(binary0, rects, centers)
            cnts = []
            for x, y in centers:
                cnts += ['unk', str(y), str(x)]
            cnts = None if len(cnts) == 0 else ' '.join(cnts)
            pred_centers.append(cnts)

            boxes = []
            for x, y, w, h in rects:
                boxes += ['unk', str(y), str(x), str(h), str(w)]
            boxes = None if len(boxes) == 0 else ' '.join(boxes)
            pred_boxes.append(boxes)
    test_predictions = pd.DataFrame({'image_id': df['image_id'], 'labels': pred_centers})
    test_predictions_box = pd.DataFrame({'image_id': df['image_id'], 'labels': pred_boxes})
    test_predictions.to_csv('./csv/dev_centers.csv', index=False)
    test_predictions_box.to_csv('./csv/dev_boxes.csv', index=False)


def show_result():
    df = pd.read_csv('./csv/dev_boxes.csv')
    for img_id, labels in tqdm(df.values):
        if isinstance(labels, float):
            cv2.namedWindow(img_id)
            path = os.path.join(root, 'kuzushiji-recognition', 'test_images', img_id + '.jpg')
            img = cv2.imread(path)
            cv2.imshow(img_id, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            continue
        path = os.path.join(root, 'kuzushiji-recognition', 'test_images', img_id + '.jpg')
        img = cv2.imread(path)
        s = labels.split(' ')
        num_obj = len(s) // 5

        for n in range(num_obj):
            x, y, w, h = eval(s[n * 5 + 1]), eval(s[n * 5 + 2]), eval(s[n * 5 + 3]), eval(s[n * 5 + 4])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
            cv2.rectangle(img, (x-5, y-5), (x + w+10, y + h+10), (0, 0, 255), thickness=3)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(w // 4, h // 4))
        cv2.namedWindow(img_id)
        cv2.imshow(img_id, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_result()
