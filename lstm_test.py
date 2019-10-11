import torch
import torch.nn as nn
from torch.utils.data import Dataset
from functools import cmp_to_key
import  pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle


def my_sort(a, b):
    au, ax, ay = a
    bu, bx, by = b
    r = 1
    if abs(ax - bx) < 200:
        if ay > by:
            r = -1
        else:
            r = 1
    elif ax - bx > 200:
        r = 1
    elif ax - bx < 200:
        r = -1
    return r


def sort_csv(csv_path):
    csv = pd.read_csv(csv_path)
    s = []
    for i, (image_id, labels) in tqdm(enumerate(csv.values), ncols=100):
        if isinstance(labels, float):
            s.append(None)
            continue
        labels_split = labels.split(' ')
        num_obj = len(labels_split) // 5
        cnts = []
        for n in range(num_obj):
            code = labels_split[n * 5]
            cx = eval(labels_split[n * 5 + 1]) + eval(labels_split[n * 5 + 3]) // 2
            cy = eval(labels_split[n * 5 + 2]) + eval(labels_split[n * 5 + 4]) // 2
            cnts.append([code ,cx, cy])
        key = cmp_to_key(my_sort)
        cnts = sorted(cnts, key=key, reverse=True)
        # p = os.path.join(r'E:\input\kuzushiji-recognition\train_images', image_id + '.jpg')
        # img = cv2.imread(p)
        # for i, (u, x, y) in enumerate(cnts):
        #     img = cv2.circle(img, (x, y), radius=100, color=(0, 0, 255), thickness=6)
        #     img = cv2.putText(img, str(i), (x-40, y+40), cv2.FONT_HERSHEY_COMPLEX,3,(100,0,255), thickness=5)
        # h, w, _ = img.shape
        # img = cv2.resize(img, dsize=(w //5, h // 5))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cnts = np.asarray(cnts)
        s.append(' '.join(cnts[:, 0]))
    csv['sorted_label'] = s
    csv.to_csv('./csv/sorted_train.csv', index=False)


class MyData(Dataset):
    def __init__(self, seq_len = 120, is_training=True):
        self.is_training = is_training
        if not os.path.exists('./csv/sorted_train.csv'):
            sort_csv(r'E:\input\train.csv')
        csv = pd.read_csv('./csv/sorted_train.csv')
        self.seqs = csv.sorted_label.dropna().values
        with open('./dictionary/word_to_int.txt', 'rb') as f:
            word_to_int = eval(f.read())
        for i, s in enumerate(self.seqs):
            b = []
            for c in s.split():
                b.append(word_to_int[c.split('+')[-1]])
            self.seqs[i] = b
        self.seqs = np.concatenate(self.seqs, axis=0)
        self.seqs = self.seqs[:len(self.seqs) // seq_len * seq_len]
        self.seqs_x = [self.seqs[seq_len * i: seq_len * (i+1)] for i in range(len(self.seqs) // seq_len - 1)]
        self.seqs_y = [self.seqs[seq_len * i + 1: seq_len * (i + 1) + 1] for i in range(len(self.seqs) // seq_len - 1)]
        self.batch_num = len(self.seqs)
        train_x, valid_x, train_y, valid_y = train_test_split(self.seqs_x, self.seqs_y, test_size=0.15, random_state=666)
        if is_training:
            self.seqs_x = train_x
            self.seqs_y = train_y
        else:
            self.seqs_x = valid_x
            self.seqs_y = valid_y

    def __getitem__(self, idx):
        seq_x = self.seqs_x[idx]
        seq_y = self.seqs_y[idx]
        return torch.LongTensor(seq_x), torch.LongTensor(seq_y)

    def __len__(self):
        return len(self.seqs_x)


class MyModule(nn.Module):
    def __init__(self, embedding_dim=256, num_units=256):
        super(MyModule, self).__init__()
        self.num_units = num_units
        self.embedding = nn.Embedding(4212, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=num_units,num_layers=2, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(in_features=num_units, out_features=1024)
        self.logit = nn.Linear(in_features=num_units, out_features=4212)

    def forward(self, seqs):
        batch_size, seq_len = seqs.size()
        x = self.embedding(seqs)
        outputs, hidden = self.rnn(x)
        outputs = outputs.view(-1, seq_len, 2, self.num_units)
        x = outputs[:, :,  -1, :].contiguous()
        x = x.view(-1, self.num_units)
        # x = self.fc(x)
        x = self.logit(x)
        x = x.view(-1, seq_len, 4212)
        return x

def train():
    device = torch.device('cuda')
    data = {
        'train': torch.utils.data.DataLoader(
            MyData(is_training=True),
            shuffle=True,
            batch_size=32,
            drop_last=True
        ),
        'valid': torch.utils.data.DataLoader(
            MyData(is_training=False),
            batch_size=32,
            shuffle=True,
            drop_last=True
        )}
    model = MyModule()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    ent = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        for phase in data:
            acc = 0
            ls = 0
            bar = tqdm(data[phase], ncols=150, desc=f"epoch:{epoch}, phase:{phase}", postfix={'loss': 0, 'acc': 0})
            for step, (seq, target) in enumerate(bar):
                seq = seq.to(device)
                target = target.to(device)
                logit = model(seq)
                loss = 0
                for i in range(32):
                    loss += ent(logit[i], target[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                p = torch.argmax(logit, dim=2)
                acc += (p == target).to(torch.float32).mean().data.cpu().numpy()
                ls += loss.data.cpu().numpy()
                if step % 20 is 0:
                    bar.set_postfix({'loss': ls / 20, 'acc': acc / 20})
                    ls = 0
                    acc = 0
    torch.save(model.state_dict(), './checkpoint/rnn/bilstm.pt')

def test():
    device = torch.device('cuda')
    model = MyModule()
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoint/rnn/bilstm.pt'))
    test = torch.utils.data.DataLoader(
        MyData(is_training=False, seq_len=114),
        batch_size=40,
        drop_last=True
    )
    with torch.no_grad():
        for x, y in test:

            # x = x.to(device)
            # y = y.to(device)
            x = torch.randint(4212, size=(1, 233)).cuda()
            logit = model(x)
            val, idx = torch.topk(torch.softmax(logit, dim=2), k=1)
            p = torch.argmax(logit, dim=2)
            # acc = (p == y).to(torch.float32)
            # acc = acc.mean().data.cpu().numpy()
            print('xixi')



def check(
        fpath=r'csv/dict_for_top10_prob_with_image_id_boxes_centers_09_19_12.pkl',
        mpath = r'checkpoint/rnn/bilstm.pt'
):
    device = torch.device('cuda')
    with open(fpath, 'rb') as f:
        record = pickle.load(f)
    with open('./dictionary/int_to_word.txt', 'r') as f:
        int_to_word = f.read()
        int_to_word = eval(int_to_word)
    model = MyModule()
    model.to(device)
    model.load_state_dict(torch.load(mpath))
    model.eval()

    ### result: {}
    '''
    result:
    image_id:
    boxes:
    labes:
    '''
    print('xix')
    df = pd.DataFrame(columns=['image_id', 'labels'])
    df_id = []
    df_labels = []
    for img in record:
        image_id = img['image_id']
        df_id.append(image_id)
        boxes = img['boxes']
        if boxes is None:
            df_labels.append(boxes)
            continue

        labels = img['labels']
        val = labels['val']
        idx = labels['idx']
        num_obj = len(boxes)
        origin_seq = np.expand_dims([idx[i][0] for i in range(10)], axis=0)
        origin_tensor = torch.LongTensor(origin_seq).cuda()

        logit = model(origin_tensor)
        predict_softmax = torch.nn.functional.softmax(logit, dim=-1)
        predict_prob, predict_idx = torch.topk(predict_softmax, k=1, dim=-1)

        print('xixi')

        logit = torch.squeeze(logit)
        predict_softmax = torch.squeeze(predict_softmax)
        predict_prob, predict_idx = torch.squeeze(predict_prob), torch.squeeze(predict_idx)
        print('xixi')



if __name__ == "__main__":
    train()


