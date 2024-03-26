import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
import librosa
from torch.nn.functional import pad
import random
import os
from tqdm import tqdm


class AudioMNISTDataset(Dataset):
    def __init__(self, root, train=True):
        # 只取1和2的语音做二分类
        ds_1 = []
        ds_2 = []
        for speaker in os.listdir(root):
            speaker_path = os.path.join(root, speaker)
            if not os.path.isdir(speaker_path):
                continue
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                digit = file.split('_')[0]
                if digit == '1':
                    ds_1.append((file_path, digit))
                elif digit == '2':
                    ds_2.append((file_path, digit))

        # 打乱
        random.seed(0)
        random.shuffle(ds_1)
        random.shuffle(ds_2)

        # 9:1分训练集和测试集
        train_1 = ds_1[:int(len(ds_1)*0.9)]
        test_1 = ds_1[int(len(ds_1)*0.9):]
        train_2 = ds_2[:int(len(ds_2)*0.9)]
        test_2 = ds_2[int(len(ds_2)*0.9):]

        if train:
            self.audio = [x[0] for x in train_2 + train_1]
            self.labels = [x[1] for x in train_2 + train_1]
        else:
            self.audio = [x[0] for x in test_2 + test_1]
            self.labels = [x[1] for x in test_2 + test_1]
        assert(len(self.audio) == len(self.labels))
    
    def __len__(self):
        return len(self.audio)
    
    def get_data(self, file, target_sr=16000):
        data, sr = librosa.load(file)
        # 重采样，默认16000
        down_d = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        # 所有音频长度限定为12000，不足的补0，超过的截断
        fix_len_d = librosa.util.fix_length(down_d, size=12000)
        return fix_len_d, target_sr
    
    def mfcc_data(self, file):
        data,sr = self.get_data(file)
        # 使用MFCC特征, 40维
        # MFCC是FBank经过DCT变换得到的特征
        # 而FBank是音频经过mel滤波器组得到的特征
        data = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        return data
    
    def __getitem__(self, idx):
        if self.labels[idx] == '1':
            label = torch.tensor(-1)
        elif self.labels[idx] == '2':
            label = torch.tensor(1)
        
        audio_seq = self.mfcc_data(self.audio[idx])
        audio_seq = torch.tensor(audio_seq).to(dtype=torch.float32)
        return audio_seq, label
    

if __name__ == '__main__':
    # 创建训练集和测试集
    root = os.path.join('.','data','AudioMNIST','data')

    train_dataset = AudioMNISTDataset(root, train=True)
    test_dataset = AudioMNISTDataset(root, train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 实现感知器模型
    epochs = 1
    lr = 0.01
    w = torch.zeros(40 * 24)
    print(w.shape)

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        acc = 0
        cnt = 0
        flag = True
        for audio_seq, label in (t:=tqdm(train_loader)):
            audio_seq = audio_seq.view(audio_seq.shape[0], -1).squeeze(0)
            y_hat = audio_seq.dot(w)
            if y_hat * label <= 0:
                flag = False
                w += lr * label * audio_seq
            else:
                acc += 1
            cnt += 1
            t.set_description(f"Epoch {epoch}/{epochs-1}, Train Acc: {0.0 if cnt==0 else acc/cnt:.4f}")
        print(f'Accuracy on epoch {epoch}: {acc/cnt:.4f}')
        # 全部分类正确
        if flag == True:
            break

    acc = 0
    cnt = 0

    for audio_seq, label in tqdm(test_loader, desc='Testing'):
        audio_seq = audio_seq.view(audio_seq.shape[0], -1).squeeze(0)
        y_hat = audio_seq.dot(w)
        if y_hat * label > 0:
            acc += 1
        cnt += 1

    print(f'Test Acc: {acc/cnt:.4f}')