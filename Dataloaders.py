import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

#ReadDatasetクラスの定義
#データの読み込みと変換を処理する
class ReadDataset(Dataset):
    def __init__(self,  source):  #データセットのインスタンスメソッド
     
        self.data = torch.from_numpy(source).float()  #sourceでnumpy配列のデータセットを受け取り，それをPytorchのfloat型のテンソルに変換し，self.dataに格納

    def __len__(self):  #データセットのサイズを返すインスタンスメソッド
        return len(self.data)

    def __getitem__(self, index):  #指定されたインデックスのデータ数を返すインスタンスメソッド
        return self.data[index]

#RandomSplit関数の定義
def RandomSplit(datasets, train_set_percentage):  #データセットを訓練用とテスト用に分ける
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)

#データセットとデータローダーを作成するための関数 GetDataLoaders を定義
#ReadDatasetを使用して，numpy配列 npArray をPyTorchデータセットに変換
#RandomSplit 関数を使用し，訓練セットとテストセットを作成
#最後に，PyTorchの DataLoader を使用し，トレーニングセットとテストセット用のデータローダーを作成します．データのシャッフルや並列処理，バッチサイズなどの設定を行う
def GetDataLoaders(npArray, batch_size, train_set_percentage = 0.9, shuffle=True, num_workers=0, pin_memory=True):
    
    pc = ReadDataset(npArray)

    train_set, test_set = RandomSplit(pc, train_set_percentage)

    train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    
    return train_loader, test_loader