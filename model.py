import torch
import torch.nn as nn
import torch.nn.functional as F

'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''
# 流れ
# 1.n個のポイントを入力
# 2.特徴変換の適用
# 3.マックスプーリングで特徴を集約


#PointCloudAEクラス
class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        #モデルのパラメータを初期化
        self.latent_size = latent_size
        self.point_size = point_size
        
        #エンコーダ部分の定義
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  #3次元から64次元への畳み込み.
        #(入力のチャネル数，出力のチャネル数，カーネル数)
        #"3"は入力データのチャンネル数（入力データの特徴量数）を表す．3次元データ（x、y、z座標）が入力されることを意味する．
        #"64"は出力のチャンネル数を表す．つまり、この畳み込み層は64個の異なる特徴量を抽出する．これにより、モデルは64次元の表現を学習できる．
        #"1"はカーネルサイズを指定する．1D畳み込みでは、窓のサイズが1つの要素のみを対象とすることを示す．したがって、畳み込みは隣接する1つの要素に適用され、その値を計算する．
        #この畳み込み層は、3次元の入力データを受け取り、それを64次元の出力データに変換する役割を持つ．
        #畳み込み層はフィルターを使用して入力データの特徴を学習する．

        self.conv2 = torch.nn.Conv1d(64, 128, 1)  #64次元から128次元への畳み込み
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)  #128次元からlatent_size次元への畳み込み．128次元のベクトルがエンコードされ、その後デコーダー部分で再構築される．

        #バッチ正規化．バッチ正規化は、ニューラルネットワークの各層において、データを平均0、分散1の分布にスケール変換すること.
        # ニューラルネットワークの訓練プロセスを安定化させ、収束速度を向上させるための手法である．また、過学習を減少させるのにも役立つ．
        #バッチ正規化は、各バッチ内で平均と分散を計算し、データを正規化する．これにより、モデルの学習が安定化し、訓練プロセスの収束が向上する．
        self.bn1 = nn.BatchNorm1d(64)  #バッチ正規化
        self.bn2 = nn.BatchNorm1d(128)  #バッチ正規化
        self.bn3 = nn.BatchNorm1d(self.latent_size)  #バッチ正規化
        
        
        #デコーダ部分の定義
        self.dec1 = nn.Linear(self.latent_size,256)  #latent_sizeから256次元への全結合層
        self.dec2 = nn.Linear(256,256)  #256次元から256次元への全結合層
        self.dec3 = nn.Linear(256,self.point_size*3)  #256次元からpoint_size*3次元(入力と同じ次元)への全結合層

    #エンコーダの定義
    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))  #畳み込み>バッチ正規化>ReLU活性化．特徴の抽出と，ReLU関数による非線形性の導入．
        x = F.relu(self.bn2(self.conv2(x)))  #畳み込み>バッチ正規化>ReLU活性化
        x = self.bn3(self.conv3(x))  #畳み込み>バッチ正規化
        x = torch.max(x, 2, keepdim=True)[0]  #Maxpoolingを適用>PointNetの重要な部分!!各点の特徴量について要素毎に最大値を求め，点群全体の特徴量とする．
        x = x.view(-1, self.latent_size)  #テンソルの形状を確認
        return x
    
    #デコーダの定義
    def decoder(self, x):
        x = F.relu(self.dec1(x))  #全結合層>ReLu活性化
        x = F.relu(self.dec2(x))  #全結合層>ReLu活性化
        x = self.dec3(x)  #全結合層
        return x.view(-1, self.point_size, 3)  #テンソルの形状を変更
    
    #機械学習モデルの順伝播
    def forward(self, x):
        x = self.encoder(x)  #エンコーダを適用
        x = self.decoder(x)  #デコーダを適用
        return x
    