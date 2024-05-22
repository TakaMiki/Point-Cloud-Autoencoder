import numpy as np
import torch
import model
import os

# モデルのパラメータ
latent_size = 128
point_size = 1024  # データのポイントのサイズに合わせて変更してください
model_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/SotukenB_Miki_linux/AHFE/10000data/1024points/output_10000data_1024points/final_best_model.pth"  # 保存されたモデルのパス

# GPUを使用する場合
use_GPU = True
device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")

# モデルを読み込む
net = model.PointCloudAE(point_size, latent_size)
net.load_state_dict(torch.load(model_path, map_location=device))
net = net.to(device)
net.eval()  # モデルを評価モードに設定

# 1つのデータポイントを再構築して保存する関数
def reconstruct_and_save_data(data_point, output_folder):
    # 入力データの形状を (1, 3, 4096) に変更
    data_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
    data_tensor = data_tensor.unsqueeze(0).permute(0, 2, 1)
    
    # モデルにデータを供給して再構築
    reconstructed_data = net(data_tensor)
    reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
    # NumPyファイルに保存
    output_filename = os.path.join(output_folder, "reconstructed_kakudai_10000data_1024points.npy")
    np.save(output_filename, reconstructed_data)
    
    print("データポイントの再構築が完了し、新しいNumPyファイルに保存されました。")

# 既存のNumPyファイルからデータを読み込む
existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/SotukenB_Miki_linux/AHFE/10000data/1024points/original_data/kakudai_1024points_10000data_normalize.npy"
existing_data = np.load(existing_data_path)

# 再構築と保存を実行
output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/SotukenB_Miki_linux/AHFE/10000data/1024points/original_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for data_point in existing_data:
    reconstruct_and_save_data(data_point, output_folder)



















# import numpy as np
# import torch
# import model

# # モデルのパラメータ
# latent_size = 128
# point_size = 4096  # データのポイントのサイズに合わせて変更してください
# model_path = "model.pth"  # 保存されたモデルのパス

# # GPUを使用する場合
# use_GPU = True
# device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")

# # モデルを読み込む
# net = model.PointCloudAE(point_size, latent_size)
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# net.eval()  # モデルを評価モードに設定

# # 1つのデータポイントを再構築して保存する関数
# def reconstruct_and_save_data(data_point, output_folder):
#     # 入力データの形状を (1, 3, 4096) に変更
#     data_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
#     data_tensor = data_tensor.unsqueeze(0).permute(0, 2, 1)
    
#     # モデルにデータを供給して再構築
#     reconstructed_data = net(data_tensor)
#     reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
#     # NumPyファイルに保存
#     output_filename = f"{output_folder}/reconstructed_data_slope.npy"
#     np.save(output_filename, reconstructed_data)
    
#     print("データポイントの再構築が完了し、新しいNumPyファイルに保存されました。")

# # 既存のNumPyファイルからデータを読み込む
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_slope_touitu.npy"
# existing_data = np.load(existing_data_path)

# # 再構築と保存を実行
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"
# for i, data_point in enumerate(existing_data):
#     reconstruct_and_save_data(data_point, output_folder)























# import numpy as np
# import torch
# import model

# # モデルのパラメータ
# latent_size = 128
# point_size = 4096  # データのポイントのサイズに合わせて変更してください
# model_path = "model.pth"  # 保存されたモデルのパス

# # GPUを使用する場合
# use_GPU = True
# device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")

# # モデルを読み込む
# net = model.PointCloudAE(point_size, latent_size)
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# net.eval()  # モデルを評価モードに設定

# # 1つのデータポイントを再構築して保存する関数
# def reconstruct_and_save_data(data_point, output_folder, data_index):
#     # 入力データの形状を (1, 3, 4096) に変更
#     data_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
#     data_tensor = data_tensor.unsqueeze(0).permute(0, 2, 1)
    
#     # モデルにデータを供給して再構築
#     reconstructed_data = net(data_tensor)
#     reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
#     # 1つのNumPyファイルに保存
#     # output_filename = f"{output_folder}/reconstructed_data_{data_index}.npy"
#     output_filename = f"{output_folder}/reconstructed_data_test_slope.npy"
#     np.save(output_filename, reconstructed_data)
    
#     print(f"データポイント {data_index} の再構築が完了し、新しいNumPyファイルに保存されました。")

# # 既存のNumPyファイルからデータを読み込む
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_slope_touitu.npy"
# existing_data = np.load(existing_data_path)

# # existing_dataの形状を確認する
# print(existing_data)

# # 再構築と保存を実行
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"
# for i, data_point in enumerate(existing_data):
#     reconstruct_and_save_data(data_point, output_folder, i)


# # 1つのデータポイントを再構築して保存する関数
# def reconstruct_and_save_data(data_point, output_folder, data_index):
#     # 入力データの形状を (1, 3, 4096) に変更
#     data_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
#     data_tensor = data_tensor.unsqueeze(0).permute(0, 2, 1)
    
#     # モデルにデータを供給して再構築
#     reconstructed_data = net(data_tensor)
#     reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
#     # 1つのNumPyファイルに保存
#     output_filename = f"{output_folder}/reconstructed_data_{data_index}.npy"
#     np.save(output_filename, reconstructed_data)
    
#     print(f"データポイント {data_index} の再構築が完了し、新しいNumPyファイルに保存されました。")

# # 既存のNumPyファイルからデータを読み込む
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_box_touitu.npy"
# existing_data = np.load(existing_data_path)

# # existing_dataの形状を確認する
# print(existing_data)

# # 再構築と保存を実行
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"
# reconstruct_and_save_data(existing_data, output_folder, 0)















# import numpy as np
# import torch
# import model

# # モデルのパラメータ
# latent_size = 128
# point_size = 4096  # データのポイントのサイズに合わせて変更してください
# model_path = "model.pth"  # 保存されたモデルのパス

# # GPUを使用する場合
# use_GPU = True
# device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")

# # モデルを読み込む
# net = model.PointCloudAE(point_size, latent_size)
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# net.eval()  # モデルを評価モードに設定

# # 1つのデータポイントを再構築して保存する関数
# def reconstruct_and_save_data(data_point, output_folder, data_index):
#     data_tensor = torch.tensor(data_point, dtype=torch.float32).to(device)
    
#     # 入力データの形状を (バッチサイズ, ポイント数, チャネル数) に変更
#     data_tensor = data_tensor.permute(0, 2, 1)
    
#     # モデルにデータを供給して再構築
#     reconstructed_data = net(data_tensor)
#     reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
#     # 1つのNumPyファイルに保存
#     output_filename = f"{output_folder}/reconstructed_data_{data_index}.npy"
#     np.save(output_filename, reconstructed_data)
    
#     print(f"データポイント {data_index} の再構築が完了し、新しいNumPyファイルに保存されました。")


# # 既存のNumPyファイルからデータを読み込む
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_box_touitu.npy"
# existing_data = np.load(existing_data_path)

# # existing_dataの形状を確認する
# print(existing_data)

# # 再構築と保存を実行
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"
# reconstruct_and_save_data(existing_data[0], output_folder, 0)



# import numpy as np
# import torch
# from Dataloaders import ReadDataset
# import model

# # モデルのパラメータ
# latent_size = 128
# point_size = 4096  # データのポイントのサイズに合わせて変更してください
# model_path = "model.pth"  # 保存されたモデルのパス

# # GPUを使用する場合
# use_GPU = True
# device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")import numpy as np
# import torch
# from Dataloaders import ReadDataset
# import model

# # モデルのパラメータ
# latent_size = 128
# point_size = 4096  # データのポイントのサイズに合わせて変更してください
# model_path = "model.pth"  # 保存されたモデルのパス

# # GPUを使用する場合
# use_GPU = True
# device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")

# # モデルを読み込む
# net = model.PointCloudAE(point_size, latent_size)
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# net.eval()  # モデルを評価モードに設定

# # 元のデータが格納されている既存のNumPyファイルのパスを指定
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_box_touitu.npy"  # 既存のデータファイルのパスを設定してください
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"  # 保存先フォルダのパスを設定してください

# # 既存のNumPyファイルからデータを読み込む
# existing_data = np.load(existing_data_path)

# # 前処理を行い、テンソルに変換
# validation_dataset = ReadDataset(existing_data)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

# # 新しいNumPyファイルに再構築データを累積するためのリストを作成
# reconstructed_data_list = []

# # 再構築を行い、結果をリストに追加
# for i, data in enumerate(validation_loader):
#     data = data.to(device)
#     # 転置ではなく、次元を入れ替える
#     data = data.unsqueeze(1)  # ポイント数の次元を追加
#     reconstructed_data = output[0].cpu().detach().numpy()
    
#     # リストに再構築データを追加
#     reconstructed_data_list.append(reconstructed_data)

# # リスト内のデータをNumPy配列に変換
# reconstructed_data_array = np.array(reconstructed_data_list)

# # 1つのNumPyファイルに保存
# output_filename = f"{output_folder}/reconstructed.npy"
# np.save(output_filename, reconstructed_data_array)

# print("再構築が完了し、新しいNumPyファイルに保存されました。")


# # モデルを読み込む
# net = model.PointCloudAE(point_size, latent_size)
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# net.eval()  # モデルを評価モードに設定

# # 元のデータが格納されている既存のNumPyファイルのパスを指定
# existing_data_path = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2/test_data_box_touitu.npy"  # 既存のデータファイルのパスを設定してください
# output_folder = "/home/dl-box/Desktop/Miki_Sotuken/WorkSpace/Sotuken_B_linux/Point-Cloud-Autoencoder/SotukenB-Point-Cloud-Autoencoder/output_10000_4096points_32batch_2"  # 保存先フォルダのパスを設定してください

# # 既存のNumPyファイルからデータを読み込む
# existing_data = np.load(existing_data_path)

# # 前処理を行い、テンソルに変換
# validation_dataset = ReadDataset(existing_data)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

# # 新しいNumPyファイルに再構築データを累積するためのリストを作成
# reconstructed_data_list = []

# # 再構築を行い、結果をリストに追加
# for i, data in enumerate(validation_loader):
#     data = data.to(device)
#     # 転置ではなく、次元を入れ替える
#     data = data.unsqueeze(1)  # ポイント数の次元を追加
#     reconstructed_data = output[0].cpu().detach().numpy()
    
#     # リストに再構築データを追加
#     reconstructed_data_list.append(reconstructed_data)

# # リスト内のデータをNumPy配列に変換
# reconstructed_data_array = np.array(reconstructed_data_list)

# # 1つのNumPyファイルに保存
# output_filename = f"{output_folder}/reconstructed.npy"
# np.save(output_filename, reconstructed_data_array)

# print("再構築が完了し、新しいNumPyファイルに保存されました。")
