import os
import copy
import sys
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch as Batch
from torch_geometric.data import Data as Geometric_Data
from torch_geometric.datasets import ModelNet
#自作
from data_utils.Data_set_transformation import Dataset_Transformation as transform


BASE_DIR = os.getcwd()


# """
# データセットにノイズとrigid transformationを加えるクラス
# """
# class Data_set_maker_add_noise_and_rigid_transformation_optuna():
#     def __init__(self, BASE_DIR = BASE_DIR, dataset = ModelNet, train_or_test = "train" , source_point_num = 2048, target_point_num = 2048, Dataset_Transformation =transform, noise_mean = 0, noise_sigma = 0.03, rigid_transform_angle_range = 45, rigid_transform_translation_range = 1):
#         super().__init__()
#         self.mean = noise_mean
#         self.sigma = noise_sigma
#         self.angle_range = rigid_transform_angle_range
#         self.translation_range = rigid_transform_translation_range
#         self.Dataset_Transformation = Dataset_Transformation
#         self.ModelNet_source = self.load_dataset(dataset, BASE_DIR, train_or_test, source_point_num)
#         self.ModelNet_target = self.load_dataset(dataset, BASE_DIR, train_or_test, target_point_num)

#     #ModelNetの点群数と、train_or_testを指定して、データセットを作成する関数
#     def load_dataset(self, dataset, BASE_DIR = BASE_DIR, train_or_test = "train", point_num = 2048):
#         dataset_dir = BASE_DIR + "/modelnet" + "/modelnet10_" + str(point_num)
#         pre_transform = T.Compose([
#             T.SamplePoints(point_num, remove_faces=True, include_normals=False),
#             T.NormalizeScale(),
#             ])
#         #train_or_testがtrainかtestかを判定
#         if train_or_test == "train":
#             train = True
#         elif train_or_test == "test":
#             train = False
#         else:
#             print("Error! you set long srtings in Data_set_maker_add_noise_and_rigid_transformation! please set train_or_test as train or test")
#             sys.exit(1)
#         return dataset(dataset_dir, name = "10", train = train, transform = None, pre_transform = pre_transform, pre_filter = None)

#     #pytorch Geometricのposソース点群、yにターゲット点群の情報を入れる関数
#     def make_dataset_add_pos_and_y_rotation_translation(self,source_dataset, target_dataset):
#         if len(source_dataset) != len(target_dataset):
#             return "Error"
#         Data_list = []
#         for i in range(len(source_dataset)):
#             Data = Geometric_Data(pos = source_dataset[i].pos, y = target_dataset[i].pos)
#             Data_list.append(Data)
#         Data_set = Batch().from_data_list(Data_list)
#         return Data_set

#     #ノイズを加える.
#     def add_noise(self, data, mean=0, sigma=0.03):
#         data_shape = data[0].pos.shape
#         noisy_data = copy.deepcopy(data)
#         for i in range(len(noisy_data)):
#             noise = torch.normal(mean = mean, std = sigma, size = data_shape)
#             noisy_data[i].pos += noise
#         return noisy_data


#     def __call__(self):
#         #add noise
#         source_dataset_added_noise = self.add_noise(self.ModelNet_source, mean = self.mean, sigma = self.sigma)
#         #rigit transformation
#         source_dataset_added_noise_rigid_transform = self.Dataset_Transformation(len(source_dataset_added_noise), angle_range = self.angle_range, translation_range= self.translation_range)
#         source_dataset_added_noise_rigid_transformed = source_dataset_added_noise_rigid_transform(source_dataset_added_noise)

#         #データセットの結合 posはソースデータセットの座標, yはターゲットデータセットの座標
#         dataset = self.make_dataset_add_pos_and_y_rotation_translation(source_dataset=source_dataset_added_noise_rigid_transformed, target_dataset= self.ModelNet_target)
        
#         return dataset, source_dataset_added_noise_rigid_transform.igt_rotation_list, source_dataset_added_noise_rigid_transform.igt_translation_list








def split_train_validation_dataset(dataset, val_split=0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset






"""
データセットにノイズとrigid transformationを加えるクラス
"""
class Data_set_maker_add_noise_and_rigid_transformation_optuna():
    def __init__(self, BASE_DIR = BASE_DIR, dataset = ModelNet, train_or_test = "train" , source_point_num = 2048, target_point_num = 2048, Dataset_Transformation =transform, noise_mean = 0, noise_sigma = 0.03, rigid_transform_angle_range = 45, rigid_transform_translation_range = 1):
        super().__init__()
        self.mean = noise_mean
        self.sigma = noise_sigma
        self.angle_range = rigid_transform_angle_range
        self.translation_range = rigid_transform_translation_range
        self.Dataset_Transformation = Dataset_Transformation
        self.ModelNet_source = self.load_dataset(dataset, BASE_DIR, train_or_test, source_point_num)
        self.ModelNet_target = self.load_dataset(dataset, BASE_DIR, train_or_test, target_point_num)

    #ModelNetの点群数と、train_or_testを指定して、データセットを作成する関数
    def load_dataset(self, dataset, BASE_DIR = BASE_DIR, train_or_test = "train", point_num = 2048):
        dataset_dir = BASE_DIR + "/modelnet" + "/modelnet10_" + str(point_num)
        pre_transform = T.Compose([
            T.SamplePoints(point_num, remove_faces=True, include_normals=False),
            T.NormalizeScale(),
            ])
        #train_or_testがtrainかtestかを判定
        if train_or_test == "train":
            train = True
        elif train_or_test == "test":
            train = False
        else:
            print("Error! you set long srtings in Data_set_maker_add_noise_and_rigid_transformation! please set train_or_test as train or test")
            sys.exit(1)
        return dataset(dataset_dir, name = "10", train = train, transform = None, pre_transform = pre_transform, pre_filter = None)

    #pytorch Geometricのposソース点群、yにターゲット点群の情報を入れる関数
    def make_dataset_add_pos_and_y_rotation_translation(self,source_dataset, target_dataset, rotation, translation):
        if len(source_dataset) != len(target_dataset):
            return "Error"
        Data_list = []
        for i in range(len(source_dataset)):
            Data = Geometric_Data(pos = source_dataset[i].pos, y = target_dataset[i].pos, edge_index= rotation[i], edge_attr= translation[i])
            Data_list.append(Data)
        Data_set = Batch().from_data_list(Data_list)
        return Data_set

    #ノイズを加える.
    def add_noise(self, data, mean=0, sigma=0.03):
        data_shape = data[0].pos.shape
        noisy_data = copy.deepcopy(data)
        for i in range(len(noisy_data)):
            noise = torch.normal(mean = mean, std = sigma, size = data_shape)
            noisy_data[i].pos += noise
        return noisy_data


    def __call__(self):
        #add noise
        source_dataset_added_noise = self.add_noise(self.ModelNet_source, mean = self.mean, sigma = self.sigma)
        #rigit transformation
        source_dataset_added_noise_rigid_transform = self.Dataset_Transformation(len(source_dataset_added_noise), angle_range = self.angle_range, translation_range= self.translation_range)
        source_dataset_added_noise_rigid_transformed = source_dataset_added_noise_rigid_transform(source_dataset_added_noise)

        #データセットの結合 posはソースデータセットの座標, yはターゲットデータセットの座標
        dataset = self.make_dataset_add_pos_and_y_rotation_translation(source_dataset=source_dataset_added_noise_rigid_transformed, target_dataset= self.ModelNet_target, rotation= source_dataset_added_noise_rigid_transform.igt_rotation_list, translation= source_dataset_added_noise_rigid_transform.igt_translation_list)
        
        return dataset




"""
data_batchからtensor型のsourceと、tensor型のtargetを取得
"""
def get_tensor_data_from_geometric_batch(batch_data, source_or_target = "source"):
	"""
	input: Data_batch
	output: tensor of shape (batch_size, num_points, feature_dim)
	"""
	list = []
	if source_or_target == "source":
		for i in range(len(batch_data)):
			list.append(batch_data[i].pos)
	elif source_or_target == "target":
		for i in range(len(batch_data)):
			list.append(batch_data[i].y)
	tensor = torch.cat(list).reshape(len(list), *list[0].shape)
	return tensor















# def split_train_validation_dataset(dataset, val_split=0.2):
#     val_size = int(len(dataset) * val_split)
#     train_size = len(dataset) - val_size
#     train_subset = torch.utils.data.Subset(dataset, range(0, train_size))
#     val_subset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
#     print("train_subset", train_subset[0])
#     return train_subset, val_subset