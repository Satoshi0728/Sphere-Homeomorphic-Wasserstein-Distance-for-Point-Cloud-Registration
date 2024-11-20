import os
import copy
import numpy as np
import sys
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
import torch.nn.functional as F

BASE_DIR = os.getcwd()


#ノイズを加える. =============================================================================
def add_noise(data, mean=0, sigma=0.03):
    data_shape = data[0].pos.shape
    noisy_data = copy.deepcopy(data)

    for i in range(len(noisy_data)):
        noise = torch.normal(mean = mean, std = sigma, size = data_shape)
        noisy_data[i].pos += noise
    return noisy_data

#オイラー角とクオータニオン系 =============================================================================
def qmul(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1
    return result.reshape(original_shape)


#データ系 =============================================================================
"""
データをtrainとvalidationに分割する関数
"""
def split_train_validation_dataset(dataset, val_split=0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


# """
# data_batchからtensor型のsourceと、tensor型のtargetを取得
# """
# def get_tensor_data_from_geometric_batch(batch_data, source_or_target = "source"):
# 	"""
# 	input: Data_batch
# 	output: tensor of shape (batch_size, num_points, feature_dim)
# 	"""
# 	list = []
# 	if source_or_target == "source":
# 		for i in range(len(batch_data)):
# 			list.append(batch_data[i].pos)
# 	elif source_or_target == "target":
# 		for i in range(len(batch_data)):
# 			list.append(batch_data[i].y)
# 	tensor = torch.cat(list).reshape(len(list), *list[0].shape)
# 	return tensor



#データセットの変形 =============================================================================
class Dataset_Transformation():
    def __init__(self, data_size, angle_range=45, translation_range=1):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.data_size = data_size
        self.index = 0
        #一旦保存しない
        # self.igt_rotation_list = []
        # self.igt_translation_list = []

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg


    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)
        #一般的な回転
        rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
        #x軸回りのみ
        # rot = np.array([[max_rotation, 0, 0]])
        #y軸回りのみ
        # rot = np.array([[0, max_rotation, 0]])
        #z軸回りのみ
        # rot = np.array([[0, 0, max_rotation]])

        trans = np.random.uniform(-1, 1, [1, 3])
        trans = np.sqrt(max_translation) *(trans/ np.linalg.norm(trans))
        # print((trans*trans).sum())
        quat = euler_to_quaternion(rot, "xyz")
        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)
        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
        return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = Dataset_Transformation.get_quaternion(pose_7d).expand([N, -1])
            # print("quat", quat.shape)
            rotated_point_cloud = qrot(quat, point_cloud)

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = Dataset_Transformation.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = qrot(quat, point_cloud)

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = Dataset_Transformation.quaternion_rotate(point_cloud, pose_7d) + Dataset_Transformation.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    def __call__(self, source):
        self.igt = self.transformations[self.index]
        igt = self.create_pose_7d(self.igt)
        self.igt_rotation = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)        # [3x3]
        self.igt_translation = self.get_translation(igt)
        # #回転と、平行移動をリストで保存
        # self.igt_rotation_list.append(self.igt_rotation)
        # self.igt_translation_list.append(self.igt_translation)
        transformed_source = self.quaternion_rotate(source, igt) + self.get_translation(igt)
        return transformed_source



#==============================================================================
"""
pytorch_geometricのデータセットから、pytorchのデータローダーで読み込めるように変換するクラス
"""
class Dataset_pytorch(Dataset):
    def __init__(self, source_point_num, target_point_num, noise_mean , noise_sigma , rigid_transform_angle_range , rigid_transform_translation_range, BASE_DIR = BASE_DIR, train_or_test = "train"):
        super().__init__()
        #ノイズを加えるクラス
        self.Data_set_maker_add_noise = Data_set_maker_add_noise(BASE_DIR = BASE_DIR, train_or_test = train_or_test, source_point_num = source_point_num, target_point_num = target_point_num, noise_mean = noise_mean, noise_sigma = noise_sigma)

        #ソースとターゲット
        self.source_dataset_added_noise = self.Data_set_maker_add_noise()
        self.target_dataset = self.Data_set_maker_add_noise.ModelNet_target
        #剛体変換を加えるクラス
        self.dataset_transform = Dataset_Transformation(len(self.source_dataset_added_noise),angle_range= rigid_transform_angle_range, translation_range= rigid_transform_translation_range)

    def __len__(self) -> int:
        return len(self.source_dataset_added_noise)

    # ここで取り出すデータを指定している
    def __getitem__(self,index: int):
        self.dataset_transform.index = index
        source = self.source_dataset_added_noise[index].pos
        target = self.target_dataset[index].pos
        transformed_source = self.dataset_transform(source)
        return target, transformed_source, self.dataset_transform.igt_rotation, self.dataset_transform.igt_translation



#==============================================================================
"""
ロードして、noiseを加えるクラス
"""
class Data_set_maker_add_noise():
    def __init__(self, BASE_DIR = BASE_DIR, dataset = ModelNet, train_or_test = "train" , source_point_num = 2048, target_point_num = 2048, noise_mean = 0, noise_sigma = 0.03):
        super().__init__()
        self.mean = noise_mean
        self.sigma = noise_sigma
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

    #ノイズを加える.
    def add_noise(self, data, mean=0, sigma=0.03):
        data_shape = data[0].pos.shape
        # print(data[0].pos)
        noisy_data = copy.deepcopy(data)
        for i in range(len(noisy_data)):
            noise = torch.normal(mean = mean, std = sigma, size = data_shape)
            noisy_data[i].pos += noise
        return noisy_data

    def __call__(self):
        return self.add_noise(self.ModelNet_source, mean = self.mean, sigma = self.sigma)

