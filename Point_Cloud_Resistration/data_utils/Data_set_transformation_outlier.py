from pathlib import Path
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

import torch
from torch_geometric.data import Data as Data
from torch_geometric.utils import to_networkx as to_networkx
import copy
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import sys
"""
参考文献
・https://qiita.com/take4eng/items/ae487c82a6f7d60ceba8#:~:text=%E3%82%AA%E3%82%A4%E3%83%A9%E3%83%BC%E8%A7%92%E3%81%8B%E3%82%89%E3%81%AE%E5%A4%89%E6%8F%9B,-%E5%89%8D%E9%A0%85%E3%81%BE%E3%81%A7%E3%81%A7&text=%E3%82%AA%E3%82%A4%E3%83%A9%E3%83%BC%E8%A7%92%E3%82%92%E3%82%AF%E3%82%A9%E3%83%BC%E3%82%BF%E3%83%8B%E3%82%AA%E3%83%B3%E3%81%AB,%CE%B2%2C%CE%B3
"""


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


#データセットの変形 =============================================================================
class Dataset_Transformation_outlier:
    def __init__(self, data_size, angle_range=45, translation_range=1, outlier_num = 10, outlier_sigma = 0.5):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.data_size = data_size
        self.outlier_num = outlier_num
        self.outlier_sigma = outlier_sigma

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)
        # print(max_rotation)
        # rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
        rot = np.array([[max_rotation, 0, 0]])
        # rot = np.array([[max_rotation, max_rotation, max_rotation]])
        # print(rot[0])
        
        
        trans = np.random.uniform(-1, 1, [1, 3])
        trans = np.sqrt(max_translation) *(trans/ np.linalg.norm(trans))
        # trans = np.array([[0, 0, 0]])
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
            quat = Dataset_Transformation_outlier.get_quaternion(pose_7d).expand([N, -1])
            rotated_point_cloud = qrot(quat, point_cloud)

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = Dataset_Transformation_outlier.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = qrot(quat, point_cloud)

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = Dataset_Transformation_outlier.quaternion_rotate(point_cloud, pose_7d) + Dataset_Transformation_outlier.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix


    @staticmethod
    def add_outlier(point_cloud, outlier_num, outlier_sigma):
        for i in range(len(point_cloud)):
            for _ in range(outlier_num):
                random_index = torch.randint(low=0, high= len(point_cloud[0].pos)-1, size=(1,)).item()
                outlier_transform = torch.normal(mean = 0, std = outlier_sigma,  size=(1, 3))
                point_cloud[i].pos[random_index] = outlier_transform
        return point_cloud


    def __call__(self, template):
        rotated_templete = copy.deepcopy(template)
        index = 0
        for i in range(self.data_size):
            self.igt = self.transformations[index]
            igt = self.create_pose_7d(self.igt)
            self.igt_rotation = self.quaternion_rotate(torch.eye(3), igt).permute(1, 0)        # [3x3]
            self.igt_translation = self.get_translation(igt)                                   # [1x3]
            rotated_templete[index].pos += self.quaternion_rotate(template[index].pos, igt) + self.get_translation(igt) - rotated_templete[index].pos
            #回数をインクリメント
            index += 1

        if self.outlier_num > 0:
            rotated_templete = self.add_outlier(rotated_templete, self.outlier_num, self.outlier_sigma)
        elif self.outlier_num < 0:
            print("outlier should be positive.")
            sys.exit(1)
        return rotated_templete



    # def add_outlier(self, point_cloud, outlier):
    #     if outlier < 0:
    #         print("outlier should be positive.")
    #         sys.exit(1)
    #     elif outlier > 0:
    #         transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range, outlier = False) for _ in range(self.data_size - outlier)]
    #         for _ in range(outlier):
    #             random_index = np.random.randint(0, len(transformations)-1)
    #             outlier_transform = self.create_random_transform(torch.float32, self.angle_range, self.translation_range, outlier = True, outlier_sigma = 10)
    #             transformations.insert(random_index, outlier_transform)
    #     elif outlier == 0:
    #         transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range, outlier = False) for _ in range(self.data_size) ]
    #     return transformations


#==============================================================================
"""
pytorch_geometricのデータセットから、pytorchのデータローダーで読み込めるように変換するクラス
"""
class Dataset_pytorch(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    # ここで取り出すデータを指定している
    def __getitem__(self,index: int):
        source = self.dataset[index].pos
        target = self.dataset[index].y
        return target, source

    def __len__(self) -> int:
        return len(self.dataset)





#main =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=float, default=0, help='mean of noise')
    parser.add_argument('-s', type=float, default=0.00, help='sigma of noise')
    args = parser.parse_args()

    # current_path = os.getcwd()
    # print(current_path)
    dataset_dir = "/Users/satoshiishikawa/Library/CloudStorage/GoogleDrive-sato728.ktennis@gmail.com/マイドライブ/1.Imaizumi_lab/1.Research/1.CD_VS_SD/modelnet/modelnet10_1024"
    print(dataset_dir)
    #pre_transform
    pre_transform = T.Compose([
        T.SamplePoints(1024, remove_faces=True, include_normals=True),
        T.NormalizeScale(),
    ])

    #データセットの作成
    target_train_dataset = ModelNet(dataset_dir, name="10", train=True, transform=None, pre_transform=pre_transform, pre_filter=None)
    target_test_dataset = ModelNet(dataset_dir, name="10", train=False, transform=None, pre_transform=pre_transform, pre_filter=None)
    #add noise
    source_train_dataset = add_noise(target_train_dataset, mean = args.m, sigma = args.s)
    source_test_dataset = add_noise(target_test_dataset, mean = args.m, sigma = args.s)
    #rigit transformation
    transformed_target_train_dataset = Dataset_Transformation_outlier(len(source_train_dataset), angle_range=0, translation_range=0, outlier_num = 10, outlier_sigma = 0.4)(source_train_dataset)
    # visualize data
    # show_point_cloud(source_train_dataset[0], transformed_target_train_dataset[0])
    # show_point_cloud(target_test_dataset[0], source_test_dataset[0])
    for i in range(3):
        show_point_cloud(transformed_target_train_dataset[i], source_train_dataset[i])
















