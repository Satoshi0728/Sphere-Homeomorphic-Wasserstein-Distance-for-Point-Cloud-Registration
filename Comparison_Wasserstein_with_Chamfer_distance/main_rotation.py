"""
ライブラリのインポート
"""
import torch
import argparse
import numpy as np
import os
import ot
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader as geometric_DataLoader
from torch_geometric.utils import to_networkx as to_networkx
from torch_geometric.data import Batch as Batch
from tqdm import tqdm
# Loss func
from pytorch3d.loss import chamfer_distance
#自作ライブラリー
from Data_set_transformation import Dataset_Transformation as transform

from Data_set_maker import Data_set_maker_add_noise_and_rigid_transformation
from Data_set_maker import get_tensor_data_from_geometric_batch
import open3d as o3d
from torch.utils.data import DataLoader as pytorch_DataLoader
from Data_set_transformation import Dataset_pytorch
from losses import log_Sinkhorn_Distance_Loss
import pickle


def show_point_cloud(data, data2 = None):
    xyz1 = (data).detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz1)
    pcd.paint_uniform_color([0, 0, 1])

    if data2 == None:
        o3d.visualization.draw_geometries([pcd])
    if data2 != None:
        xyz2 = (data2).detach().numpy()
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz2)
        pcd2.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd,pcd2])


def show_point_cloud2(data, data2 = None):
    xyz1 = (data.pos).detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz1)
    pcd.paint_uniform_color([0, 0, 1])

    if data2 == None:
        o3d.visualization.draw_geometries([pcd])
    if data2 != None:
        xyz2 = (data2.y).detach().numpy()
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz2)
        pcd2.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd,pcd2])

#==============================================================================
"""
POT loss
"""
def POT_loss(x, y, criteria , device, p = 2):
    C = cost_matrix(x, y, p).to(device)
    x_points = x.shape[-2]
    y_points = y.shape[-2]
    if x.dim() == 2:
        batch_size = 1
    else:
        batch_size = x.shape[0]
    # probability vectors (経験分布なら 1/ sample size)
    probability_vector_of_x = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
    probability_vector_of_y = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

    emd = 0
    for index in range(batch_size):
        loss = torch.pow(criteria(probability_vector_of_x[index], probability_vector_of_y[index], C[index]), 1. / p)
        emd += loss
    return emd


def cost_matrix(x, y, p=2):
    """
    input source shape: B x N x D(= 3)
    input target shape B x M x D(= 3)
    output C shape: B x N x M
    p : Lp norm
    """
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.pow(torch.sum((torch.abs(x_col - y_lin)) ** p, -1), 1. / p)
    return C

#==============================================================================
"""
ディレクトリの絶対パス取得
"""
# ディレクトリの絶対パス取得
BASE_DIR = os.getcwd()
#==============================================================================
"""
args の設定
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--noise_m', type=int, default= 0, help='noise_mean')
    parser.add_argument('--noise_s', type= float, default= 0.00, help='noise_sigma')
    parser.add_argument('--angle_r', type=int, default= 90, help='angle_range')
    parser.add_argument('--translation_r', type=int, default= 0, help='translation_range')
    parser.add_argument('--cuda_num', type=int, default=0 , help='cuda_number')
    parser.add_argument('--ex_date', type=str, default='experiment_date', help='experiment_date')
    parser.add_argument('--ex_ver', type=str, default='experiment_version', help='experiment_version')
    parser.add_argument('--source_p_n', type=int, default= 1024, help='source_point_num')
    parser.add_argument('--target_p_n', type=int, default= 1024, help='target_point_num')
    parser.add_argument('--num_epoch', type=int, default= 3001, help='num_epoch')
    parser.add_argument('--batch_size', type=int, default= 32, help='batch_size')
    parser.add_argument('--workers', type=int, default= 2, help='workers')
    parser.add_argument('--lr', type=float, default= 1e-3, help='lr')
    parser.add_argument('--pcr_iteration_num', type=int, default= 8, help='PCR_iteration_num')
    parser.add_argument('--seed', type=int, default= 1234, help='SEED')
    parser.add_argument('--load_model', type=str, default= 'None', help='load_model')
    parser.add_argument('--sinkhorn_eps', type=float, default= 0.01, help='sinkhorn_eps')
    parser.add_argument('--sinkhorn_iter', type=int, default= 100, help='sinkhorn_iter')
    args = parser.parse_args()
    return args

#==============================================================================
"""
main
"""
def main():
    args = options()
    #ハイパラ=====================================
    #実験日
    experiment_date = args.ex_date
    experiment_version = args.ex_ver
    cuda_number = args.cuda_num
    # ノイズ関係
    mean = args.noise_m
    sigma = args.noise_s
    #疎密を変更する時は、点群数をソースとターゲットで変更すること
    source_point_num = args.source_p_n
    target_point_num = args.target_p_n

    #ソースデータセットの剛体変換関係
    angle_range= args.angle_r
    translation_range = args.translation_r
    #モデル関係
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    workers = args.workers
    lr = args.lr
    iteration_num = args.pcr_iteration_num  #iterativeにする時は、複数回を指定すること
    #学習率を変更するタイミングを指定
    # scheduler_step_size =  num_epoch//4
    #シード数
    SEED = args.seed
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_max_iter = args.sinkhorn_iter
    
    
    device = torch.device(f'cuda:{cuda_number}' if torch.cuda.is_available() else 'cpu')
    #ログの記録内容=============================================
	#実験名
    experiment_name = f"{experiment_date}" + f"_{experiment_version}"
    #ログの中身
    run_log_script = ("=======================================" + '\n'+
                "Experiment_name " + str(experiment_date) + ' ' + str(experiment_version) + '\n'+
                "SEED " + str(SEED) + '\n'+
                "source_point_num " + str(source_point_num) + '\n'+
                "target_point_num " + str(target_point_num)  + '\n'+
                "Learning late " + str(lr) + '\n'+
                "mean " + str(mean) + '\n'+
                "sigma " + str(sigma) + '\n'+
                "angle_range " + str(angle_range) + '\n'+
                "Learning late " + str(lr) + '\n'+
                "translation_range " + str(translation_range) + '\n'+
                "batch_size " + str(batch_size) + '\n'+
                "iteration_num  " + str(iteration_num) + '\n'+
                "translation_range " + str(translation_range) + '\n'+
                "cuda_num " + str(cuda_number) + '\n'+
                "sinkhorn_eps " + str(sinkhorn_eps) + '\n'+
                "sinkhorn_max_iter " + str(sinkhorn_max_iter) + '\n'+
				"=======================================")
    #ログの名前
    log_dir = BASE_DIR +  "/log" + '/' + experiment_name
    point_clouds_save_dir = log_dir + '/' + "point_clouds" + '_' + experiment_name
    log_dir_run_log = log_dir + "/" + "run.log"

    test_loss_list = []
    rotation_range = np.arange(90, 180.1, 1)
    for angle_limit in rotation_range:
        print("angle_limit", angle_limit)
        test_dataset = Data_set_maker_add_noise_and_rigid_transformation(BASE_DIR = BASE_DIR, train_or_test = "test", source_point_num = source_point_num, target_point_num = target_point_num, Dataset_Transformation = transform, noise_mean = mean, noise_sigma = sigma, rigid_transform_angle_range = angle_limit, rigid_transform_translation_range = translation_range)()
        test_set = Dataset_pytorch(test_dataset)
        #データローダーの作成
        test_dataloader = pytorch_DataLoader(test_set, batch_size=batch_size, shuffle=False)

        CD_loss = 0
        for i, data in enumerate(test_dataloader):
            template, source = data
            loss = chamfer_distance(source, template,  batch_reduction = 'sum')[0]
            CD_loss += loss.item()

        SD_loss = 0
        criteria = log_Sinkhorn_Distance_Loss(eps= sinkhorn_eps, max_iter= sinkhorn_max_iter, batch_reduction ='sum', type_of_cost_norm = 'L2')
        for i, data in enumerate(test_dataloader):
            template, source = data
            loss, P, C  = criteria(template, source, device)
            SD_loss += loss.item()

        WD_criteria = ot.emd2
        WD_loss = 0
        for i, data in enumerate(test_dataloader):
            template, source = data
            loss = POT_loss(template, source, WD_criteria, device, p = 2)
            # print(loss)
            WD_loss += loss.item()


        losses = [angle_limit, CD_loss / len(test_dataset), SD_loss / len(test_dataset), WD_loss / len(test_dataset)]
        test_loss_list.append(losses)

    print(test_loss_list)
    with open( BASE_DIR +  '/' + "log" + '/' + f"{experiment_version}_xyz_axis_rotation_loss.pkl", 'wb') as f:
        pickle.dump(test_loss_list, f)



"""
mainの実行
"""
if __name__ == '__main__':
    main()
