"""
ライブラリのインポート
"""
import torch
import argparse
import numpy as np
import os
import pickle
# from tensorboardX import SummaryWriter
from tqdm import tqdm
# Loss func
from pytorch3d.loss import chamfer_distance
import transforms3d
from torch.utils.data import DataLoader
from log_utils import Save_point_cloud

"""""""""""""""""""""""""""""""""""""""
model
"""""""""""""""""""""""""""""""""""""""
from models import PCRNet
from models import MLP_Architecture

"""""""""""""""""""""""""""""""""""""""
Dataset
"""""""""""""""""""""""""""""""""""""""
#自作ライブラリー
from data_utils.Data_set_maker import Dataset_pytorch
from data_utils.Data_set_maker import split_train_validation_dataset

#==============================================================================
"""
ディレクトリの絶対パス取得
"""
# ディレクトリの絶対パス取得
BASE_DIR = os.getcwd()

#==============================================================================
"""
シード固定
"""
def fix_seed(seed):
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed+ 1)
    torch.cuda.manual_seed_all(seed + 2)
    torch.backends.cudnn.deterministic = True


#==============================================================================
"""
エラー計算
"""
class calculate_accuracies():
    def __init__(self, rotation, translation, est_rotation, est_translation, success_ratio_rotation, success_ratio_translation):
        self.rotation = rotation
        self.translation = translation
        self.est_rotation = est_rotation
        self.est_translation = est_translation
        self.success_ratio_rotation = success_ratio_rotation
        self.success_ratio_translation = success_ratio_translation
        self.success_ratio_translation_cnt = 0
        self.success_ratio_rotation_cnt = 0

    @staticmethod
    def calculate_errors(rotation, translation, est_rotation, est_translation):
        translation = -np.matmul(rotation.T, translation.T).T
        trans_error = np.sqrt(np.sum(np.square(translation - est_translation)))
        error_rotation = np.dot(rotation, est_rotation)
        _, angle = transforms3d.axangles.mat2axangle(error_rotation)
        rot_error = abs(angle*(180/np.pi))
        return trans_error, rot_error

    def calculate_accuracy(self, rotation, translation, est_rotation, est_translation, success_ratio_rotation, success_ratio_translation):
        error_temp = []

        for rotation_i, translation_i, est_rotation_i, est_translation_i in zip(rotation, translation, est_rotation, est_translation):
            error_temp.append(self.calculate_errors(rotation_i, translation_i, est_rotation_i, est_translation_i))

        #errorカウント
        for error in error_temp:
            if error[1] <= success_ratio_rotation:
                self.success_ratio_rotation_cnt += 1
            if error[0] <= success_ratio_translation:
                self.success_ratio_translation_cnt += 1
        self.success_ratio_rotation_cnt = self.success_ratio_rotation_cnt/len(error_temp)
        self.success_ratio_translation_cnt = self.success_ratio_translation_cnt/len(error_temp)
        return np.mean(error_temp, axis=0)

    def __call__(self):
        return self.calculate_accuracy(self.rotation, self.translation, self.est_rotation, self.est_translation, self.success_ratio_rotation, self.success_ratio_translation)



#==============================================================================
"""
test_one_epoch
"""
def test_one_epoch(device, model, test_loader, criteria, iteration_num, limit, rot_or_trans):
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        count = 0
        errors = []
        success_ratio_rotation = 0.0
        success_ratio_translation = 0.0

        get_point_cloud = False
        if rot_or_trans == "rot":
            rot_limit = limit
            trans_limit = 1
        elif rot_or_trans == "trans":
            rot_limit = 90
            trans_limit = limit
        else:
            get_point_cloud = True

        for i, data in enumerate(tqdm(test_loader)):
            template, source, rotation, translation = data

            template = template.to(device)
            source = source.to(device)
            translation = translation.to(device)
            # mean substraction
            translation = translation - torch.mean(source, dim=1).unsqueeze(1)
            source = source - torch.mean(source, dim=1, keepdim=True)
            template = template - torch.mean(template, dim=1, keepdim=True)

            output = model(template, source, iteration_num)
            loss_val = criteria(template, output['transformed_source'])[0]
            est_rotation = output['est_R']
            est_translation = output['est_t']

            if get_point_cloud:
                rot_or_trans(source, template, output['transformed_source'], "0")
                return loss_val, 0, 0, 0, 0
            
            calculate_accuracy = calculate_accuracies(rotation.detach().cpu().numpy(), translation.detach().cpu().numpy(), est_rotation.detach().cpu().numpy(), est_translation.detach().cpu().numpy(), rot_limit, trans_limit)

            errors.append(calculate_accuracy())
            success_ratio_rotation += calculate_accuracy.success_ratio_rotation_cnt
            success_ratio_translation += calculate_accuracy.success_ratio_translation_cnt
            test_loss += loss_val.item()
            count += 1

        errors = np.mean(np.array(errors), axis=0)
        test_loss = float(test_loss)/count
        success_ratio_rotation  = float(success_ratio_rotation )/count
        success_ratio_translation  = float(success_ratio_translation )/count
        # print("success_ratio_rotation: ", success_ratio_rotation)
        # print("success_ratio_translation: ", success_ratio_translation)
        return test_loss, errors[0], errors[1], success_ratio_rotation, success_ratio_translation

"""
test
"""
def test(test_criteria, iteration_num, device, model, test_loader, limit, rot_or_trans):
    test_loss, translation_error, rotation_error, success_ratio_rotation, success_ratio_translation  = test_one_epoch(device, model, test_loader, test_criteria, iteration_num, limit, rot_or_trans)
    return test_loss, translation_error, rotation_error, success_ratio_rotation, success_ratio_translation


#==============================================================================
"""
args の設定
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--ex_ver', type=str, default='experiment_version', help='experiment_version')
    parser.add_argument('--cuda_num', type=int, default=0 , help='cuda_number')
    parser.add_argument('--noise_m', type=int, default= 0, help='noise_mean')
    parser.add_argument('--noise_s', type= float, default= 0.02, help='noise_sigma')
    parser.add_argument('--source_p_n', type=int, default= 1024, help='source_point_num')
    parser.add_argument('--target_p_n', type=int, default= 1024, help='target_point_num')
    parser.add_argument('--angle_r', type=int, default= 45, help='angle_range')
    parser.add_argument('--translation_r', type=float, default= 1, help='translation_range')
    parser.add_argument('--batch_size', type=int, default= 32, help='batch_size')
    parser.add_argument('--workers', type=int, default= 2, help='workers')
    parser.add_argument('--pcr_iteration_num', type=int, default= 8, help='PCR_iteration_num')
    parser.add_argument('--seed', type=int, default= 1234, help='SEED')
    parser.add_argument('--pretrained', type=str, default= 'pcrnet_ishikawa_ver/log/5_14_1_chamfer_with_0.02_noise/models/best_model.t7', help='path to pretrained model file')
    parser.add_argument('--test_epoch', type = int, default= 1, help='test_epoch' )
    args = parser.parse_args()
    return args


#==============================================================================
"""
main
"""
def main():
    args = options()
    #シード固定=====================================
    #シード数
    SEED = args.seed
    fix_seed(SEED)

    #ハイパラ=====================================
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
    batch_size = args.batch_size
    workers = args.workers
    iteration_num = args.pcr_iteration_num  #iterativeにする時は、複数回を指定すること

    #損失関数
    test_criteria = chamfer_distance
    # test_criteria = test_criteria.to(device)

    #保存場所=====================================
    #ログの名前
    log_dir = BASE_DIR + '/' + 'log' + '/' + experiment_version
    point_clouds_save_dir = log_dir + '/' + "point_clouds_" + experiment_version
    save_point_cloud = Save_point_cloud(point_cloud_save_dir = point_clouds_save_dir)

    print(point_clouds_save_dir)
    #モデル=====================================
    feature_model = MLP_Architecture()
    model = PCRNet(feature_model = feature_model)
    device = torch.device(f"cuda:{cuda_number}") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model = model.to(device)
    #現在のディレクトリの表示
    print("current_path", BASE_DIR)

    #テスト=====================================
    # test_loss_list = []
    rotation_error_list = []
    translation_error_list = []
    source_point_num = "1024"
    target_point_num = "1024"

    dataset = Dataset_pytorch(BASE_DIR = BASE_DIR, train_or_test = "test", source_point_num = source_point_num, target_point_num = target_point_num, noise_mean = mean, noise_sigma = sigma, rigid_transform_angle_range = angle_range, rigid_transform_translation_range = translation_range)
    #データローダーの作成
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers,  pin_memory=True)

    #回転
    rotation_range = np.arange(0, 181, 1)
    for rotation_limit in rotation_range:
        test_loss, translation_error, rotation_error, success_ratio_rotation, success_ratio_translation  = test(test_criteria = test_criteria, iteration_num = iteration_num, device = device, model = model, test_loader = test_dataloader, limit = rotation_limit, rot_or_trans = "rot")
        # test_loss_list.append(test_loss)
        rotation_error_list.append([rotation_limit,success_ratio_rotation])
    # print("rotation_error_list", rotation_error_list)
    # print("Test Loss: {}, Rotation Error: {} & Translation Error: {}".format(test_loss, rotation_error, translation_error))
    #translation_errorとrotation_errorを保存
    with open(point_clouds_save_dir +  '/' + f"{experiment_version}_error.log", 'a') as file:
        file.write(f'rotation ave. {rotation_error}\ntranslation ave. {translation_error}\n')
    with open(point_clouds_save_dir +  '/' + f"{experiment_version}_rotation_error_list.pkl", 'wb') as f:
        pickle.dump(rotation_error_list, f)
    #並進
    translation_range = np.arange(0, 1.01, 0.01)
    for translation_limit in translation_range:
        test_loss, translation_error, rotation_error, success_ratio_rotation, success_ratio_translation  = test(test_criteria = test_criteria, iteration_num = iteration_num, device = device, model = model, test_loader = test_dataloader, limit = translation_limit, rot_or_trans = "trans")
        # test_loss_list.append(test_loss)
        translation_error_list.append([translation_limit,success_ratio_translation])
    # print("translation_error_list_list", translation_error_list_list)
    # print("Test Loss: {}, Rotation Error: {} & Translation Error: {}".format(test_loss, rotation_error, translation_error))
    with open(point_clouds_save_dir +  '/' + f"{experiment_version}_translation_error_list.pkl", 'wb') as f:
        pickle.dump(translation_error_list, f)
    #点群可視化
    test_loss, translation_error, rotation_error, success_ratio_rotation, success_ratio_translation  = test(test_criteria = test_criteria, iteration_num = iteration_num, device = device, model = model, test_loader = test_dataloader, limit = 0.0, rot_or_trans = save_point_cloud)


"""
mainの実行
"""
if __name__ == '__main__':
    main()
