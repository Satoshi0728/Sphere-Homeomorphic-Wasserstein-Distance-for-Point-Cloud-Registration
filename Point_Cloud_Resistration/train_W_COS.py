"""
ライブラリのインポート
"""
import time, datetime
import torch
import argparse
import numpy as np
import os
import transforms3d
from tensorboardX import SummaryWriter
# from torch_geometric.utils import to_networkx as to_networkx
from tqdm import tqdm
from torch.utils.data import DataLoader
"""""""""""""""""""""""""""""""""""""""
Dataset
"""""""""""""""""""""""""""""""""""""""
#自作ライブラリー
from data_utils.Data_set_maker import Dataset_pytorch
from data_utils.Data_set_maker import split_train_validation_dataset
"""""""""""""""""""""""""""""""""""""""
model
"""""""""""""""""""""""""""""""""""""""
from models import PCRNet
from models import MLP_Architecture

"""""""""""""""""""""""""""""""""""""""
log
"""""""""""""""""""""""""""""""""""""""
from log_utils import IOStream
from log_utils import _init_
from log_utils import Save_point_cloud

"""""""""""""""""""""""""""""""""""""""
losses
"""""""""""""""""""""""""""""""""""""""
# from losses import max_cos_similarity_wassersten_distance
# from losses import Flow_structure
# from losses import Cos_similarity_W
from losses import max_cos_disimilarity_wassersten_distance
from losses import Norm_Flow_structure
from losses import Cos_disimilarity_W

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

        # #errorカウント
        # for error in error_temp:
        #     if error[1] <= success_ratio_rotation:
        #         self.success_ratio_rotation_cnt += 1
        #     if error[0] <= success_ratio_translation:
        #         self.success_ratio_translation_cnt += 1
        # self.success_ratio_rotation_cnt = self.success_ratio_rotation_cnt/len(error_temp)
        # self.success_ratio_translation_cnt = self.success_ratio_translation_cnt/len(error_temp)
        return np.mean(error_temp, axis=0)

    def __call__(self):
        return self.calculate_accuracy(self.rotation, self.translation, self.est_rotation, self.est_translation, self.success_ratio_rotation, self.success_ratio_translation)


#==============================================================================
"""
test_one_epoch
"""
def test_one_epoch(device, model, test_loader, criteria, iteration_num, rotation_limit):
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        count = 0
        errors = []
        # success_ratio_rotation = 0.0
        # success_ratio_translation = 0.0
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

            loss_val, sphere_template, sphere_source = criteria(template, output['transformed_source'], train_or_test = "test")
            est_rotation = output['est_R']
            est_translation = output['est_t']

            calculate_accuracy = calculate_accuracies(rotation.detach().cpu().numpy(), translation.detach().cpu().numpy(), est_rotation.detach().cpu().numpy(), est_translation.detach().cpu().numpy(), rotation_limit, 1)
            errors.append(calculate_accuracy())

            test_loss += loss_val.item()
            count += 1

        errors = np.mean(np.array(errors), axis=0)
        test_loss = float(test_loss)/count
        # success_ratio_rotation  = float(success_ratio_rotation )/count
        # success_ratio_translation  = float(success_ratio_translation )/count
        # print("success_ratio_rotation: ", success_ratio_rotation)
        # print("success_ratio_translation: ", success_ratio_translation)
        return test_loss, errors[0], errors[1]

#==============================================================================
"""
train_one_epoch
"""
def train_one_epoch(device, model, train_loader, criteria, optimizer, iteration_num,epoch,save_point_cloud):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        template, source, rotation, translation = data
        # GPUに載せる
        template = template.to(device)
        source = source.to(device)

        # # mean substraction
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source, iteration_num)
        loss_val, sphere_template, sphere_source = criteria(template, output['transformed_source'], train_or_test = "train")
        loss_val.backward()

        optimizer.step()
        count += 1
        train_loss += loss_val.item()

    train_loss = float(train_loss)/count
    return train_loss


#==============================================================================
"""
train
"""
def train(train_criteria, test_criteria, iteration_num, device, optimizer, num_epoch, model, train_loader, test_loader, writer ,log_dir, run_log, save_point_cloud):
    best_test_loss = np.inf
    best_rot_error = np.inf
    best_trans_error = np.inf
    
    for epoch in range(num_epoch):
        time_sta = time.time()
        train_loss = train_one_epoch(device, model, train_loader, train_criteria, optimizer, iteration_num, epoch, save_point_cloud)
        test_loss, trans_error, rot_error = test_one_epoch(device, model, test_loader, test_criteria, iteration_num, epoch)
        time_end = time.time()
        tim_fast = time_end- time_sta

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer' : optimizer.state_dict(),
                    'phi': train_criteria.phi.state_dict(),
                    'phi_op': train_criteria.phi_op.state_dict(),
                    }
            torch.save(snap, '%s/models/best_model_snap.t7' % (log_dir))
            torch.save(model.state_dict(), '%s/models/best_model.t7' % (log_dir))
            torch.save(model.feature_model.state_dict(), '%s/models/best_ptnet_model.t7' % (log_dir))

        if rot_error < best_rot_error:
            best_rot_error = rot_error
            snap_rot = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_rot_error,
                    'optimizer' : optimizer.state_dict(),
                    'phi': train_criteria.phi.state_dict(),
                    'phi_op': train_criteria.phi_op.state_dict(),
                    }
            torch.save(snap_rot, '%s/models/best_rot_error_snap.t7' % (log_dir))
            torch.save(model.state_dict(), '%s/models/best_rot_error_model.t7' % (log_dir))
            torch.save(model.feature_model.state_dict(), '%s/models/best_rot_error_ptnet_model.t7' % (log_dir))

        if trans_error < best_trans_error:
            best_trans_error = trans_error
            snap_trans = {'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'min_loss': best_trans_error,
                        'optimizer' : optimizer.state_dict(),
                        'phi': train_criteria.phi.state_dict(),
                        'phi_op': train_criteria.phi_op.state_dict(),
                    }
            torch.save(snap_trans, '%s/models/best_trans_error_snap.t7' % (log_dir))
            torch.save(model.state_dict(), '%s/models/best_trans_error_model.t7' % (log_dir))
            torch.save(model.feature_model.state_dict(), '%s/models/best_trans_error_ptnet_model.t7' % (log_dir))

        # torch.save(model.state_dict(), '%s/models/model.t7' % (log_dir))
        # torch.save(model.feature_model.state_dict(), '%s/models/ptnet_model.t7' % (log_dir))

        writer.add_scalar('Train Loss', train_loss, epoch+1)
        writer.add_scalar('Test Loss', test_loss, epoch+1)
        writer.add_scalar('Best Test Loss', best_test_loss, epoch+1)
        writer.add_scalar("Rot Error", rot_error, epoch+1)
        writer.add_scalar("Best Rot Error", best_rot_error, epoch+1)
        writer.add_scalar("Trans Error", trans_error, epoch+1)
        writer.add_scalar("Best Trans Error", best_trans_error, epoch+1)
        
        run_log.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f, Rot error: %f, Best Rot error: %f,  Trans error: %f, Best Trans error: %f, Time: %f'%(epoch+1, train_loss* 10**2, test_loss* 10**2, best_test_loss* 10**2,rot_error, best_rot_error, trans_error, best_trans_error ,tim_fast))



def load_checkpoint(model, optimizer, phi, phi_op, device, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        phi.load_state_dict(checkpoint['phi'])
        phi_op.load_state_dict(checkpoint['phi_op'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    for state in phi_op.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, phi, phi_op, start_epoch



#==============================================================================
"""
args の設定
"""

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--ex_date', type=str, default='experiment_date', help='experiment_date')
    parser.add_argument('--ex_ver', type=str, default='experiment_version', help='experiment_version')
    parser.add_argument('--cuda_num', type=int, default=0 , help='cuda_number')
    parser.add_argument('--noise_m', type=int, default= 0, help='noise_mean')
    parser.add_argument('--noise_s', type= float, default= 0.02, help='noise_sigma')
    parser.add_argument('--source_p_n', type=int, default= 1024, help='source_point_num')
    parser.add_argument('--target_p_n', type=int, default= 1024, help='target_point_num')
    parser.add_argument('--angle_r', type=int, default= 45, help='angle_range')
    parser.add_argument('--translation_r', type=float, default= 1, help='translation_range')
    parser.add_argument('--num_epoch', type=int, default= 3001, help='num_epoch')
    parser.add_argument('--batch_size', type=int, default= 32, help='batch_size')
    parser.add_argument('--workers', type=int, default= 2, help='workers')
    parser.add_argument('--lr', type=float, default= 1e-3, help='lr')
    parser.add_argument('--weight_decay', type=float, default= 1.4096013153858628e-08, help='weight_decay')
    parser.add_argument('--phi_op_lr', type=float, default= 9.213233310357477e-05, help='phi_op_lr')
    parser.add_argument('--phi_op_weight_decay', type=float, default= 1.4096013153858628e-08, help='phi_op_weight_decay')
    parser.add_argument('--phi_num_flow_layer', type=int, default= 3, help='phi_num_flow_layer')
    parser.add_argument('--phi_lamda_of_regularization', type=float, default= 1.3111961119405346e-05, help='phi_lamda_of_regularization')
    parser.add_argument('--phi_max_iter', type=int, default= 1, help='phi_max_iter')
    parser.add_argument('--flow_name', type=str, default= 'Residual', help='flow_name')
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
    #シード固定=====================================
    #シード数
    SEED = args.seed
    fix_seed(SEED)
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
    weight_decay = args.weight_decay
    iteration_num = args.pcr_iteration_num  #iterativeにする時は、複数回を指定すること
    #学習率を変更するタイミングを指定
    # scheduler_step_size =  num_epoch//4


    #データセットの作成=====================================
    """
    pytorch geometricを使う場合
    """
    #データセットの作成
    dataset = Dataset_pytorch(BASE_DIR = BASE_DIR, train_or_test = "train", source_point_num = source_point_num, target_point_num = target_point_num, noise_mean = mean, noise_sigma = sigma, rigid_transform_angle_range = angle_range, rigid_transform_translation_range = translation_range)

    #trainデータセットの分割
    train_dataset, valid_dataset = split_train_validation_dataset(dataset, val_split=0.2)

    #データローダーの作成
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # #シード固定=====================================
    # fix_seed(SEED)

    #モデル=====================================
    feature_model = MLP_Architecture()
    model = PCRNet(feature_model = feature_model)
    # model = torch.compile(model, fullgraph=True)
    device = torch.device(f"cuda:{cuda_number}") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr, weight_decay=weight_decay)

    #SW with regularition=====================================
    if args.flow_name == "Residual":
        flow = "Residual"
    elif args.flow_name == "Planar":
        flow = "Planar"

    phi_op_lr = args.phi_op_lr
    phi_op_weight_decay = args.phi_op_weight_decay
    phi_num_flow_layer = args.phi_num_flow_layer
    phi_lamda_of_regularization = args.phi_lamda_of_regularization
    phi_max_iter = args.phi_max_iter
    lamda_of_regularization = phi_lamda_of_regularization


    phi = Norm_Flow_structure(flow_name =flow , n_flow_layer = phi_num_flow_layer)
    # phi = torch.compile(phi)
    phi_op = torch.optim.Adam(params=phi.parameters(), lr=  phi_op_lr, weight_decay=phi_op_weight_decay)
    CSW =  Cos_disimilarity_W(device = device, p = 2)


    #load model=====================================
    if args.load_model != 'None':
        # /home/ishikawa/Research/Point_Cloud_Registration/pcrnet_ishikawa_ver/log/5_15_1_sinkhorn_noise/models/best_model_snap.t7
        model ,optimizer, phi, phi_op, start_epoch = load_checkpoint(model, optimizer,  phi, phi_op, device,filename = args.load_model)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size , gamma=0.1)
    model = model.to(device)
    phi = phi.to(device)
    train_criteria = max_cos_disimilarity_wassersten_distance(phi = phi, CSW = CSW, phi_op = phi_op, lam = lamda_of_regularization, max_iter=phi_max_iter, device= device)


    #ログの記録内容=============================================
    #実験名
    experiment_name = f"{experiment_date}" + f"_{experiment_version}"
    today = datetime.datetime.fromtimestamp(time.time())
    #ログの中身
    run_log_script = (
                "Date of Experiment " + str(today.strftime('%b %d, %Y %I:%M:%S%p')) + '\n'+
                "=======================================" + '\n'+
                "Experiment_name " + str(experiment_date) + ' ' + str(experiment_version) + '\n'+
                "load model " + str(args.load_model) + '\n'+
                "SEED " + str(SEED) + '\n'+
                "source_point_num " + str(source_point_num) + '\n'+
                "target_point_num " + str(target_point_num)  + '\n'+
                "Learning late " + str(lr) + '\n'+
                "weight_decay " + str(weight_decay) + '\n'+
                "mean " + str(mean) + '\n'+
                "sigma " + str(sigma) + '\n'+
                "angle_range " + str(angle_range) + '\n'+
                "translation_range " + str(translation_range) + '\n'+
                "batch_size " + str(batch_size) + '\n'+
                "iteration_num  " + str(iteration_num) + '\n'+
                "cuda_num " + str(cuda_number) + '\n'+
                # "sinkhorn_eps " + str(sinkhorn_eps) + '\n'+
                # "sinkhorn_max_iter " + str(sinkhorn_max_iter) + '\n'+
                "phi_op_lr " + str(phi_op_lr) + '\n'+
                "phi_op_weight_decay " + str(phi_op_weight_decay) + '\n'+
                "phi_num_flow_layer " + str(phi_num_flow_layer) + '\n'+
                "phi_lamda_of_regularization " + str(phi_lamda_of_regularization) + '\n'+
                "phi_max_iter " + str(phi_max_iter) + '\n'+
                "flow " + str(flow) + '\n'+
                "=======================================")
    #ログの名前
    log_dir = BASE_DIR +  "/log" + '/' + experiment_name
    point_clouds_save_dir = log_dir + '/' + "point_clouds" + '_' + experiment_name
    log_dir_run_log = log_dir + "/" + "run.log"
    

    # # #データ保存関係=====================================
    #ディレクトリの作成
    _init_(log_dir = log_dir, point_clouds_save_dir = point_clouds_save_dir)
    run_log = IOStream(log_dir_run_log)
    writer = SummaryWriter(log_dir= log_dir)
    #実験内容の表示
    run_log.cprint(run_log_script)
    save_point_cloud = Save_point_cloud(point_cloud_save_dir = point_clouds_save_dir)
    #現在のディレクトリの表示
    print("current_path", BASE_DIR)

    #学習=====================================
    train(train_criteria = train_criteria,test_criteria = train_criteria, iteration_num =iteration_num, device = device, optimizer = optimizer, num_epoch = num_epoch, model = model, train_loader = train_dataloader, test_loader = valid_dataloader, writer = writer ,log_dir = log_dir, run_log = run_log, save_point_cloud = save_point_cloud)

"""
mainの実行
"""
if __name__ == '__main__':
    main()

