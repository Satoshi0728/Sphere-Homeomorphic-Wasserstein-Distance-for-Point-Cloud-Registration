import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# import losses.normflows_ishikawa as nf
import normflows_ishikawa as nf
import ot
from torch import linalg as LA
import numpy as np


class Cos_disimilarity_W(nn.Module):
    def __init__(self, device , p = 1):
        super(Cos_disimilarity_W, self).__init__()
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        
        """
        self.device = device
        self.p = p
        # self.criteria = ot.emd2
    
    def calcurate_cos_W(self, x, y , device, p = 1):
        C = self.cos_cost_matrix(x, y, p).to(device)
        
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
        if batch_size >= 2:
            for index in range(batch_size):
                losses = torch.pow(ot.emd2(probability_vector_of_x[index], probability_vector_of_y[index], C[index]), 1. / p)
                emd = emd + losses
            emd = emd / int(batch_size)
            return emd
        elif batch_size == 1:
            losses = torch.pow(ot.emd2(probability_vector_of_x, probability_vector_of_y, C), 1. / p)
            return losses
        else:
            raise ValueError("batch_size is not valid")

    def cos_cost_matrix(self, x, y, p=1):
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        output C shape: B x N x M
        p : Lp norm
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = (1 - F.cosine_similarity(x_col, y_lin, dim=-1))**p
        return C

    def forward(self, x, y):
        return self.calcurate_cos_W(x, y, self.device, self.p)



"""
測地線ver
"""
class Geodesic_distance_W(nn.Module):
    def __init__(self, device , p = 1):
        super(Geodesic_distance_W, self).__init__()
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        
        """
        self.device = device
        self.p = p
        # self.criteria = ot.emd2
    
    def calcurate_geodesic_W(self, x, y , device, p = 1):
        C = self.geodesic_cost_matrix(x, y, p).to(device)
        
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
        if batch_size >= 2:
            for index in range(batch_size):
                losses = torch.pow(ot.emd2(probability_vector_of_x[index], probability_vector_of_y[index], C[index]), 1. / p)
                emd = emd + losses
            emd = emd / int(batch_size)
            return emd
        elif batch_size == 1:
            losses = torch.pow(ot.emd2(probability_vector_of_x, probability_vector_of_y, C), 1. / p)
            return losses
        else:
            raise ValueError("batch_size is not valid")

    def geodesic_cost_matrix(self, x, y, p=1):
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        output C shape: B x N x M
        p : Lp norm
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        #calculate geodesic distance
        C = (torch.acos(F.cosine_similarity(x_col, y_lin, dim=-1)))**p
        return C

    def forward(self, x, y):
        return self.calcurate_geodesic_W(x, y, self.device, self.p)




"""
Norm flow
"""
class Norm_Flow_structure(nn.Module):
    def __init__(self, input_dim = 3, flow_name =  "Planar", n_flow_layer = 3):
        super(Norm_Flow_structure, self).__init__()
        self.net = nn.ModuleList(self.create__NF_structure(flow_name, input_dim ,n_flow_layer))

    def create__NF_structure(self, flow_name, input_dim ,n_flow_layer):
        if flow_name == "Planar":
            flows = []
            for i in range(n_flow_layer):
                flows += [nf.flows.Planar((input_dim,))]
            return flows

        elif flow_name == "Residual":
            latent_size = input_dim
            hidden_units = 8
            hidden_layers = 7
            flows = []
            for i in range(n_flow_layer):
                net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                        init_zeros=True, lipschitz_const=0.95)
                flows += [nf.flows.Residual(net, reverse = False, reduce_memory=True)]
                # flows += [nf.flows.ActNorm(latent_size)]
            return flows
        else:
            raise ValueError("Flow name is not valid")

    def forward(self, x):
        for flow in self.net:
            x, _= flow(x)
        return x




"""
optuna用
"""
class Norm_Flow_structure_optuna(nn.Module):
    def __init__(self, input_dim = 3, flow_name =  "Planar", n_flow_layer = 3, Residual_hidden_units = 8, Residual_hidden_layers = 3):
        super(Norm_Flow_structure_optuna, self).__init__()
        self.Residual_hidden_units = Residual_hidden_units
        self.Residual_hidden_layers = Residual_hidden_layers
        self.net = nn.ModuleList(self.create__NF_structure(flow_name, input_dim ,n_flow_layer))

    def create__NF_structure(self, flow_name, input_dim ,n_flow_layer):
        if flow_name == "Planar":
            flows = []
            for i in range(n_flow_layer):
                flows += [nf.flows.Planar((input_dim,))]
            return flows

        elif flow_name == "Residual":
            latent_size = input_dim
            hidden_units = self.Residual_hidden_units
            hidden_layers = self.Residual_hidden_layers
            flows = []
            for i in range(n_flow_layer):
                net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                        init_zeros=True, lipschitz_const=0.95)
                flows += [nf.flows.Residual(net, reverse = False, reduce_memory=True)]
                # flows += [nf.flows.ActNorm(latent_size)]
            return flows
        else:
            raise ValueError("Flow name is not valid")
    def forward(self, x):
        for flow in self.net:
            x, _= flow(x)
        return x






"""
提案手法
"""
class max_cos_disimilarity_wassersten_distance(nn.Module):
    def __init__(self,phi,CSW, device, phi_op,max_iter=10, lam = 0.1, psi_minibatch_size = 5):
        super(max_cos_disimilarity_wassersten_distance, self).__init__()
        self.phi = phi
        self.CSW = CSW
        self.phi_op = phi_op
        self.max_iter = max_iter
        #ミニバッチ化する時に使う
        # self.psi_minibatch_size = psi_minibatch_size
        self.device = device
        #正則化項の係数
        self.reg_lam = lam
        
    def regularization_of_normalizing_flow(self,x):
        """
        ノーマライジングフローの出力を超級面上に分布するように正則化をかける
        inputs x: B x N x D
        output y: scalar
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return torch.sum(torch.abs(LA.vector_norm(x, dim=-1) - 1))

    def forward(self, first_samples,second_samples, train_or_test = "train"):
        first_samples_detach= first_samples.detach()
        second_samples_detach = second_samples.detach()

        #train
        if train_or_test == "train":
            self.phi.train()
            for _ in range(self.max_iter):
                self.phi_op.zero_grad()
                first_samples_transform =self.phi(first_samples_detach)
                second_samples_transform = self.phi(second_samples_detach)
                # calcurate cos_similarity_wassersten_distance
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                # # calcurate regularization_of_normalizing_flow
                reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) / (first_samples_transform.shape[0]*first_samples_transform.shape[1])
                reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform) / (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                regularization = self.reg_lam * (reg_first_samples + reg_second_samples)
                #gradient ascent
                loss= regularization - cswd
                loss.backward(retain_graph=True)
                self.phi_op.step()
        #test
        elif train_or_test == "test":
            self.phi.eval()

        first_samples_transform =self.phi(first_samples)
        second_samples_transform= self.phi(second_samples)
        cswd = self.CSW(first_samples_transform, second_samples_transform)
        return  cswd, first_samples_transform, second_samples_transform











"""
提案手法 Maxを変えたバージョン
"""
class pseudo_max_cos_disimilarity_wassersten_distance(nn.Module):
    def __init__(self, CSW, device, phi_num=10, lam = 0.1, n_flow_layer = 5, flow_name = "Residual", mean_or_max_or_softmax = "max"):
        super(pseudo_max_cos_disimilarity_wassersten_distance, self).__init__()
        self.CSW = CSW
        self.phi_num = phi_num
        self.n_flow_layer = n_flow_layer
        self.device = device
        self.flow_name = flow_name
        #正則化項の係数
        self.reg_lam = lam
        self.phi_list = self.norm_flow(self.phi_num, self.flow_name, self.n_flow_layer)
        self.mean_or_max_or_softmax = mean_or_max_or_softmax

    # def regularization_of_normalizing_flow(self,x):
    #     """
    #     ノーマライジングフローの出力を超級面上に分布するように正則化をかける
    #     inputs x: B x N x D
    #     output y: scalar
    #     """
    #     if x.dim() == 2:
    #         x = x.unsqueeze(0)
    #     return torch.sum(torch.abs(LA.vector_norm(x, dim=-1) - 1))

    def norm_flow(self, phi_num = 10, flow_name =  "Residual", n_flow_layer = 3):
        phi_list = []
        for i in range(phi_num):
            phi = Norm_Flow_structure(flow_name = flow_name , n_flow_layer = n_flow_layer)
            phi = phi.to(self.device)
            phi_list.append(phi)
        return phi_list


    def forward(self, first_samples,second_samples):
        first_samples_detach= first_samples.detach()
        second_samples_detach = second_samples.detach()
        
        if self.mean_or_max_or_softmax == "max":
            max_cswd = -1
            for phi in self.phi_list:
                first_samples_transform = phi(first_samples_detach)
                second_samples_transform = phi(second_samples_detach)
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                if cswd > max_cswd:
                    max_cswd = cswd
            print(max_cswd.shape)
            return max_cswd, first_samples_transform, second_samples_transform

        elif self.mean_or_max_or_softmax == "mean":
            mean_cswd = 0
            for phi in self.phi_list:
                first_samples_transform = phi(first_samples_detach)
                second_samples_transform = phi(second_samples_detach)
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                mean_cswd += cswd
            mean_cswd = mean_cswd / self.phi_num
            return mean_cswd, first_samples_transform, second_samples_transform

        elif self.mean_or_max_or_softmax == "softmax":
            softmax_cswd = torch.zeros(len(self.phi_list))
            softmax_input = torch.tensor(self.phi_list)
            for i in range(len(self.phi_list)):
                phi = self.phi_list[i]
                first_samples_transform = phi(first_samples_detach)
                second_samples_transform = phi(second_samples_detach)
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                softmax_cswd[i] = cswd
            softmax_cswd = F.softmax(softmax_cswd, dim=0)
            softmax_cswd_output = softmax_cswd @ softmax_input
            return softmax_cswd_output, first_samples_transform, second_samples_transform

        else:
            raise ValueError("mean_or_max_or_softmax is not valid")





# """
# シード固定
# """
# def fix_seed(seed):
#     # Numpy
#     np.random.seed(seed)
#     # Pytorch
#     torch.manual_seed(seed+ 1)
#     torch.cuda.manual_seed_all(seed + 2)
#     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    """
    動作チェック
    """
    SEED = 111
    # fix_seed(SEED)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    
    # inputs1 = F.normalize(torch.randn((100,128,3)).to(device), p=2, dim=-1)
    # inputs2 = F.normalize(torch.randn((100,128,3)).to(device), p=2, dim=-1)
    inputs1 = torch.randn((2,512,3)).to(device)
    inputs2 = torch.randn((2,512,3)).to(device)
    flow_name = ["Planar", "Residual"]
    lamda_of_regularization = 1

    ###############################################################################################
    # phi = Norm_Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
    n_flow_layer = 5
    mean_or_max_or_softmax = "max"
    phi_num=20
    
    
    CSW =  Cos_disimilarity_W(device = device, p = 2)
    loss = pseudo_max_cos_disimilarity_wassersten_distance(CSW = CSW, device= device, phi_num = phi_num, lam = lamda_of_regularization,  n_flow_layer = n_flow_layer, mean_or_max_or_softmax = mean_or_max_or_softmax)

    time_sta = time.time()
    #train
    for i in range(1):
        ssw, output1, output2 = loss(inputs1, inputs2)
        # print("loss", ssw.item())
        # print("output1", output1)
        # print("output2", output2)
    #test
    # print("test")
    # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    time_end = time.time()
    tim = time_end- time_sta
    print("loss", ssw.item())
    print("time", tim)
    ###############################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ###############################################################################################
    # phi = Norm_Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
    # #コンパイル
    # # phi = torch.compile(phi)
    # phi = phi.to(device)

    # phi_op = optim.Adam(phi.parameters(), lr=1e-2, betas=(0.5, 0.999))
    # CSW =  Geodesic_distance_W(device = device, p = 2)
    # loss = max_cos_disimilarity_wassersten_distance(phi = phi, CSW = CSW, phi_op = phi_op, lam = lamda_of_regularization, max_iter=1, device= device)

    # time_sta = time.time()
    # #train
    # for i in range(10):
    #     ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
    #     # print("loss", ssw.item())
    #     # print("output1", output1)
    #     # print("output2", output2)
    # #test
    # # print("test")
    # # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    # time_end = time.time()
    # tim = time_end- time_sta
    # # print("loss", ssw.item())
    # print("time", tim)
    # ###############################################################################################



    # SEED = 1111
    # fix_seed(SEED)
    # device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # # inputs1 = F.normalize(torch.randn((100,256,3)).to(device), p=2, dim=-1)
    # # inputs2 = F.normalize(torch.randn((100,256,3)).to(device), p=2, dim=-1)
    # inputs1 = torch.randn((2,512,3)).to(device)
    # inputs2 = torch.randn((2,512,3)).to(device)
    # flow_name = ["Planar", "Residual"]
    # lamda_of_regularization = 1

    # phi = Norm_Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
    # #コンパイル
    # # phi = torch.compile(phi)
    # phi = phi.to(device)

    # phi_op = optim.Adam(phi.parameters(), lr=100, betas=(0.5, 0.999))
    # CSW =  Cos_disimilarity_W(device = device, p = 2)
    # loss = max_cos_disimilarity_wassersten_distance(phi = phi, CSW = CSW, phi_op = phi_op, lam = lamda_of_regularization, max_iter=1, device= device)


    # # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
    # # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    # time_sta = time.time()
    # #train
    # for i in range(10):
    #     ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
    #     # print("loss", ssw.item())
    #     # print("output1", output1)
    #     # print("output2", output2)
    # #test
    # # print("test")
    # # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    # time_end = time.time()
    # tim = time_end- time_sta
    # # print("loss", ssw.item())
    # print("time", tim)












































# if __name__ == "__main__":

#     """
#     動作チェック
#     """
#     device = "cuda:6" if torch.cuda.is_available() else "cpu"
#     inputs1 = F.normalize(torch.randn((32, 512,3)).to(device), p=2, dim=-1)
#     inputs2 = F.normalize(torch.randn((32, 512,3)).to(device), p=2, dim=-1)
#     flow_name = ["Planar", "Residual"]
#     lamda_of_regularization = 1

#     #球面からの距離がthresholdより大きいときは、正則化項のみでロスを計算。そうでないときは、CSW+ 正則化項でロスを計算
#     customize_loss_threshold = 0.1

#     psi = Norm_Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
#     #コンパイル
#     # psi = torch.compile(psi)
#     psi = psi.to(device)

#     psi_op = optim.Adam(psi.parameters(), lr=1e-3, betas=(0.5, 0.999))
#     CSW =  Cos_disimilarity_W(device = device, p = 1)

#     loss = max_cos_disimilarity_wassersten_distance(phi = psi, CSW = CSW, phi_op = psi_op, lam = lamda_of_regularization, customize_loss_threshold = customize_loss_threshold, max_iter=50, device= device)

#     time_sta = time.time()
#     ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
#     # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
#     time_end = time.time()
#     tim = time_end- time_sta
#     # print(ssw.item())
#     print("time", tim)




# """
# early stop
# """
# class max_cos_similarity_wassersten_distance_early_stop(nn.Module):
#     def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, psi_minibatch_size = 5, device='cuda'):
#         super(max_cos_similarity_wassersten_distance_early_stop, self).__init__()
#         self.phi = phi
#         self.CSW = CSW
#         self.phi_op = phi_op
#         self.max_iter = max_iter
#         #ミニバッチ化する時に使う
#         # self.psi_minibatch_size = psi_minibatch_size
#         self.device = device
#         #正則化項の係数
#         self.lam = lam
#         self.customize_loss_threshold = customize_loss_threshold
#         self.custom_loss_count = 0
        
#     def regularization_of_normalizing_flow(self,x):
#         """
#         ノーマライジングフローの出力を超級面上に分布するように正則化をかける
#         inputs x: B x N x D
#         output y: scalar
#         """
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         return torch.sum(torch.abs(LA.vector_norm(x, dim=-1) - 1))

#     def forward(self, first_samples,second_samples, early_stop_cnt = 0, train_or_test = "train"):
#         first_samples_detach= first_samples.detach()
#         second_samples_detach = second_samples.detach()

#         #train
#         if train_or_test == "train":
#             self.phi.train()
#             if early_stop_cnt <= 3:
#                 reg_lam = self.lam
#                 for _ in range(self.max_iter):
#                     self.phi_op.zero_grad()

#                     first_samples_transform =self.phi(first_samples_detach)
#                     second_samples_transform = self.phi(second_samples_detach)
#                     # calcurate cos_similarity_wassersten_distance
#                     cswd = self.CSW(first_samples_transform, second_samples_transform)
#                     # # calcurate regularization_of_normalizing_flow
#                     reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                     reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#                     regularization = reg_lam * (reg_first_samples + reg_second_samples)

#                     #gradient ascent
#                     loss= regularization - cswd
#                     loss.backward(retain_graph=True)
#                     self.phi_op.step()
#                 #更新のたびに、正則化項の係数を小さくする
#                 self.lam = self.lam * 0.999
#         #test
#         elif train_or_test == "test":
#             self.phi.eval()

#         first_samples_transform =self.phi(first_samples)
#         second_samples_transform= self.phi(second_samples)
#         cswd = self.CSW(first_samples_transform, second_samples_transform)

#         return  cswd, first_samples_transform, second_samples_transform

