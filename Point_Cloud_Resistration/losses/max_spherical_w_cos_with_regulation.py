"""
コサイン非類似度を使ったW距離の最大化
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import losses.normflows_ishikawa as nf
# import normflows_ishikawa as nf
import ot
from torch import linalg as LA

# class Cos_similarity_W:
#     def __init__(self, device , p = 1):
#         """
#         input source shape: B x N x D(= 3)
#         input target shape B x M x D(= 3)
        
#         """
#         self.device = device
#         self.p = p
#         self.criteria = ot.emd2
    
#     def calcurate_W(self, x, y , device, p = 1):
#         C = Cos_similarity_W.cost_matrix(x, y, p).to(device)
#         x_points = x.shape[-2]
#         y_points = y.shape[-2]

#         if x.dim() == 2:
#             batch_size = 1
#         else:
#             batch_size = x.shape[0]

#         # probability vectors (経験分布なら 1/ sample size)
#         probability_vector_of_x = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
#         probability_vector_of_y = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

#         emd = 0
#         if batch_size >= 2:
#             for index in range(batch_size):
#                 loss = self.criteria(probability_vector_of_x[index], probability_vector_of_y[index], C[index])
#                 emd += loss
#             emd = emd / int(batch_size)
#             return emd
#         elif batch_size == 1:
#             loss = self.criteria(probability_vector_of_x, probability_vector_of_y, C)
#             return loss
#         else:
#             raise ValueError("batch_size is not valid")

#     @staticmethod
#     def cost_matrix(x, y, p=1):
#         """
#         input source shape: B x N x D(= 3)
#         input target shape B x M x D(= 3)
#         output C shape: B x N x M
#         p : Lp norm
#         """
#         x_col = x.unsqueeze(-2)
#         y_lin = y.unsqueeze(-3)
#         C = torch.pow((1 - F.cosine_similarity(x_col, y_lin, dim=-1)) ** p, 1. / p)
#         return C

#     def __call__(self, x, y):
#         return self.calcurate_W(x, y, self.device, self.p)



# class Flow_structure(nn.Module):
#     def __init__(self, input_dim = 3, flow_name =  "Planar", n_flow_layer = 3):
#         super(Flow_structure, self).__init__()
#         self.net = nn.ModuleList(self.create__NF_structure(flow_name, input_dim ,n_flow_layer))

#     def create__NF_structure(self, flow_name, input_dim ,n_flow_layer):
#         if flow_name == "Planar":
#             flows = []
#             for i in range(n_flow_layer):
#                 flows += [nf.flows.Planar((input_dim,))]
#             return flows

#         elif flow_name == "Residual":
#             latent_size = input_dim
#             hidden_units = 64
#             hidden_layers = 3

#             flows = []
#             for i in range(n_flow_layer):
#                 net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
#                                         init_zeros=True, lipschitz_const=0.9)
#                 # flows += [nf.flows.Residual(net,reverse= False, reduce_memory=True)]
#                 flows += [nf.flows.Residual(net, reverse = False, reduce_memory=True)]
#                 # if i < n_flow_layer - 1:
#                 flows += [nf.flows.ActNorm(latent_size)]
#             return flows
#         else:
#             raise ValueError("Flow name is not valid")


#     def forward(self, x):
#         for flow in self.net:
#             x, _= flow(x)
#         return x





# class max_cos_similarity_wassersten_distance(nn.Module):
#     def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, device='cuda'):
#         super(max_cos_similarity_wassersten_distance, self).__init__()
#         self.phi = phi
#         self.CSW = CSW
#         self.phi_op = phi_op
#         self.max_iter = max_iter
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


#     def nan_check(self, reg_samples, first_samples_transform, second_samples_transform, phi):
#         """
#         nanが発生したら、異常終了させる
#         """
#         if torch.isnan(reg_samples) or torch.isnan(first_samples_transform.any()) or torch.isnan(second_samples_transform.any()):
#             print("reg_samples",reg_samples)
#             print("first_samples_transform",first_samples_transform)
#             print("second_samples_transform",second_samples_transform)
#             print("phi",phi.parameters())
#             raise ValueError("nanが発生したので、異常終了します。")


#     def customize_loss(self, reg_lam, reg_first_samples, reg_second_samples, cswd):
#         """
#         球面からの距離が大きいと時は、正則化項のみでロスを計算。そうでないときは、CSW+ 正則化項でロスを計算
#         """
#         if (reg_first_samples + reg_second_samples) > self.customize_loss_threshold:
#             print("でかい")
#             return reg_lam * (reg_first_samples + reg_second_samples)
#         else:
#             self.custom_loss_count += 1
#             return reg_lam * (reg_first_samples + reg_second_samples) - cswd



#     def forward(self, first_samples,second_samples, train_or_test = "train"):
#         first_samples_detach= first_samples.detach()
#         second_samples_detach = second_samples.detach()

#         #train
#         if train_or_test == "train":
#             reg_lam = self.lam
#             for _ in range(self.max_iter):
#                 print("reg_lam", reg_lam)
#                 first_samples_transform =self.phi(first_samples_detach)
#                 second_samples_transform = self.phi(second_samples_detach)

#                 # calcurate cos_similarity_wassersten_distance
#                 cswd = self.CSW(first_samples_transform, second_samples_transform)
#                 # calcurate regularization_of_normalizing_flow
#                 reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                 print("reg_first_samples",reg_first_samples.item())
#                 reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#                 print("reg_second_samples",reg_second_samples.item())
                
#                 #例外処理　nanが出たら、プリント
#                 self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(), phi = self.phi)

#                 # #十分に収束したら、計算しない
#                 # if (reg_first_samples + reg_second_samples) < 1e-4:
#                 #     break

#                 # gradient ascent
#                 # loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd
#                 loss= self.customize_loss(reg_lam, reg_first_samples, reg_second_samples, cswd)

#                 # loss = -cswd
#                 self.phi_op.zero_grad()
#                 loss.backward(retain_graph=True)
#                 self.phi_op.step()

#                 # if (_ % 20 == 0 ) and (_ != 0):
#                 #     reg_lam = reg_lam * 0.9
#                 print("loss", cswd.item())
            
            
#             #もし、cswdの意味での分離回数が少なかったら、(reg_first_samples + reg_second_samples) - cswdを最後に計算する
#             if self.custom_loss_count < 20:
#                 print("calcurate reg_lam * (reg_first_samples + reg_second_samples) - cswd for 10 times")
#                 for _ in range(20):
#                     first_samples_transform =self.phi(first_samples_detach)
#                     second_samples_transform = self.phi(second_samples_detach)

#                     # calcurate cos_similarity_wassersten_distance
#                     cswd = self.CSW(first_samples_transform, second_samples_transform)
#                     # calcurate regularization_of_normalizing_flow
#                     reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                     print("reg_first_samples",reg_first_samples.item())
#                     reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#                     print("reg_second_samples",reg_second_samples.item())
#                     #例外処理　nanが出たら、プリント
#                     self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(), phi = self.phi)

#                     #gradient ascent
#                     loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd

#                     self.phi_op.zero_grad()
#                     loss.backward(retain_graph=True)
#                     self.phi_op.step()
#                     print("loss", cswd.item())

#         #test
#         elif train_or_test == "test":
#             pass
        
#         #更新のたびに、正則化項の係数を小さくする
#         # self.lam = self.lam * 0.8

#         first_samples_transform =self.phi(first_samples)
#         second_samples_transform= self.phi(second_samples)
#         cswd = self.CSW(first_samples_transform, second_samples_transform)

#         # print(cswd.item())
#         return  cswd, first_samples_transform, second_samples_transform








# class max_cos_similarity_wassersten_distance(nn.Module):
#     def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, device='cuda'):
#         super(max_cos_similarity_wassersten_distance, self).__init__()
#         self.phi = phi
#         self.CSW = CSW
#         self.phi_op = phi_op
#         self.max_iter = max_iter
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


#     def nan_check(self, reg_samples, first_samples_transform, second_samples_transform, first_samples_detach, second_samples_detach,phi):
#         """
#         nanが発生したら、異常終了させる
#         """
#         if torch.isnan(reg_samples) or torch.isnan(first_samples_transform.any()) or torch.isnan(second_samples_transform.any()):
#             print("reg_samples",reg_samples)
#             print("first_samples_transform",first_samples_transform)
#             print("second_samples_transform",second_samples_transform)
#             print("first_samples_detach",first_samples_detach)
#             print("second_samples_detach",second_samples_detach)
#             print("phi",phi)
#             raise ValueError("nanが発生したので、異常終了します。")


#     def customize_loss(self, reg_lam, reg_first_samples, reg_second_samples, cswd):
#         """
#         球面からの距離が大きいと時は、正則化項のみでロスを計算。そうでないときは、CSW+ 正則化項でロスを計算
#         """
#         if (reg_first_samples + reg_second_samples) > self.customize_loss_threshold:
#             print("でかい")
#             return reg_lam * (reg_first_samples + reg_second_samples)
#         else:
#             self.custom_loss_count += 1
#             return reg_lam * (reg_first_samples + reg_second_samples) - cswd



#     def forward(self, first_samples,second_samples, train_or_test = "train"):
#         first_samples_detach= first_samples.detach()
#         second_samples_detach = second_samples.detach()

#         #train
#         if train_or_test == "train":
#             reg_lam = self.lam
#             for _ in range(self.max_iter):
#                 print("reg_lam", reg_lam)
#                 first_samples_transform =self.phi(first_samples_detach)
#                 second_samples_transform = self.phi(second_samples_detach)

#                 # calcurate cos_similarity_wassersten_distance
#                 cswd = self.CSW(first_samples_transform, second_samples_transform)
#                 # calcurate regularization_of_normalizing_flow
#                 reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                 print("reg_first_samples",reg_first_samples.item())
#                 reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#                 print("reg_second_samples",reg_second_samples.item())
                
#                 #例外処理　nanが出たら、プリント
#                 self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(),first_samples_detach, second_samples_detach, phi = self.phi)

#                 # #十分に収束したら、計算しない
#                 # if (reg_first_samples + reg_second_samples) < 1e-4:
#                 #     break

#                 # gradient ascent
#                 # loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd
#                 # loss= self.customize_loss(reg_lam, reg_first_samples, reg_second_samples, cswd)
#                 loss = -cswd
                
                
                
#                 self.phi_op.zero_grad()
#                 loss.backward(retain_graph=True)
#                 self.phi_op.step()

#                 # if (_ % 20 == 0 ) and (_ != 0):
#                 #     reg_lam = reg_lam * 0.9
#                 print("loss", cswd.item())
            
            
#             # #もし、cswdの意味での分離回数が少なかったら、(reg_first_samples + reg_second_samples) - cswdを最後に計算する
#             # if self.custom_loss_count < 20:
#             #     print("calcurate reg_lam * (reg_first_samples + reg_second_samples) - cswd for 10 times")
#             #     for _ in range(20):
#             #         first_samples_transform =self.phi(first_samples_detach)
#             #         second_samples_transform = self.phi(second_samples_detach)

#             #         # calcurate cos_similarity_wassersten_distance
#             #         cswd = self.CSW(first_samples_transform, second_samples_transform)
#             #         # calcurate regularization_of_normalizing_flow
#             #         reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#             #         print("reg_first_samples",reg_first_samples.item())
#             #         reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#             #         print("reg_second_samples",reg_second_samples.item())
#             #         #例外処理　nanが出たら、プリント
#             #         self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(), first_samples_detach, second_samples_detach, phi = self.phi)

#             #         #gradient ascent
#             #         loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd

#             #         self.phi_op.zero_grad()
#             #         loss.backward(retain_graph=True)
#             #         self.phi_op.step()
#             #         print("loss", cswd.item())

#         #test
#         elif train_or_test == "test":
#             pass
        
#         #更新のたびに、正則化項の係数を小さくする
#         # self.lam = self.lam * 0.8

#         first_samples_transform =self.phi(first_samples)
#         second_samples_transform= self.phi(second_samples)
#         cswd = self.CSW(first_samples_transform, second_samples_transform)

#         # print(cswd.item())
#         return  cswd, first_samples_transform, second_samples_transform







"""
デバッグ用
"""
# class Cos_similarity_W:
#     def __init__(self, device , p = 1):
#         """
#         input source shape: B x N x D(= 3)
#         input target shape B x M x D(= 3)
        
#         """
#         self.device = device
#         self.p = p
#         self.criteria = ot.emd2
    
#     def calcurate_W(self, x, y , device, p = 1):
#         C = Cos_similarity_W.cost_matrix(x, y, p).to(device)
#         x_points = x.shape[-2]
#         y_points = y.shape[-2]

#         if x.dim() == 2:
#             batch_size = 1
#         else:
#             batch_size = x.shape[0]

#         # probability vectors (経験分布なら 1/ sample size)
#         probability_vector_of_x = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
#         probability_vector_of_y = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

#         emd = 0
#         if batch_size >= 2:
#             for index in range(batch_size):
#                 loss = self.criteria(probability_vector_of_x[index], probability_vector_of_y[index], C[index])
#                 #error
#                 if torch.isnan(loss).any():
#                     torch.save(loss, "emd_loss.pth")
#                     torch.save(probability_vector_of_x[index], "emd_probability_vector_of_x.pth")
#                     torch.save(probability_vector_of_y[index], "emd_probability_vector_of_y.pth")
#                     torch.save(C[index], "emd_C.pth")
#                     raise ValueError("lossにnanが発生したので、異常終了します。")
#                 emd += loss
#             emd = emd / int(batch_size)
#             return emd
#         elif batch_size == 1:
#             loss = self.criteria(probability_vector_of_x, probability_vector_of_y, C)
#             return loss
#         else:
#             raise ValueError("batch_size is not valid")

#     @staticmethod
#     def cost_matrix(x, y, p=1):
#         """
#         input source shape: B x N x D(= 3)
#         input target shape B x M x D(= 3)
#         output C shape: B x N x M
#         p : Lp norm
#         """
#         x_col = x.unsqueeze(-2)
#         y_lin = y.unsqueeze(-3)
#         C = torch.pow((1 - F.cosine_similarity(x_col, y_lin, dim=-1)) ** p, 1. / p)
#         #error
#         if torch.isnan(C).any():
#             torch.save(x_col, "x_col.pth")
#             torch.save(y_lin, "y_lin.pth")
#             torch.save(C, "C.pth")
#             print("x_col", x_col)
#             print("y_lin", y_lin)
#             print("C", C)
#             raise ValueError("Cにnanが発生したので、異常終了します。")
#         return C

#     def __call__(self, x, y):
#         return self.calcurate_W(x, y, self.device, self.p)



# class Flow_structure(nn.Module):
#     def __init__(self, input_dim = 3, flow_name =  "Planar", n_flow_layer = 3):
#         super(Flow_structure, self).__init__()
#         self.net = nn.ModuleList(self.create__NF_structure(flow_name, input_dim ,n_flow_layer))

#     def create__NF_structure(self, flow_name, input_dim ,n_flow_layer):
#         if flow_name == "Planar":
#             flows = []
#             for i in range(n_flow_layer):
#                 flows += [nf.flows.Planar((input_dim,))]
#             return flows

#         elif flow_name == "Residual":
#             latent_size = input_dim
#             hidden_units = 64
#             hidden_layers = 3

#             flows = []
#             for i in range(n_flow_layer):
#                 net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
#                                         init_zeros=True, lipschitz_const=0.9)
#                 # flows += [nf.flows.Residual(net,reverse= False, reduce_memory=True)]
#                 flows += [nf.flows.Residual(net, reverse = False, reduce_memory=True)]
#                 # if i < n_flow_layer - 1:
#                 flows += [nf.flows.ActNorm(latent_size)]
#             return flows
#         else:
#             raise ValueError("Flow name is not valid")


#     def forward(self, x):
#         for flow in self.net:
#             x, _= flow(x)
#         return x




"""
デバッグ用
"""
# class max_cos_similarity_wassersten_distance(nn.Module):
#     def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, device='cuda'):
#         super(max_cos_similarity_wassersten_distance, self).__init__()
#         self.phi = phi
#         self.CSW = CSW
#         self.phi_op = phi_op
#         self.max_iter = max_iter
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
        
        
#         if torch.isnan(torch.abs(LA.vector_norm(x, dim=-1) - 1).any()):
#                 ValueError("torch.abs(LA.vector_norm(x, dim=-1) - 1).any() でnanが発生したので、異常終了します。")
        
#         return torch.sum(torch.abs(LA.vector_norm(x, dim=-1) - 1))


#     def nan_check(self, reg_samples, first_samples_transform, second_samples_transform, first_samples_detach, second_samples_detach,phi, phi_debag):
#         """
#         nanが発生したら、異常終了させる
#         """
#         if torch.isnan(reg_samples) or torch.isnan(first_samples_transform.any()) or torch.isnan(second_samples_transform.any()):
#             print("reg_samples",reg_samples)
#             print("first_samples_transform",first_samples_transform)
#             print("second_samples_transform",second_samples_transform)
#             print("first_samples_detach",first_samples_detach)
#             print("second_samples_detach",second_samples_detach)
#             print("phi",phi)
#             torch.save(phi , "phi_model4.pth")
#             torch.save(phi.state_dict(), "phi4.pth")
#             #save first_samples_transform
#             torch.save(first_samples_transform, "first_samples_transform4.pth")
#             #save second_samples_transform
#             torch.save(second_samples_transform, "second_samples_transform4.pth")
#             #save first_samples_detach
#             torch.save(first_samples_detach, "first_samples_detach4.pth")
#             #save second_samples_detach
#             torch.save(second_samples_detach, "second_samples_detach4.pth")
#             #save phi_debag
#             torch.save(phi_debag, "phi_debag4.pth")
            
#             raise ValueError("nanが発生したので、異常終了します。")


#     def customize_loss(self, reg_lam, reg_first_samples, reg_second_samples, cswd):
#         """
#         球面からの距離が大きいと時は、正則化項のみでロスを計算。そうでないときは、CSW+ 正則化項でロスを計算
#         """
#         if (reg_first_samples + reg_second_samples) > self.customize_loss_threshold:
#             print("でかい")
#             return reg_lam * (reg_first_samples + reg_second_samples)
#         else:
#             self.custom_loss_count += 1
#             return reg_lam * (reg_first_samples + reg_second_samples) - cswd



#     def forward(self, first_samples,second_samples, train_or_test = "train"):
#         first_samples_detach= first_samples.detach()
#         second_samples_detach = second_samples.detach()

#         #train
#         if train_or_test == "train":
#             self.phi.train()
#             self.phi_op.zero_grad()
#             #デバッグ用
#             phi_debag = self.phi.state_dict()

#             reg_lam = self.lam
#             for _ in range(self.max_iter):
#                 # print("reg_lam", reg_lam)
#                 with autograd.detect_anomaly():
#                     first_samples_transform =self.phi(first_samples_detach)
#                     second_samples_transform = self.phi(second_samples_detach)
                    
#                     reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                     # print("reg_first_samples",reg_first_samples.item())
#                     reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                    
#                     #例外処理　nanが出たら、プリント
#                     self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(),first_samples_detach, second_samples_detach, phi = self.phi, phi_debag = phi_debag)

#                     # calcurate cos_similarity_wassersten_distance
#                     cswd = self.CSW(first_samples_transform, second_samples_transform)
#                     # calcurate regularization_of_normalizing_flow
#                     # print("reg_second_samples",reg_second_samples.item())




#                     reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#                     # print("reg_first_samples",reg_first_samples.item())
#                     reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                    
#                     #例外処理　nanが出たら、プリント
#                     self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(),first_samples_detach, second_samples_detach, phi = self.phi, phi_debag = phi_debag)




#                     # #十分に収束したら、計算しない
#                     # if (reg_first_samples + reg_second_samples) < 1e-4:
#                     #     break

#                     # gradient ascent
#                     # loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd
#                     # loss= self.customize_loss(reg_lam, reg_first_samples, reg_second_samples, cswd)
#                     loss = -cswd
#                     loss.backward(retain_graph=True)
                    

#                 #デバッグ用
#                 phi_debag = self.phi.state_dict()

#                 self.phi_op.step()

#                 # if (_ % 20 == 0 ) and (_ != 0):
#                 #     reg_lam = reg_lam * 0.9
#                 print("cswd loss", cswd.item())
            
            
#             # #もし、cswdの意味での分離回数が少なかったら、(reg_first_samples + reg_second_samples) - cswdを最後に計算する
#             # if self.custom_loss_count < 20:
#             #     print("calcurate reg_lam * (reg_first_samples + reg_second_samples) - cswd for 10 times")
#             #     for _ in range(20):
#             #         first_samples_transform =self.phi(first_samples_detach)
#             #         second_samples_transform = self.phi(second_samples_detach)

#             #         # calcurate cos_similarity_wassersten_distance
#             #         cswd = self.CSW(first_samples_transform, second_samples_transform)
#             #         # calcurate regularization_of_normalizing_flow
#             #         reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
#             #         print("reg_first_samples",reg_first_samples.item())
#             #         reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
#             #         print("reg_second_samples",reg_second_samples.item())
#             #         #例外処理　nanが出たら、プリント
#             #         self.nan_check(reg_first_samples.detach(), first_samples_transform.detach(), second_samples_transform.detach(), first_samples_detach, second_samples_detach, phi = self.phi)

#             #         #gradient ascent
#             #         loss= reg_lam * (reg_first_samples + reg_second_samples) - cswd

#             #         self.phi_op.zero_grad()
#             #         loss.backward(retain_graph=True)
#             #         self.phi_op.step()
#             #         print("loss", cswd.item())

#         #test
#         elif train_or_test == "test":
#             self.phi.eval()

#         first_samples_transform =self.phi(first_samples)
#         second_samples_transform= self.phi(second_samples)
#         cswd = self.CSW(first_samples_transform, second_samples_transform)

#         return  cswd, first_samples_transform, second_samples_transform






























class Cos_similarity_W(nn.Module):
    def __init__(self, device , p = 1):
        super(Cos_similarity_W, self).__init__()
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        
        """
        self.device = device
        self.p = p
        self.criteria = ot.emd2
    
    def calcurate_W(self, x, y , device, p = 1):
        C = self.cost_matrix(x, y, p).to(device)
        
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
                losses = torch.pow(self.criteria(probability_vector_of_x[index], probability_vector_of_y[index], C[index]), 1. / p)
                emd = emd + losses
            emd = emd / int(batch_size)
            return emd
        elif batch_size == 1:
            losses = torch.pow(self.criteria(probability_vector_of_x, probability_vector_of_y, C), 1. / p)
            return losses
        else:
            raise ValueError("batch_size is not valid")

    def cost_matrix(self, x, y, p=1):
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        output C shape: B x N x M
        p : Lp norm
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = (1 - F.cosine_similarity(x_col, y_lin, dim=-1))**p
        # C = 1 - F.cosine_similarity(x_col, y_lin, dim=-1)
        # C = torch.pow((1 - F.cosine_similarity(x_col, y_lin, dim=-1)) ** p, 1. / p)
        return C

    def forward(self, x, y):
        return self.calcurate_W(x, y, self.device, self.p)





class Flow_structure(nn.Module):
    def __init__(self, input_dim = 3, flow_name =  "Planar", n_flow_layer = 3):
        super(Flow_structure, self).__init__()
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
            hidden_layers = 3

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
early stop
"""
class max_cos_similarity_wassersten_distance_early_stop(nn.Module):
    def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, psi_minibatch_size = 5, device='cuda'):
        super(max_cos_similarity_wassersten_distance_early_stop, self).__init__()
        self.phi = phi
        self.CSW = CSW
        self.phi_op = phi_op
        self.max_iter = max_iter
        #ミニバッチ化する時に使う
        # self.psi_minibatch_size = psi_minibatch_size
        self.device = device
        #正則化項の係数
        self.lam = lam
        self.customize_loss_threshold = customize_loss_threshold
        self.custom_loss_count = 0
        
    def regularization_of_normalizing_flow(self,x):
        """
        ノーマライジングフローの出力を超級面上に分布するように正則化をかける
        inputs x: B x N x D
        output y: scalar
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return torch.sum(torch.abs(LA.vector_norm(x, dim=-1) - 1))

    def forward(self, first_samples,second_samples, early_stop_cnt = 0, train_or_test = "train"):
        first_samples_detach= first_samples.detach()
        second_samples_detach = second_samples.detach()

        #train
        if train_or_test == "train":
            self.phi.train()
            if early_stop_cnt <= 3:
                reg_lam = self.lam
                for _ in range(self.max_iter):
                    self.phi_op.zero_grad()

                    first_samples_transform =self.phi(first_samples_detach)
                    second_samples_transform = self.phi(second_samples_detach)
                    # calcurate cos_similarity_wassersten_distance
                    cswd = self.CSW(first_samples_transform, second_samples_transform)
                    # # calcurate regularization_of_normalizing_flow
                    reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
                    reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                    regularization = reg_lam * (reg_first_samples + reg_second_samples)

                    #gradient ascent
                    loss= regularization - cswd
                    loss.backward(retain_graph=True)
                    self.phi_op.step()
                #更新のたびに、正則化項の係数を小さくする
                self.lam = self.lam * 0.999
        #test
        elif train_or_test == "test":
            self.phi.eval()

        first_samples_transform =self.phi(first_samples)
        second_samples_transform= self.phi(second_samples)
        cswd = self.CSW(first_samples_transform, second_samples_transform)

        return  cswd, first_samples_transform, second_samples_transform




"""
普通のやつ
"""
class max_cos_similarity_wassersten_distance(nn.Module):
    def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, psi_minibatch_size = 5, device='cuda'):
        super(max_cos_similarity_wassersten_distance, self).__init__()
        self.phi = phi
        self.CSW = CSW
        self.phi_op = phi_op
        self.max_iter = max_iter
        #ミニバッチ化する時に使う
        # self.psi_minibatch_size = psi_minibatch_size
        self.device = device
        #正則化項の係数
        self.lam = lam
        self.customize_loss_threshold = customize_loss_threshold
        self.custom_loss_count = 0
        
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
            reg_lam = self.lam
            for _ in range(self.max_iter):
                self.phi_op.zero_grad()

                first_samples_transform =self.phi(first_samples_detach)
                second_samples_transform = self.phi(second_samples_detach)
                # calcurate cos_similarity_wassersten_distance
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                # # calcurate regularization_of_normalizing_flow
                reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
                reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                regularization = reg_lam * (reg_first_samples + reg_second_samples)

                #gradient ascent
                loss= regularization - cswd
                loss.backward(retain_graph=True)
                self.phi_op.step()
                #更新のたびに、正則化項の係数を小さくする
            self.lam = self.lam * 0.999
        #test
        elif train_or_test == "test":
            self.phi.eval()

        first_samples_transform =self.phi(first_samples)
        second_samples_transform= self.phi(second_samples)
        cswd = self.CSW(first_samples_transform, second_samples_transform)

        return  cswd, first_samples_transform, second_samples_transform





"""
phiを毎回初期化してみる
"""
class max_cos_similarity_wassersten_distance_refresh(nn.Module):
    def __init__(self,phi,CSW, phi_op,max_iter=10, lam = 0.1,customize_loss_threshold = 0.3, psi_minibatch_size = 5, device='cuda'):
        super(max_cos_similarity_wassersten_distance_refresh, self).__init__()
        self.phi = phi
        self.CSW = CSW
        self.phi_op = phi_op
        self.max_iter = max_iter
        #ミニバッチ化する時に使う
        # self.psi_minibatch_size = psi_minibatch_size
        self.device = device
        #正則化項の係数
        self.lam = lam
        self.customize_loss_threshold = customize_loss_threshold
        self.custom_loss_count = 0




        self.flow_names = ["Planar", "Residual"]
        self.phi_num_flow_layer = 1
        self.phi_op_lr = 0.000340383810478616
        self.phi_op_weight_decay = 9.366845209461106e-10
        
        
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
        
        phi = Flow_structure(flow_name = self.flow_names[0] , n_flow_layer = self.phi_num_flow_layer)
        # phi = torch.compile(phi)
        phi = phi.to(self.device)
        phi_op = torch.optim.Adam(params=phi.parameters(), lr=  self.phi_op_lr, weight_decay=self.phi_op_weight_decay)

        #train
        if train_or_test == "train":
            phi.train()
            reg_lam = self.lam
            for _ in range(self.max_iter):
                phi_op.zero_grad()
                first_samples_transform  = phi(first_samples_detach)
                second_samples_transform = phi(second_samples_detach)
                # calcurate cos_similarity_wassersten_distance
                cswd = self.CSW(first_samples_transform, second_samples_transform)
                # # calcurate regularization_of_normalizing_flow
                reg_first_samples = self.regularization_of_normalizing_flow(first_samples_transform) /(first_samples_transform.shape[0]*first_samples_transform.shape[1])
                reg_second_samples = self.regularization_of_normalizing_flow(second_samples_transform)/ (second_samples_transform.shape[0]*second_samples_transform.shape[1])
                regularization = reg_lam * (reg_first_samples + reg_second_samples)

                #gradient ascent
                loss= regularization - cswd
                loss.backward(retain_graph=True)
                phi_op.step()
            #更新のたびに、正則化項の係数を小さくする
            self.lam = self.lam * 0.999
        #test
        elif train_or_test == "test":
            phi.eval()
        first_samples_transform = phi(first_samples)
        second_samples_transform= phi(second_samples)
        cswd = self.CSW(first_samples_transform, second_samples_transform)

        return  cswd, first_samples_transform, second_samples_transform















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

#     psi = Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
#     #コンパイル
#     # psi = torch.compile(psi)
#     psi = psi.to(device)

#     psi_op = optim.Adam(psi.parameters(), lr=1e-3, betas=(0.5, 0.999))
#     CSW =  Cos_similarity_W(device = device, p = 1)

#     loss = max_cos_similarity_wassersten_distance(phi = psi, CSW = CSW, phi_op = psi_op, lam = lamda_of_regularization, customize_loss_threshold = customize_loss_threshold, max_iter=50, device= device)

#     time_sta = time.time()
#     ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
#     # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
#     time_end = time.time()
#     tim = time_end- time_sta
#     # print(ssw.item())
#     print("time", tim)


if __name__ == "__main__":

    """
    動作チェック
    """
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    inputs1 = F.normalize(torch.randn((3, 64,3)).to(device), p=2, dim=-1)
    inputs2 = F.normalize(torch.randn((3, 64,3)).to(device), p=2, dim=-1)
    flow_name = ["Planar", "Residual"]
    lamda_of_regularization = 1

    #球面からの距離がthresholdより大きいときは、正則化項のみでロスを計算。そうでないときは、CSW+ 正則化項でロスを計算
    customize_loss_threshold = 0.1

    psi = Flow_structure(flow_name = flow_name[1] , n_flow_layer = 8)
    #コンパイル
    # psi = torch.compile(psi)
    psi = psi.to(device)

    psi_op = optim.Adam(psi.parameters(), lr=1e-3, betas=(0.5, 0.999))
    CSW =  Cos_similarity_W(device = device, p = 2)
    loss = max_cos_similarity_wassersten_distance(phi = psi, CSW = CSW, phi_op = psi_op, lam = lamda_of_regularization, customize_loss_threshold = customize_loss_threshold, max_iter=1, device= device)

    time_sta = time.time()
    ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
    # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    time_end = time.time()
    tim = time_end- time_sta
    # print(ssw.item())
    print("time", tim)