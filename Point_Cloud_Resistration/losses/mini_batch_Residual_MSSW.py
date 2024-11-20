import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import time
import losses.normflows_ishikawa as nf
# import normflows_ishikawa as nf

"""""""""""""""""""""""""""""""""""""""""
SSW
"""""""""""""""""""""""""""""""""""""""""
def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    ## https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch

    # assumes 2D array
    n_rows, n_cols = mat.shape
    
    if dim==0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)
    

def dCost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    n = u_values.shape[-1]
    m_batch, m = v_values.shape
    
    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
         
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    
    u_index = torch.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n-1))
    
    ## Deal with 1
    u_cdfm = torch.cat([u_cdf, u_cdf[:,0].view(-1,1)+1], dim=1)
    u_valuesm = torch.cat([u_values, u_values[:,0].view(-1,1)+1],dim=1)
    u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
    u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))
    
    dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdf_theta-v_values[:,:-1]), p), axis=-1)
    
    dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdfm_theta-v_values[:,:-1]), p), axis=-1)
    
    return dCp.reshape(-1,1), dCm.reshape(-1,1)


def Cost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    m_batch, m = v_values.shape
    n_batch, n = u_values.shape

    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
    
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    ## Put negative values at the end
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))# .tolist()

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)  
    
    ## Compute abscisse
    cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

    ## Compute icdf
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))
    
    if p == 1:
        ot_cost = torch.sum(delta*torch.abs(u_icdf-v_icdf), axis=-1)
    elif p == 2:
        ot_cost = torch.sum(delta*torch.square(u_icdf-v_icdf), axis=-1)
    else:
        ot_cost = torch.sum(delta*torch.pow(torch.abs(u_icdf-v_icdf), p), axis=-1)
    return ot_cost



def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, 
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
    r"""
    Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [1].

    Parameters:
    u_values : ndarray, shape (n_batch, n_samples_u)
        samples in the source domain
    v_values : ndarray, shape (n_batch, n_samples_v)
        samples in the target domain
    u_weights : ndarray, shape (n_batch, n_samples_u), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n_batch, n_samples_v), optional
        samples weights in the target domain
    p : float, optional
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.

    [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    ## Matlab Code : https://users.mccme.ru/ansobol/otarie/software.html
    
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)
    
    L = max(Lm,Lp)
    
    tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tm = tm.repeat(1, m)
    tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tp = tp.repeat(1, m)
    tc = (tm+tp)/2
    
    done = torch.zeros((u_values.shape[0],m))
        
    cpt = 0
    while torch.any(1-done):
        cpt += 1
        
        dCp, dCm = dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp*dCm)<=0) * 1
        
        mask = ((tp-tm)<eps/L) * (1-done)
        
        if torch.any(mask):
            ## can probably be improved by computing only relevant values
            dCptp, dCmtp = dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
        
            mask_end = mask * (torch.abs(dCptm-dCmtp)>0.001)
            tc[mask_end>0] = ((Ctp-Ctm+tm*dCptm-tp*dCmtp)/(dCptm-dCmtp))[mask_end>0]
            done[torch.prod(mask, dim=-1)>0] = 1
        ## if or elif?
        elif torch.any(1-done):
            tm[((1-mask)*(dCp<0))>0] = tc[((1-mask)*(dCp<0))>0]
            tp[((1-mask)*(dCp>=0))>0] = tc[((1-mask)*(dCp>=0))>0]
            tc[((1-mask)*(1-done))>0] = (tm[((1-mask)*(1-done))>0]+tp[((1-mask)*(1-done))>0])/2
    
    return Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)


def emd1D_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    if p == 1:
        ## Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
        values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)
        
        cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights),-1),-1,values_sorter),-1)
        cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)
        
        values_sorted = torch.nn.functional.pad(values_sorted, (0,1), value=1)
        delta = values_sorted[..., 1:]-values_sorted[..., :-1]
        weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

        sum_weights = torch.cumsum(weight_sorted, axis=-1)-0.5
        sum_weights[sum_weights<0] = np.inf
        inds = torch.argmin(sum_weights, axis=-1)
            
        levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1,1))
        
        return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)



def sliced_cost(Xs, Xt, Us, p=2, u_weights=None, v_weights=None):
    """
        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        Us: ndarray, shape (num_projections, d, 2)
            Independent samples of the Uniform distribution on V_{d,2}
        p: float
            Power
    """
    n_projs, d, k = Us.shape
    n, _ = Xs.shape
    m, _ = Xt.shape

    
    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(Us,1,2)[:,None], Xs[:,:,None]).reshape(n_projs, n, 2)
    Xpt = torch.matmul(torch.transpose(Us,1,2)[:,None], Xt[:,:,None]).reshape(n_projs, m, 2)
        
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    Xpt = F.normalize(Xpt, p=2, dim=-1)
    
    ## Get coords
    Xps = (torch.atan2(-Xps[:,:,1], -Xps[:,:,0])+np.pi)/(2*np.pi)
    Xpt = (torch.atan2(-Xpt[:,:,1], -Xpt[:,:,0])+np.pi)/(2*np.pi)
        
    if p==1:
        w1 = emd1D_circle(Xps, Xpt, u_weights=u_weights, v_weights=v_weights)
    else:
        w1 = binary_search_circle(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

    return torch.mean(w1)
    

def sliced_wasserstein_sphere(Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
    """
        Compute the sliced-Wasserstein distance on the sphere.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    d = Xs.shape[1]
    
    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections,d,2), device=device)
    U, _ = torch.linalg.qr(Z)

    return sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights)





"""""""""""""""""""""""""""""""""""""""""
Normalinig flow
\phi : \mathbb{R}^3 \rightarrow \mathbb{R}^2
\psi : \mathbb{R}^2 \rightarrow \mathbb{S}^2
"""""""""""""""""""""""""""""""""""""""""

class MLP_Architecture(torch.nn.Module):
	def __init__(self, emb_dims=2):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(MLP_Architecture, self).__init__()

		self.emb_dims = emb_dims
		self.layers = self.create_structure()
		self.net = torch.nn.Sequential(*self.layers)

	def create_structure(self):
		self.conv1 = torch.nn.Conv1d(3, 8, 1)
		self.conv2 = torch.nn.Conv1d(8, 8, 1)
		self.conv3 = torch.nn.Conv1d(8, self.emb_dims, 1)
		self.relu = torch.nn.ReLU()

		layers = [self.conv1, self.relu,
				  self.conv2, self.relu,
				  self.conv3,
                    ]
		return layers

	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		input_data = input_data.permute(0, 2, 1)
		output = self.net(input_data)
		output = output.permute(0, 2, 1)
		return output



class Flow_structure(nn.Module):
    def __init__(self, input_dim = 2, flow_name =  "Planar", n_flow_layer = 3):
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
            hidden_units = 4
            hidden_layers = 2

            flows = []
            for i in range(n_flow_layer):
                net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
                                        init_zeros=True, lipschitz_const=0.9)
                flows += [nf.flows.Residual(net, reduce_memory=True)]
                # if i < n_flow_layer - 1:
                flows += [nf.flows.ActNorm(latent_size)]
            return flows
        else:
            raise ValueError("Flow name is not valid")

    def forward(self, x):
        for flow in self.net:
            x, _= flow(x)
        return x

class transform_to_sphere(nn.Module):
    def __init__(self,flow_name, n_flow_layer = 3, two_d_encoder = MLP_Architecture, flow = Flow_structure):
        super(transform_to_sphere, self).__init__()
        self.net = two_d_encoder()
        self.flows = flow(flow_name = flow_name, n_flow_layer = n_flow_layer)

    def forward(self, x):
        x = self.net(x)
        x = x.contiguous()
        x = self.flows(x)
        theta_1 = np.pi * (torch.tanh(x[:, :, 0])/2 + 0.5)
        theta_2 = np.pi * torch.tanh(x[:, :, 1])
        sphere = torch.stack([torch.sin(theta_1) * torch.cos(theta_2), torch.sin(theta_1) * torch.sin(theta_2), torch.cos(theta_1) ], dim=2)

        return sphere



"""""""""""""""""""""""""""""""""""""""""
Minibatch Max SSW
"""""""""""""""""""""""""""""""""""""""""
class max_spherical_wassersten_distance_Residual(nn.Module):
    def __init__(self,num_projections, phi,  phi_op ,SSW = sliced_wasserstein_sphere, p=2,max_iter=10, psi_minibatch_size = 5, device='cuda'):
        super(max_spherical_wassersten_distance_Residual, self).__init__()
        self.num_projections = num_projections
        self.phi = phi
        self.SSW = SSW
        self.phi_op = phi_op
        self.p = p
        self.max_iter = max_iter
        self.psi_minibatch_size = psi_minibatch_size
        self.device = device

    def forward(self, first_samples,second_samples, train_or_test = "train"):
        first_samples_detach = first_samples.detach()
        second_samples_detach = second_samples.detach()

        if train_or_test == "train":
            for _ in range(self.max_iter):
                first_samples_transform=self.phi(first_samples_detach)
                second_samples_transform = self.phi(second_samples_detach)
                mini_batch = np.random.choice(len(first_samples_transform), size=self.psi_minibatch_size, replace=False)
                ssw = 0
                for i in mini_batch:
                    ssw += self.SSW(first_samples_transform[i],second_samples_transform[i], self.num_projections, self.device, p = self.p)

                loss= -ssw #gradient ascent
                self.phi_op.zero_grad()
                loss.backward(retain_graph=True)
                self.phi_op.step()
                print(ssw.item())
        elif train_or_test == "test":
            pass

        first_samples_transform =self.phi(first_samples)
        second_samples_transform = self.phi(second_samples)
        ssw = 0
        for i in range(len(first_samples_transform)):
            ssw += self.SSW(first_samples_transform[i],second_samples_transform[i], self.num_projections, self.device, p = self.p)
        return  ssw, first_samples_transform, second_samples_transform



"""
動作チェック
"""
if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    flow_name = ["Planar", "Residual"]
    num_projections = 10
    # n_layers = 1
    inputs1 = torch.randn((32,64, 3)).to(device)
    inputs2 = torch.randn((32,64, 3)).to(device)
    # inputs1 = F.normalize(torch.randn((32,512, 3)).to(device), p=2, dim=-1)
    # inputs2 = F.normalize(torch.randn((32,512, 3)).to(device), p=2, dim=-1)
    init_inputs1 = inputs1.clone()
    init_inputs2 = inputs2.clone()


    psi = transform_to_sphere(flow_name = flow_name[1] ,n_flow_layer = 2)
    psi = psi.to(device)

    init_inputs1 = psi(init_inputs1)
    init_inputs2 = psi(init_inputs2)

    psi_op = optim.Adam(psi.parameters(), lr=0.01, betas=(0.5, 0.999))
    SSW =  sliced_wasserstein_sphere
    loss = max_spherical_wassersten_distance_Residual(num_projections, phi = psi, SSW = SSW,  phi_op = psi_op, p=2,max_iter=20 ,psi_minibatch_size = 2, device= device)

    time_sta = time.time()
    ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "train")
    # ssw, output1, output2 = loss(inputs1, inputs2, train_or_test = "test")
    time_end = time.time()
    tim = time_end- time_sta
    print(ssw.item())
    print("time", tim)

