import torch

"""
ちゃんと Lnノルムで動くように変更した。
"""



    
class log_Sinkhorn_Distance_Loss(torch.nn.Module):
    """
    Sinkhorn Distance Loss
    """
    def __init__(self, eps, max_iter, batch_reduction ='none', type_of_cost_norm = 'L2'):
        super(log_Sinkhorn_Distance_Loss, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.batch_reduction = batch_reduction
        self.p = self._type_of_cost_norm(type_of_cost_norm)

    def forward(self, x, y, device):
        #コスト行列の計算
        C = self._cost_matrix(x, y, self.p).to(device)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # probability vectors (経験分布なら 1/ sample size)
        probability_vector_of_x = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        probability_vector_of_y = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(probability_vector_of_x).to(device)
        v = torch.zeros_like(probability_vector_of_y).to(device)

        # u が十分に収束したら終了するための閾値
        thresh = 1e-9

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            #収束判定用に u を保存
            u_init = u
            #メインの計算
            u = self.eps * (torch.log(probability_vector_of_x+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(probability_vector_of_y+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1))  + v
            #収束判定
            err = (u - u_init).abs().sum(-1).mean()
            if err.item() < thresh:
                break

        U, V = u, v
        # P = diag(a)*K*diag(b)
        P = torch.exp(self.M(C, U, V))
        # 主問題の解 つまり、これがシンクホーンディスタンス
        cost = torch.sum(P * C, dim=(-2, -1))

        # batch reduction 基本的には、バッチサイズが異なることを考慮に入れて,mean を指定すること
        if self.batch_reduction == 'mean':
            cost = cost.mean()
        elif self.batch_reduction == 'sum':
            cost = cost.sum()
        elif self.batch_reduction == 'none':
            pass

        return cost, P, C

    def M(self, C, u, v):
        """
        C: B x N x M
        u: B x N
        v: B x M
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """
        input source shape: B x N x D(= 3)
        input target shape B x M x D(= 3)
        output C shape: B x N x M
        p : Lp norm
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.pow(torch.sum((torch.abs(x_col - y_lin)) ** p, -1), 1/p)
        return C

    @staticmethod
    def _type_of_cost_norm(type_of_cost_norm = 'L2'):
        return int(type_of_cost_norm[-1])



if __name__ == "__main__":
    """
    動作チェック
    """
    batch_size = 2
    a_points = 9
    b_points = 12

    a = torch.tensor([[[i, 0, 0] for i in range(a_points)]]*batch_size)
    b = torch.tensor([[[i, 8, 0] for i in range(b_points)]]*batch_size)

    sinkhorn = log_Sinkhorn_Distance_Loss(eps=0.1, max_iter=10, batch_reduction ='mean', type_of_cost_norm = 'L1')

    loss, P, C = sinkhorn(a, b, device = 'cpu')

    print("loss: ", loss.shape)
    print("shape of a: ", a.shape)
    print("shape of b: ", b.shape)
    print("shape of C: ", C.shape)
    print(loss)