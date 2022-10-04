import torch
import torch.nn as nn

class SPAUCI(nn.Module):
    def __init__(self, alpha, beta):
        super(SPAUCI,self).__init__()
        self.device = torch.device('cuda:0')
        # FPR range
        self.alpha = torch.tensor(alpha).cuda(0)
        self.na = alpha
        self.beta = torch.tensor(beta).cuda(0)
        # auxiliary variable
        self.kappa = torch.tensor(2).cuda(0)
        # optimzation variable
        self.a = torch.tensor(0.5, requires_grad=True, device=self.device)
        self.b = torch.tensor(0.5, requires_grad=True, device=self.device)
        self.g = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.s_p = torch.tensor(0.5, requires_grad=True, device=self.device)
        self.s_n = torch.tensor(0.5, requires_grad=True, device=self.device)
        self.lam_b = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.lam_a = torch.tensor(0.0, requires_grad=True, device=self.device)
    def forward(self, pred, target):
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        # one way partial auc
        if self.na == 1:
            max_val_n = torch.log(1+torch.exp(self.kappa*(torch.square(pred_n - self.b) + \
                          2 * (1 + self.g) * pred_n - self.s_n)))/self.kappa
            res = torch.mean(torch.square(pred_p - self.a) - \
                          2 * (1 + self.g) * pred_p) + (self.beta * self.s_n + \
                    torch.mean(max_val_n))/self.beta -\
                    1*self.g**2 - self.lam_b * (self.b-1-self.g)
        # two way partial auc
        else:
            max_val_p = torch.log(1+torch.exp(self.kappa*(torch.square(pred_p - self.a) - \
                          2 * (1 + self.g) * pred_p - self.s_p)))/self.kappa
            max_val_n = torch.log(1+torch.exp(self.kappa*(torch.square(pred_n - self.b) + \
                          2 * (1 + self.g) * pred_n - self.s_n)))/self.kappa
            res = self.s_p + torch.mean(max_val_p)/self.alpha + \
                  self.s_n + torch.mean(max_val_n)/self.beta -\
                    1*self.g**2 - self.lam_b * (self.b-1-self.g) + self.lam_a * (self.a+self.g)
            
        return res

