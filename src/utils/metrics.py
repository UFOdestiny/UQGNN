import numpy as np
import properscoring as ps
import torch
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import torch.distributions as dist

class MNB:
    def __init__(self, mu, r):
        self.mu = mu
        self.r = r
        # Broadcast r if it has a lower dimensionality
        self.r = r.expand_as(mu) if r.shape != mu.shape else r

    def sample(self):
        # Step 1: Generate Gamma random variables for the Poisson rate parameter
        gamma_rate = dist.Gamma(self.r, self.r / self.mu).sample()
        # Step 2: Generate Poisson samples with the Gamma random rates
        poisson_samples = dist.Poisson(gamma_rate).sample()
        return poisson_samples

    def log_prob(self, counts):
        # Compute the log-probability for each feature
        term1 = torch.lgamma(counts + self.r) - torch.lgamma(self.r) - torch.lgamma(counts + 1)
        term2 = self.r * torch.log(self.r) + counts * torch.log(self.mu)
        term3 = -(counts + self.r) * torch.log(self.r + self.mu)
        log_prob = term1 + term2 + term3
        # Sum over features for each (batch, N) pair
        return log_prob



def masked_mse(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if preserve:
        return loss
    else:
        return torch.mean(loss)


def masked_rmse(preds, labels, null_val, preserve=False):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, preserve=preserve))


def masked_mae(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if preserve:
        return loss
    else:
        return torch.mean(loss)


def masked_mape(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if preserve:
        return loss
    else:
        return torch.mean(loss)


def masked_kl(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # distribution discrete gaussian distribution con
    loss = preds * torch.log((preds + 1e-5) / (labels + 1e-5))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if preserve:
        return loss
    else:
        return torch.mean(loss)


def masked_mpiw(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    m, v = preds
    if v.shape != m.shape:
        v = torch.diagonal(v, dim1=-2, dim2=-1)
    loss = 2 * 1.96 * v  # / (12 ** 0.5)
    # print(loss.shape,mask.shape)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    if preserve:
        return loss
    else:
        return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val, uq=False, preserve=False):
    if preserve:
        mae = masked_mae(preds, labels, null_val, preserve)
        mape = masked_mape(preds, labels, null_val, preserve)
        rmse = masked_rmse(preds, labels, null_val, preserve)
    else:
        mae = masked_mae(preds, labels, null_val, preserve).item()
        mape = masked_mape(preds, labels, null_val, preserve).item()
        rmse = masked_rmse(preds, labels, null_val, preserve).item()

    res = [mae, mape, rmse]

    if uq:
        if preserve:
            kl = masked_kl(preds, labels, null_val, preserve)
            crps = 1  # masked_CRPS(preds.unsqueeze(1), labels.unsqueeze(1), null_val, preserve)
        else:
            kl = masked_kl(preds, labels, null_val, preserve).item()
            crps = 1  # masked_CRPS(preds, labels, null_val, preserve).item()

        res.append(kl)
        res.append(crps)
        # mpiw = masked_mpiw(preds, labels, null_val).item()
        # res.append(mpiw)

    return res


def nb_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    n, p, pi = preds
    pi = torch.clip(pi, 1e-3, 1 - 1e-3)
    p = torch.clip(p, 1e-3, 1 - 1e-3)

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    lambda_ = 1e-4

    L_yeq0 = torch.log(pi_yeq0 + lambda_) + torch.log(lambda_ + (1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0 + lambda_) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(
        n_yg0 + lambda_) + n_yg0 * torch.log(p_yg0 + lambda_) + yg0 * torch.log(1 - p_yg0 + lambda_)

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def nb_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    n, p, pi = preds

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    index1 = p_yg0 == 1
    p_yg0[index1] = torch.tensor(0.9999)
    index2 = pi_yg0 == 1
    pi_yg0[index2] = torch.tensor(0.9999)
    index3 = pi_yeq0 == 1
    pi_yeq0[index3] = torch.tensor(0.9999)
    index4 = pi_yeq0 == 0
    pi_yeq0[index4] = torch.tensor(0.001)

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(
        n_yg0) + n_yg0 * torch.log(p_yg0) + yg0 * torch.log(1 - p_yg0)

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def gaussian_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    var = torch.pow(scale, 2)
    loss = (labels - loc) ** 2 / var + torch.log(2 * torch.pi * var)

    # pi = torch.acos(torch.zeros(1)).item() * 2
    # loss = 0.5 * (torch.log(2 * torch.pi * var) + (torch.pow(labels - loc, 2) / var))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def laplace_nll_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    loss = torch.log(2 * scale) + torch.abs(labels - loc) / scale

    # d = torch.distributions.poisson.Poisson
    # loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def mnormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    dis = MultivariateNormal(loc=loc, covariance_matrix=scale)
    loss = dis.log_prob(labels)

    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss



def mnormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    dis = MultivariateNormal(loc=loc, covariance_matrix=scale)
    loss = dis.log_prob(labels)

    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def mnb_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds
    d = MNB(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def lognormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    dis = LogNormal(loc, scale)
    loss = dis.log_prob(labels + 0.000001)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def tnormal_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    d = Normal(loc, scale)
    prob0 = d.cdf(torch.Tensor([0]).to(labels.device))
    loss = d.log_prob(labels) - torch.log(1 - prob0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def laplace_loss(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loc, scale = preds

    d = Laplace(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def masked_crps(preds, labels, null_val, preserve=False):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    m, v = preds
    if v.shape != m.shape:
        v = torch.diagonal(v, dim1=-2, dim2=-1)

    # loss = ps.crps_gaussian(labels, mu=m, sig=v)
    loss = ps.crps_ensemble(labels, m)
    if preserve:
        return loss
    else:
        return loss.mean()


# def masked_crps(preds, labels, null_val, preserve=False):
#     """
#     target: (B, T, V), torch.Tensor
#     forecast: (B, n_sample, T, V), torch.Tensor
#     eval_points: (B, T, V): which values should be evaluated,
#     """
#     eval_points = torch.ones_like(labels)
#     quantiles = torch.arange(0.05, 1.0, 0.05)
#     denom = torch.sum(torch.abs(labels * eval_points))
#     crps = 0
#     length = len(quantiles)
#
#     for i in range(length):
#         q_pred = []
#         for j in range(len(preds)):
#             q_pred.append(torch.quantile(preds[j: j + 1], quantiles[i], dim=1))
#         q_pred = torch.cat(q_pred, 0)
#         q_loss = 2 * torch.sum(torch.abs((q_pred - labels) * eval_points * ((labels <= q_pred) * 1.0 - quantiles[i])))
#
#         crps += q_loss / denom
#     return crps / length


def crps(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples ** 2
    return np.average(per_obs_crps, weights=sample_weight)


if __name__ == "__main__":
    crps = ps.crps_gaussian(1500, mu=1500, sig=1200)
    print(crps)
