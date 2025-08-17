import numpy as np
import torch


def CRPS(preds, labels):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """
    eval_points = torch.ones_like(labels)
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = torch.sum(torch.abs(labels * eval_points))
    crps = 0
    length = len(quantiles)

    for i in range(length):
        q_pred = []
        for j in range(len(preds)):
            q_pred.append(torch.quantile(preds[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = 2 * torch.sum(torch.abs((q_pred - labels) * eval_points * ((labels <= q_pred) * 1.0 - quantiles[i])))

        crps += q_loss / denom
    return crps.item() / length


def crps2(y_pred, y_test, quantiles):
    y_cdf = np.zeros((y_pred.shape[0], quantiles.size + 2))
    y_cdf[:, 1:-1] = y_pred
    y_cdf[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
    y_cdf[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]

    ind = np.zeros(y_cdf.shape)
    ind[y_cdf > y_test.reshape(-1, 1)] = 1.0

    qs = np.zeros((1, quantiles.size + 2))
    qs[0, 1:-1] = quantiles
    qs[0, 0] = 0.0
    qs[0, -1] = 1.0

    return np.trapz((qs - ind) ** 2.0, y_cdf)


if __name__ == "__main__":
    # test for CRPS
    B, T, V = 32, 12, 36
    n_sample = 1

    target = torch.randn((B, n_sample, T, V))
    forecast = torch.randn((B, n_sample, T, V))

    # label = target.unsqueeze(1).expand_as(forecast)
    # eval_points = torch.randn_like(target)
    #
    # print(target.shape)
    # print(forecast.shape)
    # print(label.shape)
    # print(eval_points.shape)

    crps = CRPS(forecast, target)
    print('crps:', crps)

    # crps = CRPS(target, label)
    # print('crps:', crps)
    #
    # mis = calc_mis(target, forecast)
    # print('mis:', mis)
    #
    # mis = calc_mis(target, label)
    # print('mis:', mis)
