import time

import numpy as np
import torch

from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, masked_mae, compute_all_metrics, masked_kl, masked_mpiw, \
    masked_crps


class UQGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(UQGNN_Engine, self).__init__(**args)
        self.normalize = args["normalize"]

    def train_batch(self):
        self.model.train()
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_kl = []
        train_mpiw = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))

            pred = self.model(X, label, self._iter_cnt)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            if self.normalize:
                pred, label = self._inverse_transform([pred, label])

            mpiw = masked_mpiw(pred, label, mask_value).item()
            pred, _ = pred
            mae = masked_mae(pred, label, mask_value).item()
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            kl = masked_kl(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            train_kl.append(kl)
            train_mpiw.append(mpiw)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse), np.mean(
            train_kl), np.mean(train_mpiw)

    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_kl, mtrain_mpiw = self.train_batch()
            t2 = time.time()

            # v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_kl, mvalid_mpiw = self.evaluate('val')
            # v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, T Loss: {:.4f}, T MAE: {:.4f}, T RMSE: {:.4f}, T MAPE: {:.4f}, T KL: {:.4f}, T MPIW: {:.4f}, V MAE: {:.4f}, V RMSE: {:.4f}, V MAPE: {:.4f}, V KL: {:.4f}, V MPIW: {:.4f}, T Time: {:.4f}s/epoch, LR: {:.4e}'
            self._logger.info(
                message.format(epoch + 1, mtrain_loss, mtrain_mae, mtrain_rmse, mtrain_mape, mtrain_kl, mtrain_mpiw,
                               mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_kl, mvalid_mpiw,
                               (t2 - t1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')

    def evaluate(self, mode, model_path=None):
        if mode == 'test' or mode == 'all':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        p0 = []

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                _, sigma = pred
                if self.normalize:
                    pred, label = self._inverse_transform([pred, label])

                p0.append(pred[1].squeeze(-1).cpu())
                pred, _ = pred
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        p0 = torch.cat(p0, dim=0)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            kl = masked_kl(preds, labels, mask_value).item()
            mpiw = masked_mpiw((preds, p0), labels, mask_value).item()
            return mae, mape, rmse, kl, mpiw

        elif mode == 'test' or mode == 'all':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []

            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                print('Evaluation Shape', preds.shape)
                if mode == "all":
                    pres = True
                else:
                    pres = False
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value, uq=True, preserve=pres)
                if pres:
                    mpiw_ = masked_mpiw((preds[:, i, :], p0[:, i, :]), labels[:, i, :], mask_value, preserve=pres)

                    crps_ = masked_crps((preds[:, i, :], p0[:, i, :]), labels[:, i, :], mask_value,
                                        preserve=pres)

                else:
                    mpiw_ = masked_mpiw((preds[:, i, :], p0[:, i, :]), labels[:, i, :], mask_value,
                                        preserve=pres).item()

                    crps_ = masked_crps((preds[:, i, :], p0[:, i, :]), labels[:, i, :], mask_value,
                                        preserve=pres).item()

                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test KL: {:.4f}, Test MPIW: {:.4f}, Test CRPS: {:.4f}'
                if pres:
                    self._logger.info(log.format(i + 1, torch.mean(res[0]), torch.mean(res[2]), torch.mean(res[1]),
                                                 torch.mean(res[3]), torch.mean(mpiw_), crps_.mean()))
                else:
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1], res[3], mpiw_, crps_))

                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(mpiw_)
                test_crps.append(crps_)

            if mode == "all":
                mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                crps = torch.from_numpy(test_crps[0])
                crps = torch.mean(crps.unsqueeze(0), axis=1)

                # mae = test_mae[0].unsqueeze_(0)
                # kl = test_kl[0].unsqueeze_(0)
                # mpiw = test_mpiw[0].unsqueeze_(0)
                # crps = torch.from_numpy(test_crps[0])
                # crps = crps.unsqueeze_(0)

                # result = np.vstack((mae, kl, mpiw, crps))
                # print(result.shape)
                # np.save(f"{self._save_path}/res.npy", result)

                result = np.vstack((mae, mape, rmse, kl, mpiw, crps))
                print(result.shape)
                np.save(f"{self._save_path}/res_all.npy", result)

                # preds.squeeze_(dim=1)
                # labels.squeeze_(dim=1)
                # preds.unsqueeze_(dim=0)
                # labels.unsqueeze_(dim=0)
                #
                # result = np.vstack((preds, labels))
                # np.save(f"{self._save_path}/predlabel.npy", result)
            else:
                log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test KL: {:.4f}, Test MPIW: {:.4f}'
                self._logger.info(
                    log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape), np.mean(test_kl),
                               np.mean(test_mpiw)))
