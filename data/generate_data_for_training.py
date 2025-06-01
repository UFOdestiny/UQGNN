import argparse
import os

import numpy as np


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class LogScaler:
    def __init__(self):
        self.base = 1

    def transform(self, data):
        return np.log(data + 1)

    def inverse_transform(self, data):
        return np.exp(data) - 1


class RatioScaler:
    def __init__(self, ratio=1000):
        self.ratio = ratio

    def transform(self, data):
        return data / self.ratio

    def inverse_transform(self, data):
        return data * self.ratio.to(device="cuda")


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    print(df.shape)
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    # df = pd.DataFrame()

    # # df = df.append(df_tmp)
    # print('original data shape:', df.shape)
    #
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    # data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)

    nyc = ""
    sz = ""
    sz2 = ""
    data = np.load(nyc) #sz
    data = data.transpose(2, 0, 1)

    # important
    # data += 1
    # important

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)

    print('final data shape:', data.shape, 'idx shape:', idx.shape)

    num_samples = len(idx)
    num_train = round(num_samples * 0.8)
    num_val = round(num_samples * 0.1)

    # split idx

    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]
    idx_all = idx[:]

    # normalize
    # x_train = data[:idx_val[0] - args.seq_length_x, :, :]
    scaler = LogScaler()
    data = scaler.transform(data)
    # mean=scaler.mean
    # std=scaler.std

    # data = data.transpose(1, 2, 0)
    # means = np.mean(data, axis=(0, 2))
    # data = data - means.reshape(1, -1, 1)
    # stds = np.std(data, axis=(0, 2))
    # data = data / stds.reshape(1, -1, 1)
    # data = data.transpose(2, 0, 1)
    # mean = means
    # std = stds
    print(data.shape)
    print(data.mean())
    print(data.max())
    print(data.min())
    mean = 1
    std = 1

    # save
    out_dir = args.dataset + '/' + args.years
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=mean, std=std)
    # np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=means, std=stds)
    # np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    # np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    # np.save(os.path.join(out_dir, 'idx_test'), idx_test)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)
    np.save(os.path.join(out_dir, 'idx_all'), idx_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nyc', help='dataset name')
    # years
    parser.add_argument('--years', type=str, default='2019',
                        help='2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=1, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')

    args = parser.parse_args()
    generate_train_val_test(args)
