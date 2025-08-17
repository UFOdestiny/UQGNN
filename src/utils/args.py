import argparse
import platform

def get_public_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Shenzhen')  # Shenzhen NYC
    parser.add_argument('--years', type=str, default='2016')
    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=1)

    parser.add_argument('--feature', type=int, default=5)
    parser.add_argument('--input_dim', type=int, default=5)  # feature
    parser.add_argument('--output_dim', type=int, default=5)

    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--uq_metrics', type=bool, default=False)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mode', type=str, default='train')

    return parser

def get_log_path(args):
    if platform.system().lower() == 'linux':
        log_dir = '/home/dy23a.fsu/neu24/result/{}/{}/'.format(args.model_name, args.dataset)
    else:
        log_dir = r'D:/OneDrive - Florida State University/code/LargeST-main/result/{}/{}/'.format(
            args.model_name, args.dataset)

    return log_dir

def get_data_path():
    if platform.system().lower() == 'linux':
        path = '/blue/gtyson.fsu/dy23a.fsu/UQGNN/'
    else:
        path = 'D:/OneDrive - Florida State University/code/LargeST-main/data/'

    return path