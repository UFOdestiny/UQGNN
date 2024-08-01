import os
import platform
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + '/../../..'))
from src.engines.uqgnn import UQGNN_Engine
import torch

from src.models.uqgnn import UQGNN
from src.utils.metrics import mnormal_loss

torch.set_num_threads(3)
from src.utils.graph_algo import normalize_adj_mx
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    # parser.add_argument('--dataset', type=str, default="UAHGNN")
    # parser.add_argument('--model_name', type=str, default="STGCN")
    parser.add_argument('--diffusion_step', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)

    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    args = parser.parse_args()
    args.model_name = "UQGNN"

    args.mode = "train"  # use the model

    if platform.system().lower() == 'linux':
        log_dir = ''
    else:
        log_dir = ''

    logger = get_logger(log_dir, __name__, )
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    # device = torch.device(args.device)
    device = torch.device(0)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)

    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)

    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num
    blocks = []
    blocks.append([args.input_dim])
    for l in range(args.block_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.horizon])

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = UQGNN(node_num=node_num,
                  gso=gso,
                  blocks=blocks,
                  Kt=args.Kt,
                  Ks=args.Ks,
                  dropout=args.dropout,
                  adj_mx=adj_mx,
                  diffusion_step=args.diffusion_step,
                  # hidden=args.hidden_dim,
                  # rank=args.rank,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  feature=args.feature,
                  horizon=args.horizon,
                  device=device,
                  )

    loss_fn = mnormal_loss
    # loss_fn = nn.GaussianNLLLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = UQGNN_Engine(device=device,
                                    model=model,
                                    dataloader=dataloader,
                                    scaler=scaler,
                                    sampler=None,
                                    loss_fn=loss_fn,
                                    lrate=args.lrate,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    clip_grad_value=args.clip_grad_value,
                                    max_epochs=args.max_epochs,
                                    patience=args.patience,
                                    log_dir=log_dir,
                                    logger=logger,
                                    seed=args.seed,
                                    normalize=args.normalize)

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode, args.model_path)


if __name__ == "__main__":
    main()
