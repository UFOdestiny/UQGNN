# UQGNN
We conduct experiments on an Quad-Core 2.40GHz – Intel® Xeon X3220, 64 GB RAM linux computing server, equipped with an NVIDIA RTX A100 GPU with 16 GB memory. We adopt PyTorch 2.3.0 and CUDA 11.8 as the default deep learning library. 

## Baselines  

The deterministic baselines are inplemented based on [DCRNN](https://github.com/chnsh/DCRNN_PyTorch), [AGCRN](https://github.com/LeiBAI/AGCRN), [STGCN](https://github.com/hazdzz/STGCN), [GWNET](https://github.com/nnzhan/Graph-WaveNet), [SUMformer](https://github.com/Chengyui/SUMformer), [DSTAGNN](https://github.com/SYLan2019/DSTAGNN), and [StemGNN](https://github.com/microsoft/StemGNN).  
The probabilistic baselines are inplemented based on [DiffSTG](https://github.com/wenhaomin/DiffSTG), [TimeGrad](https://github.com/zalandoresearch/pytorch-ts), [STZINB](https://github.com/ZhuangDingyi/STZINB), [CF-GNN](https://github.com/snap-stanford/conformalized-gnn), and [DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch).
