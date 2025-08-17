# UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction
We conduct experiments on an Quad-Core 2.40GHz – Intel® Xeon X3220, 64 GB RAM linux computing server, equipped with an NVIDIA RTX A100 GPU with 24 GB memory. We adopt PyTorch 2.3.0 and CUDA 11.8 as the default deep learning library. 

## Baselines  

The deterministic baselines are inplemented based on [STGCN](https://github.com/hazdzz/STGCN),
[DCRNN](https://github.com/chnsh/DCRNN_PyTorch), 
[GWNET](https://github.com/nnzhan/Graph-WaveNet), 
[StemGNN](https://github.com/microsoft/StemGNN), 
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN), 
[AGCRN](https://github.com/LeiBAI/AGCRN), and [SUMformer](https://github.com/Chengyui/SUMformer).

The probabilistic baselines are inplemented based on 
[TimeGrad](https://github.com/zalandoresearch/pytorch-ts),
[STZINB](https://github.com/ZhuangDingyi/STZINB), 
[DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch), 
[CF-GNN](https://github.com/snap-stanford/conformalized-gnn), and [DiffSTG](https://github.com/wenhaomin/DiffSTG).
