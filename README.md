# UQGNN

**UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction**  
UQGNN is a novel graph neural network for multivariate spatiotemporal prediction that not only forecasts mean values but also quantifies uncertainty. By integrating an interaction-aware spatiotemporal embedding module and a multivariate probabilistic prediction module, UQGNN effectively captures complex urban dynamics and achieves superior performance in both accuracy and uncertainty estimation across multiple real-world datasets.

## 🔧 Implementation Details
We conduct experiments on an Quad-Core 2.40GHz – Intel® Xeon X3220, 64 GB RAM linux computing server, equipped with an NVIDIA RTX A100 GPU with 24 GB memory. We adopt PyTorch 2.3.0 and CUDA 11.8 as the default deep learning library.

## 📁 Project Structure

```
├── experiments/uqgnn/  # Traning
├── src/base/           # Fundamental model and engine
├── src/engines/        # UGQNN's enginee
├── src/models/         # UGQNN's model
├── src/utils/          # Configuration and dataloader
└── README.md           # Project documentation
```

## 📊 Baselines  

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


## 📖 Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{yu2025uqgnn,
  title={UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction},
  author={Yu, Dahai and Zhuang, Dingyi and Jiang, Lin and Xu, Rongchao and Ye, Xinyue and Bu, Yuheng and Wang, Shenhao and Wang, Guang},
  journal={Proceedings of the 33nd ACM International Conference on Advances in Geographic Information Systems},
  year={2025}
}
```


*This project is part of the Sigspatial 2025 Research Track.*