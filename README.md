<div align="center">   
  
# Scaling Multi-Camera 3D Object Detection through Weak-to-Strong Eliciting
</div>


<br><br>

---
## News <a name="news"></a>
- **`2024/04/10`** Scale-BEV [paper](https://arxiv.org/abs/2404.06700) is available on arXiv.
---

## Dataset Processing <a name="dataset-processing"></a>

我们提供了以下脚本，用于处理不同数据集。所有脚本位于 `UniBEV2/tools` 目录中：

- **Lyft 数据集**: 使用 `python UniBEV_lyft.py` 处理。
- **NuScenes 数据集**: 使用 `python UniBEV_nus.py` 处理。
- **DeepAccident 数据集**: 使用 `python Uni_DeepAccident.py` 处理。
- **Waymo 数据集**: 
  - **训练数据**: 使用 `python waymo_train_pkl.py` 处理。
  - **验证数据**: 使用 `python waymo_val_pkl.py` 处理。

请根据不同的数据集选择相应的脚本运行。

---

## How to Run <a name="how-to-run"></a>

运行训练代码的命令格式如下：

```bash
bash tools/dist_train.sh $config.py$ $GPU数量$
```

### 示例
运行配置文件 `./configs/FB-BEV/fb-r50-cbgs-pc-nus.py`，并指定 GPU 数量为 8 的命令如下：

```bash
bash tools/dist_train.sh ./configs/FB-BEV/fb-r50-cbgs-pc-nus.py 8
```

请确保你已经正确设置了配置文件路径以及对应的 GPU 环境。




## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{scale-bev,
 title={Scaling Multi-Camera 3D Object Detection through Weak-to-Strong Eliciting}, 
 author={Hao LU, Jiaqi TANG, Xinli XU, Xu CAO, Yunpeng ZHANG, Guoqing WANG, Dalong DU, Hao CHEN, Yingcong CHEN},
 booktitle={https://arxiv.org/abs/2404.06700},
 year={2024},
}
```

