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
配置文件说明

3个真实数据集（Waymo，nus， Lyft）联合训练
| **方法名称**      | **对应配置文件**                          | **描述**                     |
|-------------------|------------------------------------------|-----------------------------|
| **DETR**         | `UniBEV2/configs/Uni_BEV_v2/detr-r50-uni_v2.py`                    | DETR 方法通用版本配置。         |
| **PETR**          | `UniBEV2/configs/Uni_BEV_v2/petr-r50-uni_v2.py`                    | PETR 方法通用版本配置。        |
| **BEVDet**        | `UniBEV2/configs/Uni_BEV_v2/bevdet-r50-uni_v2.py`                  | BEVDet 方法通用版本配置。        |
| **BEVDepth**      | `UniBEV2/configs/Uni_BEV_v2/bevdepth-r50-uni_v2.py`                | BEVDepth 方法，用于 R50 数据。    |
| **BEVFormer**     | `UniBEV2/configs/Uni_BEV_v2/bevformer-r50-uni_v2.py`               | BEVFormer 方法，用于 R50 数据。   |
| **FB-BEV**         `UniBEV2/configs/Uni_BEV_v2/fb-r50-uni_v2.py`                      | FB-BEV 方法通用版本配置。          |
| **PCBEV**         | `UniBEV2/configs/Uni_BEV_v2/pdbev-r50-uni_v2.py`                   | PCBEV 方法主版本配置。          |
|  **Scale-BEV**     | `UniBEV2/configs/Uni_BEV_v2/pdbev-samv2-aug-p9-r50-uni_v2.py`      | Scale-BEV 方法增强版本配置。      |

真实和虚拟数据集（nus， DeepAccident）联合训练

| **方法名称**      | **对应配置文件**                          | **描述**                     |
|-------------------|------------------------------------------|-----------------------------|
| **DETR**       | `UniBEV2/configs/Uni_RealSim_v2/detr-r50-realsim_v2.py`                | DETR 方法，用于 RealSim 数据集。 |
| **PETR**          | `UniBEV2/configs/Uni_RealSim_v2/petr-r50-realsim-v2.py`                | PETR 方法，用于 RealSim 数据集。 |
| **BEVDet**        | `UniBEV2/configs/Uni_RealSim_v2/bevdet-r50-realsim-v2.py`              | BEVDet 方法，用于 RealSim 数据集。 |
| **BEVDepth**      | `UniBEV2/configs/Uni_RealSim_v2/bevdepth-r50-realsim-v2.py`            | BEVDepth 方法，用于 RealSim 数据集。 |
| **BEVFormer**     | `UniBEV2/configs/Uni_RealSim_v2/bevformer-r50-realsim_v2.py`           | BEVFormer 方法，用于 RealSim 数据集。 |
| **FB-BEV**                   | `UniBEV2/configs/Uni_RealSim_v2/fb-bevformer-r50-realsim-v2.py`        | FB-BEV 方法，用于 RealSim 数据集。 |
| **PCBEV**         | `UniBEV2/configs/Uni_RealSim_v2/pdbev-r50-realsim-v2.py`               | PCBEV 方法主版本配置，用于 RealSim 数据集。 |
| **Scale-BEV**                   | `UniBEV2/configs/Uni_RealSim_v2/pdbev-samv2-aug-r50-realsim-v2.py`     | Scale-BEV 方法 SAM V2 增强版本，用于 RealSim 数据集。 |

---

### 说明：
1. **方法名称**：列出了配置文件对应的模型方法（如 PETR、BEVDet 等）。
2. **对应配置文件**：列出了每个模型方法对应的具体配置文件名称。
3. **描述**：简要说明了配置文件的用途或模型版本的特点。
4. 文件路径默认为 `UniBEV2/configs/Uni_BEV_v2/`，表格中只列出了文件名。

将以上表格与之前的表格合并后，可以更清晰地展示所有方法及其配置文件。如果需要进一步修改或补充，请告诉我！

---

### 说明：
1. **方法名称**：对应的是你提到的模型方法（如 PETR、BEVDet 等）。
2. **对应配置文件**：列出了每个模型方法对应的配置文件的具体名称。
3. **描述**：简要说明了配置文件的用途或模型版本的特点。

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

