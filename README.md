---

<div align="center">   
  
# Scaling Multi-Camera 3D Object Detection through Weak-to-Strong Eliciting
</div>

<br>

---

## News <a name="news"></a>
- **`2024/04/10`**: The Scale-BEV [paper](https://arxiv.org/abs/2404.06700) is now available on arXiv.

---

## Prepare the environment refer to [BEVDet](https://github.com/HuangJunJie2017/BEVDet).**

```shell script
git clone [https://github.com/EnVision-Research/Generalizable-BEV](https://github.com/EnVision-Research/Scale-BEV).git
cd UniBEV2
pip install -v -e .
```


## Dataset Processing <a name="dataset-processing"></a>

We provide the following scripts to process different datasets, all located under the `UniBEV2/tools` directory:

- **Lyft dataset**: Processed with `python UniBEV_lyft.py`.
- **NuScenes dataset**: Processed with `python UniBEV_nus.py`.
- **DeepAccident dataset**: Processed with `python Uni_DeepAccident.py`.
- **Waymo dataset**: 
  - **Training data**: Processed with `python waymo_train_pkl.py`.
  - **Validation data**: Processed with `python waymo_val_pkl.py`.

Choose the appropriate script to process your dataset as needed.

---

## How to Run <a name="how-to-run"></a>

To train a model, you can use the following command:

```bash
bash tools/dist_train.sh $config_file$ $num_gpus$
```

### Example:
To run the configuration file `./configs/FB-BEV/fb-r50-cbgs-pc-nus.py` with 8 GPUs:

```bash
bash tools/dist_train.sh ./configs/FB-BEV/fb-r50-cbgs-pc-nus.py 8
```

---

## Configuration Overview

### Unified Training with Three Real-World Datasets (Waymo, NuScenes, Lyft)

| **Model**         | **Configuration File**                             | **Description**                                       |
|-------------------|---------------------------------------------------|-----------------------------------------------------|
| **DETR**          | `UniBEV2/configs/Uni_BEV_v2/detr-r50-uni_v2.py`    | Standard DETR configuration for multi-dataset training. |
| **PETR**          | `UniBEV2/configs/Uni_BEV_v2/petr-r50-uni_v2.py`    | Standard PETR configuration for multi-dataset training. |
| **BEVDet**        | `UniBEV2/configs/Uni_BEV_v2/bevdet-r50-uni_v2.py`  | Standard BEVDet configuration for multi-dataset training. |
| **BEVDepth**      | `UniBEV2/configs/Uni_BEV_v2/bevdepth-r50-uni_v2.py`| BEVDepth configuration for ResNet-50.               |
| **BEVFormer**     | `UniBEV2/configs/Uni_BEV_v2/bevformer-r50-uni_v2.py`| BEVFormer configuration for ResNet-50.             |
| **FB-BEV**        | `UniBEV2/configs/Uni_BEV_v2/fb-r50-uni_v2.py`      | FB-BEV standard configuration.                     |
| **PCBEV**         | `UniBEV2/configs/Uni_BEV_v2/pdbev-r50-uni_v2.py`   | Main PCBEV configuration.                          |
| **Scale-BEV**     | `UniBEV2/configs/Uni_BEV_v2/pdbev-samv2-aug-p9-r50-uni_v2.py`| Enhanced Scale-BEV configuration. |

---

### Unified Training with Real and Simulated Datasets (NuScenes, DeepAccident)

| **Model**         | **Configuration File**                                       | **Description**                                       |
|-------------------|-------------------------------------------------------------|-----------------------------------------------------|
| **DETR**          | `UniBEV2/configs/Uni_RealSim_v2/detr-r50-realsim_v2.py`      | DETR configuration for RealSim training.            |
| **PETR**          | `UniBEV2/configs/Uni_RealSim_v2/petr-r50-realsim-v2.py`      | PETR configuration for RealSim training.            |
| **BEVDet**        | `UniBEV2/configs/Uni_RealSim_v2/bevdet-r50-realsim-v2.py`    | BEVDet configuration for RealSim training.          |
| **BEVDepth**      | `UniBEV2/configs/Uni_RealSim_v2/bevdepth-r50-realsim-v2.py`  | BEVDepth configuration for RealSim training.        |
| **BEVFormer**     | `UniBEV2/configs/Uni_RealSim_v2/bevformer-r50-realsim_v2.py` | BEVFormer configuration for RealSim training.       |
| **FB-BEV**        | `UniBEV2/configs/Uni_RealSim_v2/fb-bevformer-r50-realsim-v2.py`| FB-BEV configuration for RealSim training.         |
| **PCBEV**         | `UniBEV2/configs/Uni_RealSim_v2/pdbev-r50-realsim-v2.py`     | Main PCBEV configuration for RealSim training.      |
| **Scale-BEV**     | `UniBEV2/configs/Uni_RealSim_v2/pdbev-samv2-aug-r50-realsim-v2.py`| Enhanced Scale-BEV configuration with SAM V2 for RealSim training. |

---

## Citation <a name="citation"></a>

If you find this project useful for your research, please consider citing our paper using the following BibTeX:

```bibtex
@inproceedings{scale-bev,
 title={Scaling Multi-Camera 3D Object Detection through Weak-to-Strong Eliciting}, 
 author={Hao LU, Jiaqi TANG, Xinli XU, Xu CAO, Yunpeng ZHANG, Guoqing WANG, Dalong DU, Hao CHEN, Yingcong CHEN},
 booktitle={https://arxiv.org/abs/2404.06700},
 year={2024},
}
```

---

## Notes

1. **Model Names**: The table in this document lists all configuration files alongside their respective model methods (e.g., PETR, BEVDet, etc.).
2. **Configuration Files**: The file paths provided are under `UniBEV2/configs/`, with subdirectories for `Uni_BEV_v2` and `Uni_RealSim_v2`.
3. **Description**: Each row in the table includes a brief description of the configuration file's purpose or the model it corresponds to.

---

This updated version should look clean and professional for your project documentation! If you need further adjustments, feel free to ask.
