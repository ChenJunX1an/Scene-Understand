# Scene Understand

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/), [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene). Thanks for their wonderful works.


## 🔨 Preparation

- Prepare the environment:
  
  ```shell
  conda create -n chat-scene python=3.9.17
  conda activate chat-scene
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLM backbone:
  -  We use Vicuna-7B v1.5 in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5).

  - Change the `llama_model_path` in [run.sh](./scripts/run.sh) to the path of `vicuna-7b-v1.5`.
  

- Annotations and extracted features:
  
  Please follow the instructions in [preprocess](preprocess/).


## 🤖 Training and Inference

- Training
  - Modify [run.sh](scripts/run.sh):
    ```python
    train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=False
    ```

    <details>
    <summary> Explanation of "train_tag" and "val_tag" </summary>

    - Use `#` to seperate different datasets

    - Datasets:
      - `scanrefer`: [ScanRefer](https://github.com/daveredrum/ScanRefer) Dataset
      - `scan2cap`: [Scan2Cap](https://github.com/daveredrum/Scan2Cap) Dataset
      - `scanqa`: [ScanQA](https://github.com/ATR-DBI/ScanQA) Dataset
      - `sqa3d`: [SQA3D](https://github.com/SilongYong/SQA3D) Dataset
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset
      - `nr3d_caption`: A captioning dataset originated from [Nr3D](https://github.com/referit3d/referit3d).
      - `obj_align`: A dataset originated from ScanRefer to align the object identifiers with object tokens.

    </details>
  - Run: `bash scripts/run.sh`


- Inference
  
  - Modify [run.sh](scripts/run.sh): (We provide the pretrained checkpoint in [Google Drive](https://drive.google.com/file/d/1Ziz7Be9l6MEbn3Qmlyr9gv42C0iJQgAn/view?usp=sharing))
  
    ```python
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=True
    pretrained_path="/path/to/pretrained_model.pth"
    ```
  
  - Run: `bash scripts/run.sh`
  

## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{huang2024chat,
  title={Chat-scene: Bridging 3d scene and large language models with object identifiers},
  author={Huang, Haifeng and Chen, Yilun and Wang, Zehan and Huang, Rongjie and Xu, Runsen and Wang, Tai and Liu, Luping and Cheng, Xize and Zhao, Yang and Pang, Jiangmiao and others},
  journal={Proceedings of the Advances in Neural Information Processing Systems, Vancouver, BC, Canada},
  year={2024}
}
@article{wang2023chat,
  title={Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes},
  author={Wang, Zehan and Huang, Haifeng and Zhao, Yang and Zhang, Ziang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2308.08769},
  year={2023}
}
```

Stay tuned for our project. 🔥

If you have any questions or suggestions, feel free to drop us an email (`huanghaifeng@zju.edu.cn`, `wangzehan01@zju.edu.cn`) or open an issue.

## 😊 Acknowledgement

Thanks to the open source of the following projects:

(Multi-modal) LLMs:
[LLaMA](https://github.com/facebookresearch/llama), 
[Vicuna](https://github.com/lm-sys/FastChat),
[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), 
[LEO](https://github.com/embodied-generalist/embodied-generalist)

3D Datasets:
[ScanNet](https://github.com/ScanNet/ScanNet), 
[ScanRefer](https://github.com/daveredrum/ScanRefer), 
[ReferIt3D](https://github.com/referit3d/referit3d), 
[Scan2Cap](https://github.com/daveredrum/Scan2Cap), 
[ScanQA](https://github.com/ATR-DBI/ScanQA), 
[SQA3D](https://github.com/SilongYong/SQA3D), 
[Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP)

Detectors:
[PointGroup](https://github.com/dvlab-research/PointGroup), 
[Mask3D](https://github.com/JonasSchult/Mask3D),
[DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

Representations:
[ULIP](https://github.com/salesforce/ULIP), 
[Uni3D](https://github.com/baaivision/Uni3D),
[DINOv2](https://github.com/facebookresearch/dinov2)

3D Models:
[vil3dref](https://github.com/cshizhe/vil3dref),
[OpenScene](https://github.com/pengsongyou/openscene)

