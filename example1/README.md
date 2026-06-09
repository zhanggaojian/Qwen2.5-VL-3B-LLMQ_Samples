# Quantization recipe of LLM(from Qwen2.5-VL)

## 系统依赖

- Computer with NVGPU(VRAM>64GB is fine、RAM > 32GB is fine)
- Ubuntu 22.04
- Compile & install python3(3.10~3.12 is fine)
  - verified on python setup by virtualenv/virtualenvwrapper
  - anaconda环境尚未验证
- Download the QAIRT SDK from Qualcomm® Software Center
  - https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community?osArch=Any&osType=All&version=2.42.0.251225
  - QAIRT version 2.36~2.46 is fine

## Calibration Sets & pip requirements

- Download llava_v1_5_mix665k.json from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- Download coco/train2017 from http://images.cocodataset.org/zips/train2017.zip
- Put the above files like this:
```shell
# tree -L 2 /data/huggingface/hf_dataset/
/data/huggingface/hf_dataset/
|-- coco
|   `-- train2017
|-- llava_v1_5_mix665k.json
```
- pip install -r [req.txt](../req.txt)



