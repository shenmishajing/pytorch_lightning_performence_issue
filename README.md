# pytorch lightning performance issue

see [issue](https://github.com/PyTorchLightning/pytorch-lightning/issues/12115)

## Install

- install pytorch, torchvision and mmcv from other official site.
- `pip install -r requirements.txt`
- `pip install -e .`

## Reproduce

- prepare coco dataset at `data/coco`.

### pytorch lightning version
- run `python tools/cli.py --config configs/faster_rcnn_pytorch_lightning.yaml`.

### mmdet version
- run `python tools/train.py configs/faster_rcnn_mmdet.py`.
