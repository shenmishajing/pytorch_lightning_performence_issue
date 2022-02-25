# pytorch lightning performance issue

see [issue](https://github.com/PyTorchLightning/pytorch-lightning/issues/12115)

## Install

- install pytorch, torchvision and mmcv from other official site.
- `pip install -r requirements.txt`
- `pip install -e .`

## Reproduce

- prepare coco dataset at `data/coco`.
- run `python main.py --config configs/faster_rcnn.py`.
