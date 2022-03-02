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

## Known issue

The rich progress bar implemented by Hook of mmdet, but there is no callback function of teardown or ctrl-c.
Therefore, if you use ctrl-c to stop current process, rich progress bar will not be stopped, and your cursor will be hidden.
Use the following command to reshow your cursor. By the way, any one want to fix this issue?

```bash
echo -e "\033[?25h"
```