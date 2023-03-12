# Demo command
python run_pl.py --config configs/invarnet_movi.yaml --train-only_validate True

To train for 1 small epoch:
python run_pl.py --config configs/invarnet_movi.yaml --train-num_epochs 1


## Setup
- Create a new python environment.
- Install pytorch (any version that's not super old) with appropriate cuda version (A100 GPUs require some specific setup).
- pip install torch-scatter 
- pip install pytorch_lightening pyyaml h5py matplotlib opencv-python tensorboardX termcolor pytorch3d pyrender

## My Log
- Use batch renorm
- In DPINet, ReLU helped