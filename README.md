CS224W Project: Can GNNs learn rigid body dynamics?

Chaitanya Patel, Preey Shah, Shreya Gupta

# Run a demo inference
`python run_eval.py --config configs/invarnet_movi.yaml --exp_path ./data/run_1 --model-ckpt_path last_ckpt --eval-viz True`

# Training
`python run_pl_training.py --config configs/invarnet_movi.yaml --exp_path ./data/test`

This will train on dummy dataset that comes with this repo. Edit config file to train according to your preferences.

## Setup
- Create a new python environment.
- Install pytorch (any version that's not super old) with appropriate cuda version (A100 GPUs require some specific setup).
- pip install torch-scatter (follow instruction on pyg installation site)
- pip install torch_geometric (follow instruction on pyg installation site)
- pip install pytorch_lightening pyyaml h5py matplotlib opencv-python tensorboardX termcolor pytorch3d pyrender


# Basic Explanation of the Code
- `data` directory has dummy kubric dataset with 2 train and 2 validation sequences. It also has a trained model in `run_1` directory.
- `dataset` module has dataset loading and graph creation utilities.
  - `movi_dataset.py` has code to download, post-process movi dataset. It also has a dataloader to load basic sequence data. `data_utils.py` has corresponding helper utilities.
  - `particle_graph.py` has a helper class to add particles and create nodes/edges with their features. `paritcle_data_utils.py` has helpful utility functions for it.
- `model` directory has code for GNN message passing.
- `train/invarnet_module.py` has PytorchLightening module for our model. It incapsulates all functionalities of our model. It implements trainig step, validation step, rollout and corresponding metric evaluation.