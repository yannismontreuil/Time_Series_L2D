This repo is the implementation of the paper "Learning to Defer in Non-Stationary Time Series
via Switching State-Space Models".

## Requirements
see `requirements.txt` for required packages.

## Usage
all routers models are stored in `models/` folder.

```text
models/
├── factorized_slds.py       # Implementation of the proposed Factorized Switching Linear Dynamical System router
├── l2d_baseline.py          # Implementation of the L2D baseline router
├── linucb_baseline.py       # Implementation of the LinUCB baseline router
├── neuralucb_baseline.py    # Implementation of the NeuralUCB baseline router
├── router_model.py
├── router_model_corr.py
└── router_model_corr_em.py
```

To evaluate a router model, run the following command:

```bash
python slds_imm_router.py -c config/exp_synthetic_1.yaml
```

The parameters are stored in `configs/` folder.

Real-data example configs include:

```bash
python slds_imm_router.py -c config/config_melbourne.yaml
python slds_imm_router.py -c config/config_jena.yaml
```

> [!NOTE] 
> Note that parameters are not trained yet, you need to train the parameters first before evaluation.
