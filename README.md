# Source Code for MetaMedSeg: Volumetric Meta-learning for Few-Shot Organ Segmentation accepted in DART Workshop at MICCAI 2022

Place data in a folder named "data" in the project root directory

### Preprocessing 
Extracting slices from .nii.gz volumes: 
- place unpacked folders project root directory 
- setup parameters in preprocessing_params.json file  
- run ```python preproces.py --params_file=preprocessing_params.json```
The resulting slices and masks will be saved to PROJECT_DIR/data/orgrans/task_name

### Transfer
Running pre-training sript for transfer: 
- setup parameters in pretrain_params.json file
- run ```python experiment_runner.py --exp_type=pretrain --params=pretrain_params.json```

#### Parameters guide 
- ```datasets``` - list of **tasks** to be used for pretraining (e.g. "vessel_cancer" - see folder names) 
- ```weights_path``` - path where the weights should be saved  
- ```max_samples``` - limit number of samples for each tasks to speed up training. Reasonable choice is 2000
- ```loss_type``` - loss function, available are: iou, bce, bce_weighted. Other losses can 
also be used but have not been tested. Default is iou.
- ```norm_type``` - type of normalizaton: "instance" or "batch" (default is "instance")

### Meta
Running meta-learning script: 
- setup parameters in meta_params.json file 
- run ```python experiment_runner.py --exp_type=meta --params=meta_params.json```

#### Parameters guide
- ```tasks_per_iteration``` - how many tasks should be sampled per meta-iteration 
- ```num_shots``` - number of shots to be used for each task 
- ```outer_epochs``` - number of meta-epochs 
- ```inner_lr```, ```inner_wd``` - parameters for local model in the inner loop 
- ```inter_epoch_decay``` - indicate whether the lr should decay over meta-epochs 
- ```meta_decay``` - indicate whether the meta-learning rate should decrease 
- ```val_shots``` - choose a number to reduce computation time, if set to null, then validated on the whole validation set 
- ```scheduler``` - select learning rate scheduler. Available are: "exp", "step", "plateau", "cyclic" 
- ```val_size``` - percentage of data to be hold out as a validation set (it's not necessary that all will be used)
- ```weighted_update_rule```:
    - null - standard reptile 
    - mean - computes the average model and rewards rare models
    - meta - computes the distances from meta-model and rewards more rare 
    - diff_based - penalizes models that learn more slowly 
    - loss_based - penalizes models with largest difference between train and val loss
    - ft_val - rewards models that provide better initial iou for target dataset 
- ```invert_weights``` - turns rewards into penalties and vice versa depending on weighted 
updated rule as described above
- ```norm_type``` - type of normalizaton: "instance" or "batch" (default is "instance")

### Fine-tuning
Fine-tuning: 
- setup parameters in ft_params.json file 
- run ```python experiment_runner.py --exp_type=ft --params=ft_params.json``` 

#### Parameters guide 
- ```meta_weights_path``` - path to meta-weighted obtained during meta-learning 
- ```transfer_path``` - path to weights obtained during pretraining (we can use any other weights as well, 
but the result will be saved in columns with prefix "transfer") 
- ```targets``` - dataset we're fine-tuning on. Default strategy is to pass just one target dataset 
like in the default setting file 
- ```report_path``` - path for saving the results  
-  other parameters are similar to pre-training parameters 

### Dependencies 
Standard setup (except for nibabel): 
- Pytorch 1.5.1+ (last tested on 1.7.1) 
- pandas, numpy, matplotlib, scikit-learn, skimage  
- nibabel 

### All tasks list: 
["brain_edema_FLAIR", "brain_edema_FLAIR", "brain_edema_t1gd", "brain_edema_T1w", "brain_edema_T2w",
"brain_e_tumour_FLAIR", "brain_e_tumour_t1gd", "brain_e_tumour_T1w", "brain_e_tumour_T2w", 
"brain_ne_tumour_FLAIR", "brain_ne_tumour_t1gd", "brain_ne_tumour_T1w", "brain_ne_tumour_T2w",
"colon", "heart", "hippocampus_anterior", "hippocampus_posterior", "lung", "pancreas",
"pancreas_cancer", "prostate_peripheral_T2", "prostate_transitional_T2", "spleen",
"vessel", "vessel_cancer"]

### Examples 
See example setups in "sample_setups" folder: 
- ```pretrain_params_config1``` - params to perform main experiment 
- ```meta_params_config1``` - config with the parameters that worked best
- ```ft_params_config1``` - basic fine-tuning setup to compare transfer vs meta
To run: 
```python experiment_runner.py --exp_type=ft --params_file=sample_setups\ft_params_config1.json```
