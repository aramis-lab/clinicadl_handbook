# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---
# %%
# Uncomment this cell if running in Google Colab
# !pip install clinicadl==1.2.0
# %% [markdown]
# # Generate saliency maps on trained networks

# Explaining black-box models can be useful to better understand their behaviour.
# For more information on this complex topic, we highly recommend the review of
# [Xie et al.](http://arxiv.org/abs/2004.14545).

# In ClinicaDL, the most basic method of interpretability was implemented:
# [gradients visualization](https://arxiv.org/pdf/1312.6034.pdf) (sometimes called
# saliency maps). This method shows how the voxel intensities of an input image
# should be modified in order to increase the value of a particular output node.
# Here the output nodes correspond to a label: the first one represents AD whereas
# the second represents CN.

# This method can be performed on an individual or on a group fashion (in this
# case it will be the mean value of all the individual saliency maps in the
# group).

# %% [markdown]
# ## Use of trivial datasets
# In the following, we are going to extract saliency maps from a model already trained on a 
# large trivial synthetic dataset. The second line download the mask used for trivial data
# generation, so we can compare them to the saliency maps obtained.

#%%
# Downloading pretrained model
!curl -k https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/model_trivial.tar.gz
!tar xf model_trivial.tar.gz

# Downloading masks used for trivial data generation
!curl -k https://aramislab.paris.inria.fr/files/data/masks/AAL2.tar.gz -o AAL2.tar.gz
!tar xf AAL2.tar.gz

#%% [markdown]
# In this trivial dataset, "AD" brains are atrophied according to the first mask while "CN" brains 
# are atrophied according to the second mask. The first mask include the whole cerebellum + the 
# left hemisphere while the second mask includes the right hemisphere.

#%%
from nilearn import plotting

plotting.plot_stat_map("AAL2/mask-1.nii", title="AD atrophy", cut_coords=(-50, 14), display_mode="yz")
plotting.plot_stat_map("AAL2/mask-2.nii", title="CN atrophy", cut_coords=(-50, 14), display_mode="yz")
plotting.show()

#%% [markdown]
# Saliency maps will be generated using trivial data generated from OASIS. If you did not run the notebook
# [Debug architecture search](generate.ipynb), you will need to run the following cell as well:

#%%
import os

os.makedirs("data", exist_ok=True)
# Download trivial CAPS
!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/synthetic.tar.gz -o synthetic.tar.gz
!tar xf synthetic.tar.gz -C data


#%% [markdown]
## Generate individual saliency maps
# Saliency maps on corresponding to one image can be computed with the following command:
# ```bash
# clinicadl interpret [OPTIONS] INPUT_MAPS_DIRECTORY DATA_GROUP NAME METHOD

# ```
# where:
# - `input_maps_directory` is the path to the pretrained model folder,
# - `data_group` (str) is a prefix to name the files resulting from the interpretation task.
# - `name` is the name of the interpretability job.
# - `method` (str) is the name of the saliency method (gradients or grad-cam).

#```{warning}
# For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory. 
# When performing a prediction, interpretation or tensor serialization the user must give a data group. 
# If this data group does not exist, the user MUST give a caps_path and a tsv_path. If this data group 
# already exists, the user MUST not give any caps_path or tsv_path, or set overwrite to True.
# ```

#%% [markdown]
# In the following we chose to generate saliency map based on the opposite labels:
# - the first command loads AD images and generates saliency maps based on CN node, 
# - the second command loads CN images and generates saliency maps based on AD node,

# Choosing the target node can be interesting in multi-class problems, but in binary classification 
# we expect the map of the opposite node to have opposite values than the ones in the corresponding 
# node (that is not very interesting).


#%%
# AD Target CN
!clinicadl interpret individual "model_trivial" "individual-AD_target-CN" \
    -np 0 --tsv_path "data/synthetic/data.tsv" --caps_dir "data/synthetic" \
    -d "AD" --target_diagnosis "CN" --nifti_template "AAL2/mask-1.nii"


#%%
# CN Target AD
!clinicadl interpret individual "model_trivial" "individual-CN_target-AD" \
    -np 0 --tsv_path "data/synthetic/data.tsv" --caps_dir "data/synthetic" \
    -d "CN" --target_diagnosis "AD" --nifti_template "AAL2/mask-1.nii"


#%% [markdown]
# This command will generate saliency maps for the model selected on validation 
# loss. You can obtain the same maps for the model selection on validation balanced 
# accuracy by adding the option `--selection best_balanced_accuracy`.

# One map is generated per image in the folder `gradients/selection/<name>`. These 
# images are organized in a similar way than the CAPS, with a `<participant_id>/<subject_id>` 
# structure:

#%%
!tree model_trivial/fold-0/gradients/best_loss/individual-AD_target-CN

#%%[markdown]
# Then we can plot the individual saliency maps to check which regions the CNN is focusing on.


#%%
def plot_individual_maps(diagnosis, target):
    import os
    from os import path
    
    subjects_path = f"/Users/camille.brianceau/aramis/clinicadl_handbook/src/models/maps_bis/split-0/best-loss/maps_bis_OASIS_interpret/interpret-test/mean_roi-0_map.pt"
    subjects_list = [subject for subject in os.listdir(subjects_path) 
                     if path.isdir(path.join(subjects_path, subject))]
    subjects_list.sort()
    for subject in subjects_list:
        map_path = path.join(subjects_path, subject, "ses-M00", "map.nii.gz")
        plotting.plot_stat_map(map_path, title=f"Saliency map of {subject}",
                               cut_coords=(-50, 14), display_mode="yz", threshold=10**-3)
    plotting.show()

print("Saliency maps of AD images based on CN nodes")
plot_individual_maps("AD", "CN")
print("Saliency maps of CN images based on AD nodes")
plot_individual_maps("CN", "AD")

#%% [markdown]
#These saliency maps are very noisy and may be difficult to interpret. This is why 

????????

#%% [markdown]
# ## Generate group saliency maps

# Saliency maps on corresponding to a group of images can be computed with the following command:
# ```bash
# clinicadl interpret group <model_path> <name>
# ```
# where:
# - `model_path` is the path to the pretrained model folder,
# - `name` is the name of the interpretability job.

# Default will try to load the data used for training (which is not possible here). Then you will need to set the following options:
# - `--tsv_path`, the path the TSV file with the subjects and sessions to use,
# - `--caps_dir`, the path to the CAPS in which the images defined in `tsv_path` are present.

#%%
# AD Target CN
!clinicadl interpret group "model_trivial" "group-AD_target-CN" \
    -np 0 --tsv_path "data/synthetic/data.tsv" --caps_dir "data/synthetic" \
    -d "AD" --target_diagnosis "CN" --nifti_template "AAL2/mask-1.nii"

#%%

# CN Target AD
!clinicadl interpret group "model_trivial" "group-CN_target-AD" \
    -np 0 --tsv_path "data/synthetic/data.tsv" --caps_dir "data/synthetic" \
    -d "CN" --target_diagnosis "AD" --nifti_template "AAL2/mask-1.nii"

#%% [markdown]
# This time the `gradients/<name>` folder will only contain one image corresponding to the mean 
# of individual saliency maps of the group.

#%%
!tree model_trivial/fold-0/gradients/best_loss/group-AD_target-CN

#%% [markdown]
# We can now plot the group saliency maps in the same way than for the individual ones.

#%%

def plot_group_maps(diagnosis, target):    
    map_path = f"model_trivial/fold-0/gradients/best_loss/group-{diagnosis}_target-{target}/map.nii.gz"
    plotting.plot_stat_map(map_path, title=f"Group saliency maps of {diagnosis} images",
                           cut_coords=(-50, 14), display_mode="yz", threshold=10**-3)
    plotting.show()

plot_group_maps("AD", "CN")
plot_group_maps("CN", "AD")

#%% [markdown]
# These maps are less noisy as the individual differences are less present and we can see more easily the main pattern.