# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
# Uncomment the next lines if running in Google Colab
# # !pip install clinicadl==1.2.0

# %% [markdown]
# # Prepare your neuroimaging data (pet-linear)

# There are different steps to perform before training your model or performing classification. In this notebook, we will see how to:

# 1. **Organize** your neuroimaging data.
# 2. **Preprocess** your neuroimaging data.
# 3. Check the preprocessing **quality**.
# 4. **Prepare data** by extracting tensors from your preprocessed data.

# %% [markdown]
# # Organization of neuroimaging data: the Brain Imaging Data Structure (BIDS)

# Before processing your neuroimaging data, several steps may be needed. These
# steps can include converting the images to a format readable by neuroimaging
# software tools (e.g. converting to NIfTI) and organizing your files in a
# specific way. Several tools will require that your clinical and imaging data
# follow the **Brain Imaging Data Structure (BIDS)** [(Gorgolewski et al.,
# 2016)](https://doi.org/10.1038/sdata.2016.44). The BIDS standard is based on a
# file hierarchy rather than on a database management system, thus facilitating
# its deployment. Thanks to its clear and simple way to describe neuroimaging
# and behavioral data, it has been easily adopted by the neuroimaging community.
# Organizing a dataset following the BIDS hierarchy simplifies the execution of
# neuroimaging software tools.  

##

# Here is a general overview of the BIDS structure. If you need more details,
# please check the
# [documentation](https://bids-specification.readthedocs.io/en/latest/) on the
# [website](http://bids.neuroimaging.io/).

# <pre>
# BIDS_Dataset/
# ├── participants.tsv
# ├── sub-CLNC01/
# │   │   ├── ses-M00/
# │   │   │   └── anat/
# │   │   │       └── <b>sub-CLNC01_ses-M00_T1w.nii.gz</b>
# │   │   └── sub-CLNC01_sessions.tsv
# ├── sub-CLNC02/
# │   │   ├── ses-M00/
# │   │   │   ├── anat/
# │   │   │   │   └── <b>sub-CLNC02_ses-M00_T1w.nii.gz</b>
# │   │   │   └── pet/
# │   │   │       └── <b>sub-CLNC02_ses-M00_trc-18FFDG.nii.gz</b>
# │   │   └── sub-CLNC02_sessions.tsv
# └──  ...
# </pre>


# The OASIS dataset contains imaging data in ANALYZE format and does not provide
# a BIDS version of the data. To solve this issue, [Clinica provides a
# converter](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/OASIS2BIDS/)
# to automatically convert ANALYZE files into NIfTI following the BIDS standard.

# A command line instruction is enough to get the data in BIDS format:

# ```bash
# clinica convert oasis-to-bids <dataset_directory> <clinical_data_directory> <bids_directory>
# ```

# where:

#   - `dataset_directory` is the path to the original OASIS images' directory;
#   - `clinical_data_directory` is the path to the directory containing the `oasis_cross-sectional.csv` file;
#   - `bids_directory` is the path to the output directory, where the BIDS-converted version of OASIS will be stored.

##
# **Clinica** also provides other converters suche as: [adni-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/ADNI2BIDS/),
# [aibl-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/AIBL2BIDS/),
# [habs-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/HABS2BIDS/),
# [nifd-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/NIFD2BIDS/),
# [oasis3-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/OASIS3TOBIDS/),
# [ukb-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/UKBtoBIDS/),

# %%
# Download the example dataset of 4 images
!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/AdniDatabase.tar.gz -o AdniDatabase.tar.gz
!tar xf AdniDatabase.tar.gz


# %%
# Convert the example dataset to BIDS
!clinica convert adni-to-bids AdniDatabase/RawData AdniDatabase/ClinicalData AdniBids_example
# %% [markdown]
# <a id='preprocessing:pet-linear'></a>
# ## pet-linear - Linear processing of PET images

# This pipeline performs spatial normalization to the MNI space and intensity normalization of PET images. Its steps include:

# - affine registration to the MNI152NLin2009cSym template [Fonov et al., 2011, 2009] in MNI space with the SyN algorithm [Avants et al., 2008] from the ANTs software package [Avants et al., 2014];
# - intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake value ratio (SUVR) map;
# - cropping of the registered images to remove the background.

# ### Prerequisite
# You need to have performed the t1-linear pipeline on your T1-weighted MR images.
##
# If you run this notebook locally, please check that ANTs is correctly installed. If it is not the case, uncomment the three following lines and run it.

# %%

# # !/bin/bash -c "$(curl -k https://aramislab.paris.inria.fr/files/software/scripts/install_conda_ants.sh)"
# # from os import environ
# # environ['ANTSPATH']="/usr/local/bin"

# %% [markdown]
# The pipeline can be run with the following command line:


# ```bash
#   clinica run pet-linear [OPTIONS] BIDS_DIRECTORY CAPS_DIRECTORY ACQ_LABEL
#                        {pons|cerebellumPons|pons2|cerebellumPons2}
#````
# where:

# - BIDS_DIRECTORY is the input folder containing the dataset in a [BIDS](http://www.clinica.run/doc/BIDS/) hierarchy;
# - CAPS_DIRECTORY is the output folder containing the results in a [CAPS](http://www.clinica.run/doc/CAPS/) hierarchy;
# - ACQ_LABEL is the label given to the PET acquisition, specifying the tracer used (trc-<acq_label>). It can be for instance '18FFDG' for 18F-fluorodeoxyglucose or '18FAV45' for 18F-florbetapir;
# - The reference region is used to perform intensity normalization (i.e. dividing each voxel of the image by the average uptake in this region) resulting in a standardized uptake value ratio (SUVR) map. It can be cerebellumPons or cerebellumPons2 (used for amyloid tracers) and pons or pons2 (used for FDG). See PET introduction for more details about masks versions.

# %%[markdown]
# ```{warning}
# The following command can take some time to execute, depending on the
# configuration of your host machine. Running in a classical **Colab** instance
# can take up to 30 min.

# We will increase a little bit the computation capacity using 2 cores with the
# `--n_procs 2` flag. Since there are 4 images, you can set `--n_procs 4` if
# your computer can handle this.
# ```
# %%
!clinica run pet-linear ./AdniBids_example ./AdniCaps_example 18FFDG cerebellumPons2 --n_procs 2
# %% [markdown]
#Once the pipeline has been run, the necessary outputs for the next steps are saved using a specific suffix:  
# `_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref-region>_pet.nii.gz`. 

# %% [markdown]
# (If you failed to obtain the preprocessing using the `pet-linear` pipeline,
# please uncomment the next cell)
# %%
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps1.tar.gz -o OasisCaps1.tar.gz TOCHANGE
# !tar xf OasisCaps1.tar.gz


# %% [markdown]
# ```{warning}
# The registration algorithm provided by ANTs exposes some reproducibility issues
# when running in different environments. The outputs are "visually" very close
# but not exactly the same. For further information and some clues on how to
# reduce the variability please read this
# [page](https://github.com/ANTsX/ANTs/wiki/antsRegistration-reproducibility-issues).
# ```
# %% [markdown]
# For example, we can see the difference between raw images and processed images from our dataset:
# %%
from nilearn import plotting

suffix_caps = '_trc-18FFDG_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz'
suffix_bids = '_pet.nii.gz'
sub1 = 'AdniBids_example/sub-ADNI002S0685/ses-M72/pet/sub-ADNI002S0685_ses-M72' + suffix_bids 
sub2 = 'AdniCaps_example/subjects/sub-ADNI002S0685/ses-M72/pet_linear/sub-ADNI002S0685_ses-M72' + suffix_caps

sub3 = 'AdniBids_example/sub-ADNI941S1202/ses-M00/pet/sub-ADNI941S1202_ses-M00' + suffix_bids
sub4 = 'AdniCaps_example/subjects/sub-ADNI941S1202/ses-M00/pet_linear/sub-ADNI941S1202_ses-M00' + suffix_caps

#plotting.plot_anat(sub3, title="raw data: sub-ADNI02S0685")
plotting.plot_anat(sub4, title="preprocessed data: sub-ADNI002S0685")

#plotting.plot_anat(sub1, title="raw data: sub-ADNI941S1202")
plotting.plot_anat(sub2, title="preprocessed data: sub-ADNI941S1202")

plotting.show()

# %% [markdown]
# From the visualization above, we can see that after the preprocessing, the first image have some
# missing skin voxels on top of the brain i.e. these images are slightly cropped.
# Besides, we did not compare them to the [MNI152NLin2009cSym
# template](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html)
# to evaluate the quality of the registration.

# ADNI dataset contains ??? images so quality check of the whole dataset can be
# very time consuming. The next section gives you some ideas on how to keep only
# images correctly preprocessed, when running in a large dataset.

# %% [markdown]
# ## Quality check of your preprocessed data

## PET QUALITY CHECK COMING NEXT VERSION OF CLINICADL

# The quality check can be run with the following command line:
# ```
# !clinicadl quality-check <pet-linear> <caps_directory> <output_path>
# ```
# where:

# - `preprocessing` corresponds to the preprocessing pipeline whose outputs will be checked (`t1-linear` or `t1-volume` or `pet-linear`),
# - `caps_directory` is the folder containing the results of the preprocessing pipeline in a [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy,
# - `output_path` is the path to the output TSV file containing QC results.
##
# !!! note
# Quality check are all diferent depends on the chosen preprocessing. 
# - `quality-check pet-linear` procedure is based on a metric showing the diference with a template from the MNI.
# %%
!clinicadl quality-check pet-linear OasisCaps_example QC_result.tsv 18FFDG cerebellumPons2 --no-gpu
# %% [markdown]
# After execution of the quality check procedure, the `QC_result.tsv` file will
# look like this:

# | participant_id | session_id | pass_probability   |
# |----------------| -----------|--------------------|
# | sub-OASIS10016 | ses-M00    | 0.9936990737915039 |
# | sub-OASIS10109 | ses-M00    | 0.9772214889526367 |
# | sub-OASIS10363 | ses-M00    | 0.7292165160179138 |
# | sub-OASIS10304 | ses-M00    | <font color="red">0.1549495905637741</font> |
##

# Based on this TSV file, participant `OASIS10304` should be discarded for the
# rest of your anlysis. If you compare its registration with [MNI152NLin2009cSym
# template](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html),
# you will see that temporal regions are misaligned as well as occipital regions
# and cerebellum leading to this low probabilty value. ;-)
##
# ```{warning}
#   All the `quality-check` pipelines are not the ground truth and must be seen as an advice, a little manual 
#   quality ckeck need to be done as well.
# ````
# %% [markdown]
# ##  Tensor extraction with the `prepare-data` pipeline

# Once the dataset has been preprocessed, we need to obtain files suited for the
# training phase. This pipeline prepares images generated by Clinica to be used with the PyTorch
# deep learning library [(Paszke et al., 2019)](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library).
# Four types of tensors are proposed: 3D images, 3D patches, 3D ROI or 2D slices.
##
# This pipeline selects the preprocessed images, extract the "tensors", and write
# them as output files for the entire images, for each slice, for each roi or for each patch.

# You simply need to type the following command line:

# ```bash
# clinicadl prepare-data {image|patch|slice|roi} <caps_directory> <pet-linear>
# ```
# where:

# - `caps_directory` is the folder containing the results of the [`t1-linear`
# pipeline](#preprocessing:t1-linear) and the output of
# the present command, both in a CAPS hierarchy.
# - `tensor_format` is the format of the extracted tensors. You can choose
# between `image` to convert to PyTorch tensor the whole 3D image, `patch` to
# extract 3D patches, `roi` to extract a list of regions defined by masks at 
# the root in CAPS_DIRECTORY and `slice` to extract 2D slices from the image.
##
# Output files are stored into a new folder (inside the CAPS) and follows a struture like this:

# ```text
# deeplearning_prepare_data
# ├── image_based
# │   └── pet_linear
# │       └── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_pet.pt
# ├── patch_based
# │   └── pet_linear
# │       ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_patchsize-50_stride-50_patch-0_pet.pt
# │       ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_patchsize-50_stride-50_patch-1_pet.pt
# │       ├── ...
# │       └── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_patchsize-50_stride-50_patch-N_pet.pt
# ├── roi_based
# │   └── pet_linear
# │       ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_roi-<roi_name_0>_pet.pt
# │       ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_roi-<roi_name_1>_pet.pt
# │       ├── ...
# │       └── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_roi-<roi_name_N>_pet.pt
# └── slice_based
#     └── pet_linear
#         ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_axis-axi_channel-rgb_slice-0_pet.pt
#         ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_axis-axi_channel-rgb_slice-1_pet.pt
#         ├── ...
#         ├── sub-<participant_label>_ses-<session_label>_trc-<acq_label>_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref_region>_axis-axi_channel-rgb_slice-N_pet.pt

# ```

# In short, there is a folder for each feature (**image, slice or patch**) and
# inside the numbered tensor files with the corresponding feature. 
# %% [markdown]
# <div class="alert alert-info">

# **Note:** You can choose to only extract the tensors for the whole images
# *(`clinica run deeplearning-prepare-data <caps_directory> image` ) and continue
# *working with one single file per subject/session. 
    
# The package `clinicadl` is able to extract patches or slices _on-the-fly_ (from
# one single file) when running training or inference tasks. The downside of this
# approach is that, depending on the size of your dataset, you have to make sure
# that you have enough memory ressources in your GPU card to host the full
# images/tensors for all your data. 

# If the memory size of the GPU card you use is too small, we suggest you to
# extract the patches and/or the slices using the proper `tensor_format` option of
# the command described above.
# </div>

# %% [markdown]
# To perform the feature extraction for our dataset, run the following cell:     
# %%
!clinicadl prepare-data image ./AdniCaps_example pet-linear -tsv ./AdniCaps_example/data.tsv
# %% [markdown]
# At the end of this command, a new directory named `deeplearning_prepare_data` is
# created inside each subject/session of the CAPS structure. We can easily verify:
# %%
!tree -L 3 ./OasisCaps_example/subjects/sub-OASIS10*/ses-M00/deeplearning_prepare_data/
# %%
