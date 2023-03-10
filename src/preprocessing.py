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

#%% [markdown]
# # Prepare your neuroimaging data

# There are different steps to perform before training your model or performing classification. In this notebook, we will see how to:

# 1. **Organize** your neuroimaging data.
# 2. **Preprocess** your neuroimaging data.
# 3. Check the preprocessing **quality**.
# 4. **Prepare data** by extracting tensors from your preprocessed data.

# %% [markdown]
# ## Organization of neuroimaging data: the Brain Imaging Data Structure (BIDS)

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


# Both OASIS and ADNI dataset contains imaging data in ANALYZE format and does not provide
# a BIDS version of the data. To solve this issue, clinica provides a
# [converter](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/OASIS2BIDS/)
# to automatically convert ANALYZE files into NIfTI following the BIDS standard.

# A command line instruction is enough to get the data in BIDS format:

# ```bash
# clinica convert oasis-to-bids <dataset_directory> <clinical_data_directory> <bids_directory>
# ```

# where:

#   - `dataset_directory` is the path to the original OASIS images' directory;
#   - `clinical_data_directory` is the path to the directory containing the `oasis_cross-sectional.csv` file;
#   - `bids_directory` is the path to the output directory, where the BIDS-converted version of OASIS will be stored.

# %% [markdown]
# ### Run the pipeline
# %%
# Download the example dataset of 4 images
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisDatabase.tar.gz -o OasisDatabase.tar.gz
# !tar xf OasisDatabase.tar.gz

# %%
# Convert the example dataset to BIDS
!clinica convert oasis-to-bids data_oasis/database/RawData data_oasis/database/ClinicalData data_oasis/BIDS

# %% [markdown]

# **Clinica** also provides other converters that works the same, suche as:
# [adni-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/ADNI2BIDS/),[aibl-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/AIBL2BIDS/),[habs-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/HABS2BIDS/),[nifd-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/NIFD2BIDS/),[oasis3-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/OASIS3TOBIDS/),[ukb-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/v0.7.2/Converters/UKBtoBIDS/),


# %% [markdown]
# # Why prepare data ?
# Preprocessing of neuroimaging data is essential before doing any experience and especially before training a neural network with these data. 
# - **Registration** help to standardize the neuroimaging data so that they are consistent across different subjects, scanners, and imaging modalities. This makes it easier for the deep neural network to learn patterns 
# and make accurate predictions. 
# - Preprocessing techniques such as **motion correction** and **noise reduction** can help to minimize sources of noise and improve the quality of the data 
# because due to a variety of factors, such as head motion, scanner artifacts, and biological variability, neuroimaging data can be noisy. 
# - Preprocessing can also be used to **extract features** from the neuroimaging data that are relevant to the task at hand. For example, if the goal is to classify brain regions based on their functional connectivity, preprocessing may involve 
# computing correlation matrices from the fMRI time series data. 
# - **Normalization** is another important preprocessing step for neuroimaging data that can help improve the performance of deep 
# neural networks. 
#
# Overall, preprocessing is essential in preparing neuroimaging data for deep neural network training. By standardizing and improving the quality of the data, these steps help to ensure that 
# the deep neural network can learn meaningful patterns and make accurate predictions.

# %% [markdown]

# Although convolutional neural networks (CNNs) have the potential to extract low-to-high level features from 
# raw images, a proper image preprocessing procedure is fundamental to ensure a good classification performance 
# (in particular for Alzheimer's disease (AD) classification where datasets are relatively small).
# In the context of deep learning-based classification, image preprocessing procedures often include:
# - **Bias field correction:** MR images can be corrupted by a low frequency and smooth signal caused by magnetic 
# field inhomogeneities. This bias field induces variations in the intensity of the same tissue in different locations 
# of the image, which deteriorates the performance of image analysis algorithms such as registration.
# - **Image registration:** Medical image registration consists of spatially aligning two or more images, either 
# globally (rigid and affine registration) or locally (non-rigid registration), so that voxels in corresponding 
# positions contain comparable information.
# Finally, a **Cropping** of the registered images can be performed to remove the background and to reduce the c
# omputing power required when training deep learning models.

# %% [markdown]
# This notebook presents three possible preprocessing steps using the Clinica software: 
# - `t1-linear`: Affine registration of T1w images to the MNI standard space
# - `t1-volume`: Volume-based processing of T1-weighted MR images with SPM
# - `pet-linear`: Spatial normalization to the MNI space and intensity normalization of PET images

# %% [markdown]
# <a id='preprocessing:t1-linear'></a>
# ## Image preprocessing with the `t1-linear` pipeline
# For this tutorial, we propose a "minimal preprocessing" (as described in [(Wen et al., 2020)](https://doi.org/10.1016/j.media.2020.101694)) 
# implemented in the [`t1-linear` pipeline](http://www.clinica.run/doc/Pipelines/T1_Linear/) using the [ANTs](http://stnava.github.io/ANTs/) 
# software package [(Avants et al., 2014)](https://doi.org/10.3389/fninf.2014.00044). This preprocessing includes:
# - **Bias field correction** using the N4ITK method [(Tustison et al., 2010)](https://doi.org/10.1109/TMI.2010.2046908)
# - **Affine registration** to the MNI152NLin2009cSym template (Fonov et al., [2011](https://doi.org/10.1016/j.neuroimage.2010.07.033), 
# [2009](https://doi.org/10.1016/S1053-8119(09)70884-5) ) in MNI space with the SyN algorithm [(Avants et al., 2008)](https://doi.org/10.1016/j.media.2007.06.004).
# - **Cropping** resulting in final images of size 169×208×179 with 1 mm3 isotropic voxels.

# If you run this notebook locally, please check that ANTs is correctly installed. If it is not the case, uncomment the three following lines and run it.

# %%
# # !/bin/bash -c "$(curl -k https://aramislab.paris.inria.fr/files/software/scripts/install_conda_ants.sh)"
# # from os import environ
# # environ['ANTSPATH']="/usr/local/bin"

# %% [markdown]
# These steps can be run with this simple command line:
# ```bash
#   clinica run t1-linear <bids_directory> <caps_directory>
# ```
# where:

# - `bids_directory` is the input folder containing the dataset in a [BIDS](http://www.clinica.run/doc/BIDS/) hierarchy,
# - `caps_directory` is the output folder containing the results in a [CAPS](http://www.clinica.run/doc/CAPS/) hierarchy.
# %%[markdown]
# ```{warning}
# The following command can take some time to execute, depending on the
# configuration of your host machine. Running in a classical **Colab** instance
# can take up to 30 min.

# We will increase a little bit the computation capacity using 2 cores with the
# `--n_procs 2` flag. Since there are 4 images, you can set `--n_procs 4` if
# your computer can handle this.
# ```
# %% [markdown]
# ### Run the pipeline
# %%
!clinica run t1-linear data_oasis/BIDS data_oasis/CAPS --n_procs 2
# %% [markdown]
# Once the pipeline has been run, the necessary outputs for the next steps are saved using a specific suffix: `_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz`. 
# %% [markdown]
# (If you failed to obtain the preprocessing using the `t1-linear` pipeline,
# please uncomment the next cell)
# %%
# #!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps1.tar.gz -o OasisCaps1.tar.gz
# #!tar xf OasisCaps1.tar.gz
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

suffix_caps = '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
suffix_bids = '_T1w.nii.gz'
sub1 = 'data_oasis/BIDS/sub-OASIS10016/ses-M00/anat/sub-OASIS10016_ses-M00' + suffix_bids 
sub2 = 'data_oasis/CAPS/subjects/sub-OASIS10016/ses-M00/t1_linear/sub-OASIS10016_ses-M00' + suffix_caps

sub3 = 'data_oasis/BIDS/sub-OASIS10304/ses-M00/anat/sub-OASIS10304_ses-M00' + suffix_bids
sub4 = 'data_oasis/CAPS/subjects/sub-OASIS10304/ses-M00/t1_linear/sub-OASIS10304_ses-M00' + suffix_caps

plotting.plot_anat(sub3, title="raw data: sub-OASIS10304")
plotting.plot_anat(sub4, title="preprocessed data: sub-OASIS10304")

plotting.plot_anat(sub1, title="raw data: sub-OASIS10016")
plotting.plot_anat(sub2, title="preprocessed data: sub-OASIS10016")

plotting.show()

# %% [markdown]
# <a id='preprocessing:pet-linear'></a>
# ## Image preprocessing with the `pet-linear` pipeline

# This pipeline performs spatial normalization to the MNI space and intensity normalization of PET images. Its steps include:

# - **Affine registration** to the MNI152NLin2009cSym template [Fonov et al., 2011, 2009] in MNI space with the SyN algorithm [Avants et al., 2008] from the ANTs software package [Avants et al., 2014];
# - **Intensity normalization** using the average PET uptake in reference regions resulting in a standardized uptake value ratio (SUVR) map;
# - **Cropping** of the registered images to remove the background.

# %% [markdown]
# ```{warning}
# You need to have performed the t1-linear pipeline on your T1-weighted MR images.
# ```

# %% [markdown]
# The pipeline can be run with the following command line:


# ```bash
#   clinica run pet-linear [OPTIONS] BIDS_DIRECTORY CAPS_DIRECTORY ACQ_LABEL
#                        {pons|cerebellumPons|pons2|cerebellumPons2}
#````
# where:

# - `bids_directory` is the input folder containing the dataset in a [BIDS](http://www.clinica.run/doc/BIDS/) hierarchy;
# - `caps_director` is the output folder containing the results in a [CAPS](http://www.clinica.run/doc/CAPS/) hierarchy;
# - `acq_label` is the label given to the PET acquisition, specifying the tracer used (trc-<acq_label>). It can be for instance '18FFDG' for 18F-fluorodeoxyglucose or '18FAV45' for 18F-florbetapir;
# - The reference region is used to perform intensity normalization (i.e. dividing each voxel of the image by the average uptake in this region) resulting in a standardized uptake value ratio (SUVR) map. 
# It can be cerebellumPons or cerebellumPons2 (used for amyloid tracers) and pons or pons2 (used for FDG). See [PET introduction](clinical) for more details about masks versions.

# %%[markdown]
# ```{warning}
# The following command can take some time to execute, depending on the
# configuration of your host machine. Running in a classical **Colab** instance
# can take up to 30 min.

# We will increase a little bit the computation capacity using 2 cores with the
# `--n_procs 2` flag. Since there are 4 images, you can set `--n_procs 4` if
# your computer can handle this.
# ```
# %% [markdown]
# ### Run the pipeline
# Please uncomment the next cells to download a dataset of pet images of 4 subjects from ADNI in a BIDS format (convert to BIDS with `clinica convert adni-to-bids)
# %%
# #!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps1.tar.gz -o OasisCaps1.tar.gz
# #!tar xf OasisCaps1.tar.gz
# %%
!clinica run pet-linear data_adni/BIDS data_adni/CAPS --n_procs 2
# %% [markdown]
#Once the pipeline has been run, the necessary outputs for the next steps are saved using a specific suffix: `_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-<ref-region>_pet.nii.gz`. 

# %% [markdown]
# For example, we can see the difference between raw images and processed images from our dataset:
# %%
from nilearn import plotting

suffix_caps = '_pet_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellulPons2_pet.nii.gz'
suffix_bids = '_pet.nii.gz'
sub1 = 'data_adni/BIDS/sub-ADNI002S0685/ses-M72/pet/sub-ADNI002S0685_ses-M72' + suffix_bids 
sub2 = 'data_adni/CAPS/subjects/sub-ADNI002S0685/ses-M72/pet_linear/sub-ADNI002S0685_ses-M72' + suffix_caps

sub3 = 'data_adni/BIDS/sub-ADNI126S4896/ses-M00/pet/sub-ADNI126S4896_ses-M00' + suffix_bids
sub4 = 'data_adni/CAPS/subjects/sub-ADNI126S4896/ses-M00/pet_linear/sub-ADNI126S4896_ses-M00' + suffix_caps

plotting.plot_anat(sub3, title="raw data: sub-ADNI002S0685")
plotting.plot_anat(sub4, title="preprocessed data: sub-ADNI002S0685")

plotting.plot_anat(sub1, title="raw data: sub-ADNI126S4896")
plotting.plot_anat(sub2, title="preprocessed data: sub-ADNI126S4896")

plotting.show()

# %% [markdown]
# <a id='preprocessing:t1-volume'></a>
# ## Image preprocessing with the `t1-volume` pipeline

# This pipeline performs four main processing steps on T1-weighted MR images using the SPM software:
# - **Tissue segmentation**: bias correction and spatial normalization to MNI space This corresponds to the Segmentation procedure of SPM 
# that simultaneously performs tissue segmentation, bias correction and spatial normalization, a procedure also known as "Unified segmentation" [Ashburner and Friston, 2005](https://doi.org/10.1016/j.neuroimage.2005.02.018).
# - **Inter-subject registration using Dartel**: a group template is created using DARTEL, an algorithm for diffeomorphic image registration, 
# from the subjects' tissue probability maps in native space (usually GM, WM and CSF tissues) obtained at the previous step. Here, not only 
# the group template is obtained, but also the deformation fields from each subject's native space into the Dartel template space. This is 
# achieved by wrapping the Run Dartel procedure from SPM [Ashburner, 2007](https://doi.org/10.1016/j.neuroimage.2007.07.007).
# - **Dartel template to MNI**:  Once the transformation from the subject’s T1-weighted MRI image to the Dartel template has been computed, the 
# T1-weighted MRI image of each subject can be transported to the MNI space. More precisely, for a given subject, its flow field into the 
# Dartel template is combined with the transformation of the Dartel template into MNI space, and the resulting transformation is applied to 
# the subject’s different tissue maps. As a result, all the images are in a common space, providing a voxel-wise correspondence across subjects. 
# This is achieved by wrapping the Dartel2MNI procedure from SPM [Ashburner, 2007](https://doi.org/10.1016/j.neuroimage.2007.07.007).
# - **Atlas statistics**: A set of anatomical regions is obtained from different atlases in MNI space (list of available atlases [here](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Atlases/). 
# The average gray matter density (also in MNI space) is then computed in each of the regions.


# %% [markdown]
# If you run this notebook locally, please check that SPM12 is correctly installed. If it is not the case, uncomment the three following lines and run it.
# You can find how to install these software packages [here](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Third-party/).


# %% [markdown]
# These steps can be run with this simple command line:
# ```bash
#   clinica run t1-volume <bids_directory> <caps_directory> <group_label>
# ```
# where:

# - `bids_directory` is the input folder containing the dataset in a [BIDS](http://www.clinica.run/doc/BIDS/) hierarchy,
# - `caps_directory` is the output folder containing the results in a [CAPS](http://www.clinica.run/doc/CAPS/) hierarchy.
# - `group_label` is the user-defined identifier for the provided group of subjects.

# %%[markdown]
# ```{warning}
# The following command can take some time to execute, depending on the
# configuration of your host machine. Running in a classical **Colab** instance
# can take up to 30 min.

# We will increase a little bit the computation capacity using 2 cores with the
# `--n_procs 2` flag. Since there are 4 images, you can set `--n_procs 4` if
# your computer can handle this.
# ```

# %% [markdown]
# ### Run the pipeline
# You can run this pipeline on the data_oasis and save the ouptput in the same CAPS as with the `clinica run t1-linear` pipeline. 
# %%
!clinica run t1-volume data_oasis/BIDS data_oasis/CAPS OasisT1Volume --n_procs 2
# %% [markdown]
# Once the pipeline has been run, the necessary outputs for the next steps are saved in the CAPS directory.
# If you want more information about the outputs of this pipeline, please check the [clinica documentaiton](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Volume/).
# %% [markdown]
# (If you failed to obtain the preprocessing using the `t1-volume` pipeline,
# please uncomment the next cell)
# %%
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps1.tar.gz -o OasisCaps1.tar.gz
# !tar xf OasisCaps1.tar.gz

# %% [markdown]
# For example, we can see the difference between raw images and processed images from our dataset:
# %%
from nilearn import plotting

suffix_caps = '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
suffix_bids = '_T1w.nii.gz'
sub1 = 'data_oasis/BIDS/sub-OASIS10016/ses-M00/anat/sub-OASIS10016_ses-M00' + suffix_bids 
sub2 = 'data_oasis/CAPS/subjects/sub-OASIS10016/ses-M00/t1_linear/sub-OASIS10016_ses-M00' + suffix_caps

sub3 = 'data_oasis/BIDS/sub-OASIS10304/ses-M00/anat/sub-OASIS10304_ses-M00' + suffix_bids
sub4 = 'data_oasis/CAPS/subjects/sub-OASIS10304/ses-M00/t1_linear/sub-OASIS10304_ses-M00' + suffix_caps

plotting.plot_anat(sub3, title="raw data: sub-OASIS10304")
plotting.plot_anat(sub4, title="preprocessed data: sub-OASIS10304")

plotting.plot_anat(sub1, title="raw data: sub-OASIS10016")
plotting.plot_anat(sub2, title="preprocessed data: sub-OASIS10016")

plotting.show()


# %% [markdown]
# # Quality check of your preprocessed data

# %% [markdown]
# From the 3 visualizations above, we can see that after the preprocessing, some images have some
# missing skin voxels on top of the brain i.e. these images are slightly cropped.
# Besides, we did not compare them to the [MNI152NLin2009cSym
# template](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html)
# to evaluate the quality of the registration.

# OASIS-1 dataset contains 416 images  and ADNI more than 3000 so quality check of the whole datasets can be
# very time consuming. The next section gives you some ideas on how to keep only
# images correctly preprocessed, when running in a large dataset.

# %% [markdown]
# To automatically assess the quality of the **t1-linear** preprocessing, we propose to use a
# pretrained network which learnt to classify images that are adequately
# registered to a template from others for which the registration failed. This
# procedure is adaptated from [(Fonov et al,
# 2022)](https://doi.org/10.1016/j.neuroimage.2022.119266), using their
# pretrained models. The original code of [(Fonov et al,
# 2022)](https://doi.org/10.1016/j.neuroimage.2022.119266) can be found on
# [GitHub](https://github.com/vfonov/DARQ).

# The **pet-linear** quality check is comming in the next release of clinicadl !
#  
# The **t1-volume** quality check procedure is based on thresholds on different statistics that were empirically linked to images of bad quality. Three steps are performed to 
# remove images with the following characteristics:

# - a maximum value below 0.95,
# - a percentage of non-zero values below 15% or higher than 50%,
# - a similarity with the DARTEL template around the frontal lobe below 0.40. The similarity corresponds to the normalized mutual information. This allows checking that 
# the eyes are not included in the brain volume.


# The quality check can be run with the following command line:
# ```
# !clinicadl quality-check <preprocessing> <caps_directory> <output_path>
# ```
# where:

# - `preprocessing` corresponds to the preprocessing pipeline whose outputs will be checked (`t1-linear` or `pet-linear` or `t1-volume`),
# - `caps_directory` is the folder containing the results of the preprocessing pipeline in a [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy,
# - `output_path` is the path to the output TSV file (or directory for `t1-volume`) containing QC results.
# 
#
# for `t1-volume quality-check` you need to add this argument
# - `group_label` is the identifier for the group of subjects used to create the DARTEL template. You can check which groups are available in the `groups/` folder of your caps_directory.
#
#  This pipeline outputs 4 files:
# - QC_metrics.tsv containing the three QC metrics for all the images,
# - pass_step-1.tsv including only the images which passed the first step,
# - pass_step-2.tsv including only the images which passed the two first steps,
# - pass_step-3.tsv including only the images which passed all the three steps.
##
# !!! note
# Quality checks pipelines are all different and depends on the chosen preprocessing. They should not be applied to other preprocessing procedures as the results may not be reliable.

# %%[markdown]
# ### Run the pipeline
# %%
# quality-check for t1-linear preprocessing
!clinicadl quality-check t1-linear data_oasis/CAPS data_oasis/QC_result_t1.tsv --no-gpu --threshold 0.8

# %%
# quality-check for pet-linear preprocessing (comming soon)
!clinicadl quality-check pet-linear data_adni/CAPS data_adni/QC_result_pet.tsv 18FFDG cerebellumPons2 --no-gpu

# %%
# quality-check for t1-volume preprocessing
!clinicadl quality-check t1-volume data_oasis/CAPS data_oasis/QC_result_t1V.tsv OasisT1Volume --no-gpu

# %% [markdown]
# ```{warning}
# These quality check can be really conservative and may keep some images that are not of good quality. 
# You may want to check the images kept to assess if their quality is good enough for your application.

# %%
import pandas as pd
df_T1 = pd.read_csv("data_oasis/QC_results_t1.tsv", sep="\t")
#%%
print(df_T1)


# %% [markdown]
# Based on these TSV file, participant `OASIS10304` should be discarded for the
# rest of your anlysis. If you compare its registration with [MNI152NLin2009cSym
# template](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html),
# you will see that temporal regions are misaligned as well as occipital regions
# and cerebellum leading to this low probabilty value. ;-)

#%%
df_pet = pd.read_csv("data_adni/QC_results_pet.tsv", sep="\t")
print(df_pet)

# %% [markdown]
# Based on these TSV file, participant `OASIS10304` should be discarded for the
# rest of your anlysis. If you compare its registration with [MNI152NLin2009cSym
# template](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html),
# you will see that temporal regions are misaligned as well as occipital regions
# and cerebellum leading to this low probabilty value. ;-)

#%%
df_t1V = pd.read_csv("data_adni/QC_results_t1V/f", sep="\t")
print(df_t1V)

# To display more nicely the output we implemented in this notebook
# `display_table`:
def display_table(table_path):
    """Custom function to display the clinicadl quality-check output"""
    import pandas as pd

    QC_results_df = pd.read_csv(table_path, sep='\t')
    QC_results_df.set_index(["participant_id","session_id"], drop=True, inplace=True)
    columns = ["pass_probability", "pass"]

    # Print formatted table
    format_columns = ["subjects", "session", "pass_probability", "pass"]
    format_df = pd.DataFrame(index=QC_results_df.index, columns=format_columns)
    for idx in QC_results_df.index.values:    
        row_str = "%i; %i; %.1f ± %.1f [%.1f, %.1f]; %iF / %iM; %.1f ± %.1f [%.1f, %.1f]; 0: %i, 0.5: %i, 1: %i, 2:%i, 3:%i" % tuple([OASIS_analysis_df.loc[idx, col] for col in columns])
        row_list = row_str.split(';')
        format_df.loc[idx] = row_list

    format_df.index.name = None
    display(format_df)


display_table("../data/analysis.tsv")



