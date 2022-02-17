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
# !pip install clinicadl==0.2.1
# %% [markdown]
# # Train your own model
#
# his section explains how to train a CNN on OASIS data that were processed in the previous sections. 

# ```{warning}
# If you do not have access to a GPU, training the CNN may require too much time.
# However, you can execute this notebook on Colab to run it on a GPU.
# ```

# If you already know the models implemented in `clinicadl`, you can directly jump
# to the {ref}`last section <custom_exp>` to implement your own custom experiment!
# %%
import torch

# Check if a GPU is available
print('GPU is available', torch.cuda.is_available())
# %% [markwdown]
# <div class="alert alert-block alert-warning">
# <b>Data used for training:</b><p>
#     Because they are time-costly, the preprocessing steps presented in the beginning of this tutorial were only executed on a subset of OASIS-1, but obviously two participants are insufficient to train a network! To obtain more meaningful results, you should retrieve the whole <a href="https://www.oasis-brains.org/">OASIS-1</a> dataset and run the training based on the labels and splits performed in the previous section.</p>
#     <p>Of course, you can use another dataset, but then you will have to perform again <a href="./label_extraction.ipynb">the extraction of labels and data splits</a> on this dataset.</p>
# </div>
# %% [markdown]
# ## Using `clinicadl train`
#
# Training a neural network requires a lot of inputs from the user. For
# clinicadl the main inputs are:
# * The kind of task to train (*classification*, *reconstruction* and
#   *regression*).
# * The folder containing the input images in CAPS format.
# * A file containing information on the preprocessing  `PREPROCESSING_JSON`.
# * A folder wiht files in TSV format to define where the train and validation are stored.
# * A folder to the path where the MAPS will be stored.
#
# Multiple options can be entered by using the option `-c, --config_file`, a
# file in format TOML, a human-readable format.
#
# The help for the `clinicadl train` functionality:
#
# %%
! clinicadl train -h

# %%
# Download the data
#! wget --no-check-certificate --progress=bar:force -O ../data/RandomCaps.tar.gz https://aramislab.paris.inria.fr/files/data/databases/tuto2/RandomCaps.tar.gz
! tar xf ../data/RandomCaps.tar.gz -C ../data/
# %% [markdown]
# ## Example 1: training using the whole image
# Lets suposse that we want to train a network of preprocessed images using
# slices.
# Our images comes from a synthetic dataset containing images with random noise,
# obtained with `clinicadl generate`. 
# %% [markdown]
# The configuration file `train_cnofig.toml`
# ```[toml]
# # Config file for tutotiel
# [Cross_validation]
# n_splits = 2
# split = []
#
# [Classification]
# label = "sex"
# 
# [Optimization]
# epochs = 1
#
# [Data]
# multi_cohort = false
# diagnoses = ["CN"]
# ```
# Mani other variables can be configured, [see the
# documentation](https://clinicadl.readthedocs.io/en/stable/Train/Introduction/).
# %%
! clinicadl train classification ../data/caps_v2021 extract_image_t1_linear.json ../data/labels_list/train ../data/out -c ../data/train_config.toml 


# %% [markdown]
# ## Example 2: training using only slices of the image
# %%
! clinicadl train classification ../data/random_example extract_slice.json ../data/labels_list/train ../data/out -c ../data/train_config.toml 
# %% [markdown]
# ## Visualization of the `MAPS`:
# %%
! tree -L 4 ../data/out

# %%
