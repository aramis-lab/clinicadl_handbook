
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---∏

# %% [markdown]
# ## How to use `clinicadl train`
#
# ClinicaDL is able to train networks using different kind of inputs (3D images,
# 3D patches or 2D slices).
#
# A single network can learnt diiferent task: *classification*, *reconstruction* and *regression*.
#
# All information necessary to reproduce the train (network architecture, hyperparameters, weights)
# is stored in the MAPS.
# %% [markdown]
# ## Using `clinicadl train`
#
# Training a neural network requires a lot of inputs from the user. For clinicadl the main inputs are:
# * The kind of task to train (*classification*, *reconstruction* and
#   *regression*).
# * The folder containing the input images in CAPS format.
# * A file containing information on the preprocessing  `PREPROCESSING_JSON`.
# * A folder wiht files in TSV format to define where the train and validation are stored.
# * A folder to the path where the MAPS will be stored.
#
# Multiple options can be entered by using the option `-c, --config_file`, a
# file in format TOML, a human-readable format.
# %% [markdown]
# ## Train slices
# Example of the command line for train slices.
! clinicadl train -h 

# %%
