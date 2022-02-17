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
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/dataOasis.tar.gz -o dataOasis.tar.gz
# !tar xf dataOasis.tar.gz
# %% [markdown]
# # Debug architecture search
# Previous sections were focusing on pre-built architectures available in
# ClinicaDL. These architectures were trained and validated on ADNI, and gave
# similar test results on ADNI, AIBL and OASIS. However, they might not be
# transferrable to other problems on other datasets using other modalities, and
# this is why may want to search for new architectures and hyperparameters.

# Looking for a new set of hyperparameters often means taking a lot of time
# training networks that are not converging. To avoid this pitfall, it is often
# advise to simplify the problem: focus on a subset of data / classification task
# that is more tractable than the one that is currently explored. This is the
# purpose of `clinicadl generate` which creates a set of synthetic, tractable data
# from real data to check that developed networks are working on this simple case
# before going further.

# <div class="alert alert-block alert-info">
# <b>Tractable data:</b><p>
#     In this notebook, we call tractable data a set of pairs of images and labels that can be easily classified. In ClinicaDL, tractable data is generated from real brain images and consist in creating two classes in which the intensitites of the left or the right part of the brain are decreased.</p>
#     <img src="images/generate.png" style="height: 200px;" alt="Schemes of synthetic tractable data">
# </div>

# If you ran the previous notebook, you must have a folder called
# `OasisCaps_example` in the current directory (Otherwise uncomment the next cell
# to download a local version of the necessary folders).