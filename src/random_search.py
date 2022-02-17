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
# # Launch a random search
# The previous section focused on a way to debug non-automated architecture
# search. However, if you have enough computational power, you may want to launch
# an automated architecture search to save your time. This is the point of the
# random search method of clinicadl.

# <div class="alert alert-block alert-info">
# <b>Non-optimal result:</b><p>
#     A random search may allow to find a better performing network, however there is no guarantee that this is the best performing network.
# </div>

# This notebook relies on the synthetic data generated in the previous notebook.
# If you did not run it, uncomment the following cell to generate the
# corresponding dataset.