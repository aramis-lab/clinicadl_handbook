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
# # Generate saliency maps on trained networks

Explaining black-box models can be useful to better understand their behaviour.
For more information on this complex topic, we highly recommend the review of
[Xie et al.](http://arxiv.org/abs/2004.14545).

In ClinicaDL, the most basic method of interpretability was implemented:
[gradients visualization](https://arxiv.org/pdf/1312.6034.pdf) (sometimes called
saliency maps). This method shows how the voxel intensities of an input image
should be modified in order to increase the value of a particular output node.
Here the output nodes correspond to a label: the first one represents AD whereas
the second represents CN.

This method can be performed on an individual or on a group fashion (in this
case it will be the mean value of all the individual saliency maps in the
group).