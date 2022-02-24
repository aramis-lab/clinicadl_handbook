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
# Uncomment this cell if running in Google Colab
# !pip install clinicadl==0.2.1
# %% [markdown]
# # Perfom classification using pretrained models

# <SCRIPT SRC='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></SCRIPT>
# <SCRIPT>MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}})</SCRIPT> 

# This notebook shows how to perform classification on preprocessed data using pretrained models described in ([Wen et al, 2020](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591)).

# ## Structure of the pretrained models

# All the pretrained model folders are organized as follows:
# <pre>
# <b>results</b>
# ├── commandline.json
# ├── <b>fold-0</b>
# ├── ...
# └── <b>fold-4</b>
#     ├── <b>models</b>
#     │      └── <b>best_balanced_accuracy</b>
#     │          └── model_best.pth.tar
#     └── <b>cnn_classification</b>
#            └── <b>best_balanced_accuracy</b>
#                └── validation_{patch|roi|slice}_level_prediction.tsv
# </pre>
# This file system is a part of the output of `clinicadl train` and `clinicadl classify` relies on three files:
# <ul>
#     <li> <code>commandline.json</code> contains all the options that were entered for training (type of input, architecture, preprocessing...).</li>
#     <li> <code>model_best.pth.tar</code> corresponds to the model selected when the best validation balanced accuracy was obtained.</li>
#     <li> <code>validation_{patch|roi|slice}_level_prediction.tsv</code> is specific to patch, roi and slice frameworks and is necessary to perform <b>soft-voting</b>  and find the label at the image level in unbiased way. Weighting the patches based on their performance of input data would bias the result as the classification framework would exploit knowledge of the test data.</li>
# </ul>


# <div class="admonition note" name="html-admonition" style="background: lightgreen; padding: 10px">
# <p class="title">Tip</p>
#     You can use your own previuolsy trained model (if you have used PyTorch
#     for that). Indeed, PyTorch stores model weights in a file with extension
#     <i>pth.tar</i>. You can place this file into the <i>models</i> folder and
#     try to follow the same structure that is described above. You also need to
#     fill a <i>commandline.json</i> file with all the parameters used during
#     the training (see <a
#     href="https://clinicadl.readthedocs.io/en/latest/Train/Introduction/#outputs">ClinicaDL
#     documentation</a>) for further info.</p>
# </div>

# :::{tip}
# You can use your own previuolsy trained model (if you have used PyTorch for
# that). Indeed, PyTorch stores model weights in a file with extension
# `pth.tar`. You can place this file into the `models` folder and try to follow
# the same structure that is described above. You also need to fill a
# `commandline.json` file with all the parameters used during the training (see
# [ClinicaDL
# documentation](https://clinicadl.readthedocs.io/en/latest/Train/Introduction/#outputs)
# for further info.
# :::

# <div class="alert alert-block alert-info">
# <b>Soft voting:</b><p>
# For classification tasks that take as input a part of the MRI volume
# (<i>patch, roi or slice</i>), an ensemble operation is needed to obtain the
# label at the image level.</p>
# <p>For example, size and stride of 50 voxels on linear preprocessing leads to
# the classification of 36 patches, but they are not all equally meaningful.
# Patches that are in the corners of the image are mainly composed of background
# and skull and may be misleading, whereas patches within the brain may be more
# useful.</p>
# <img src="./images/patches.png">
# <p>Then the image-level probability of AD <i>p<sup>AD</sup></i> will be:</p>
#
# $$ p^{AD} = {\sum_{i=0}^{35} bacc_i * p_i^{AD}}.$$
#
# where:<ul>
# <li> <i>p<sub>i</sub><sup>AD</sup></i> is the probability of AD for patch <i>i</i></li>
# <li> <i>bacc<sub>i</sub></i> is the validation balanced accuracy for patch <i>i</i></li>
# </ul>
# </div>
# %% [markdown]
# ## Download the pretrained models

# :::{warning} 
# For the sake of the demonstration, this tutorial uses truncated versions of
# the models, containing only the first fold.
# :::

# In this notebook, we propose to use 4 specific models , all of them where trained to predict the classification task AD vs CN. (The experiment corresponding to the pretrained model in eTable 4 of the paper mentioned above is shown below):

# 1. **3D image-level model**, pretrained with the baseline data and initialized with an autoencoder (_cf._ exp. 3).
# 2. **3D ROI-based model**, pretrained with the baseline data and initialized with an autoencoder (_cf._ exp. 8).
# 3. **3D patch-level model**, multi-cnn, pretrained with the baseline data and initialized with an autoencoder (_cf._ exp. 14).
# 4. **2D slice-level model**, pretrained with the baseline data and initialized with an autoencoder (_cf._ exp. 18).

# Commands in the next code cell will automatically download and uncompress these models.
# %%
# Download here the pretrained models stored online
# Model 1
!curl -k https://aramislab.paris.inria.fr/clinicadl/files/models/v0.2.0/model_exp3_splits_1.tar.gz  -o model_exp3_splits_1.tar.gz
!tar xf model_exp3_splits_1.tar.gz

# Model 2
!curl -k https://aramislab.paris.inria.fr/clinicadl/files/models/v0.2.0/model_exp8_splits_1.tar.gz  -o model_exp8_splits_1.tar.gz
!tar xf model_exp8_splits_1.tar.gz

# Model 3
!curl -k https://aramislab.paris.inria.fr/clinicadl/files/models/v0.2.0/model_exp14_splits_1.tar.gz  -o model_exp14_splits_1.tar.gz
!tar xf model_exp14_splits_1.tar.gz

# Model 4
!curl -k https://aramislab.paris.inria.fr/clinicadl/files/models/v0.2.0/model_exp18_splits_1.tar.gz  -o model_exp18_splits_1.tar.gz