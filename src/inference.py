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
#
# <div class="admonition tip" name="html-admonition" style="background: lightgreen; padding: 10px">
# <p class="title">Tip</p>
#     <p> You can use your own previuolsy trained model (if you have used PyTorch
#     for that). Indeed, PyTorch stores model weights in a file with extension
#     <i>pth.tar</i>. You can place this file into the <i>models</i> folder and
#     try to follow the same structure that is described above. You also need to
#     fill a <i>commandline.json</i> file with all the parameters used during
#     the training (see <a
#     href="https://clinicadl.readthedocs.io/en/latest/Train/Introduction/#outputs">ClinicaDL
#     documentation</a>) for further info.</p>
# </div>
#
# <div class="alert alert-block alert-info">
# <p class="title">Soft voting</p>
#    For classification tasks that take as input a part of the MRI volume
#    (<i>patch, roi or slice</i>), an ensemble operation is needed to obtain the
#    label at the image level.</p>
#    <p>For example, size and stride of 50 voxels on linear preprocessing leads to
#    the classification of 36 patches, but they are not all equally meaningful.
#    Patches that are in the corners of the image are mainly composed of background
#    and skull and may be misleading, whereas patches within the brain may be more
#    useful.</p>
#    <img src="../images/patches.png">
#    <p>Then the image-level probability of AD <i>p<sup>AD</sup></i> will be:</p>
#    $$ p^{AD} = {\sum_{i=0}^{35} bacc_i * p_i^{AD}}. $$
#    where:<ul>
#    <li> <i>p<sub>i</sub><sup>AD</sup></i> is the probability of AD for patch <i>i</i></li>
#    <li> <i>bacc<sub>i</sub></i> is the validation balanced accuracy for patch <i>i</i></li>
#    </ul>
# </div>
# %% [markdown]
# ## Download the pretrained models

# <div class="admonition warning" name="html-admonition" style="background: lightgreen; padding: 10px">
# <p class="title">Warning</p>
# For the sake of the demonstration, this tutorial uses truncated versions of
# the models, containing only the first fold.
# </div>

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
# %% [markdown]
# ## Run `clinicadl classify`

# Running classification on a dataset is extremly simple using `clinicadl`. In
# this case, we will continue using the data preprocessed in the [previous
# notebook](./preprocessing). The models have been trained exclusively on the ADNI
# dataset, all the subjects of OASIS-1 can be used to evaluate the model (without
# risking data leakage).

# If you ran the previous notebook, you must have a folder called
# `OasisCaps_example` in the current directory (Otherwise uncomment the next cell
# to download a local version of the necessary folders).
# %%
!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps2.tar.gz -o OasisCaps2.tar.gz
!tar xf OasisCaps2.tar.gz
!curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisBids.tar.gz -o OasisBids.tar.gz
!tar xf OasisBids.tar.gz
# %% [markdown]
# In the following steps we will classify the images using the pretrained models.
# The input necessary for `clinica classify` are:
# * a CAPS directory (`OasisCaps_example`),
# * a tsv file with subjects/sessions to process, containing the diagnosis (`participants.tsv`),
# * the path to the pretrained model,
# * an output prefix for the output file.

# Some optional parameters includes:
# * the possibility of classifying non labeled data (without known diagnosis),
# * the option to use previously extracted patches/slices.

# ```{warning}
# If your computer is not equiped with a GPU card add the option `-cpu` to the
# command.
# ```
# %% [markdown]
# First of all, we need to generate a valid tsv file. We use the tool `clinica iotools`:
# %%
!clinica iotools merge-tsv OasisBids_example OasisCaps_example/participants.tsv
# %% [markdown]
# Then, we can run the classifier for the **image-level** model:
# %%
# Execute classify on OASIS dataset
# Model 1
!clinicadl classify ./OasisCaps_example ./OasisCaps_example/participants.tsv ./model_exp3_splits_1 'test-Oasis'
# %% [markdown]
# The predictions of our classifier for the subjects of this dataset are shown next:
# %%
import pandas as pd

predictions = pd.read_csv("./model_exp3_splits_1/fold-0/cnn_classification/best_balanced_accuracy/test-Oasis_image_level_prediction.tsv", sep="\t")
predictions.head()
# %% [markdown]
# Note that 0 corresponds to the **CN** class and 1 to the **AD**. It is also
# important to remember that the last two images/subjects performed badly when
# running the quality check step.

# `clinica classify` also produces a file containing different metrics (accuracy,
# balanced accuracy, etc.) for the current dataset. It can be displayed by running
# the next cell:
# %%
metrics = pd.read_csv("./model_exp3_splits_1/fold-0/cnn_classification/best_balanced_accuracy/test-Oasis_image_level_metrics.tsv", sep="\t")
metrics.head()
# %% [markdown]
# In the same way, we can process the dataset with all the other models:
# %%
# Model 2 3D ROI-based model
!clinicadl classify ./OasisCaps_example ./OasisCaps_example/participants.tsv ./model_exp8_splits_1 'test-Oasis'

predictions = pd.read_csv("./model_exp8_splits_1/fold-0/cnn_classification/best_balanced_accuracy/test-Oasis_image_level_prediction.tsv", sep="\t")
predictions.head()
# %%
# Model 3 3D patch-level model
!clinicadl classify ./OasisCaps_example ./OasisCaps_example/participants.tsv ./model_exp14_splits_1 'test-Oasis'

predictions = pd.read_csv("./model_exp14_splits_1/fold-0/cnn_classification/best_balanced_accuracy/test-Oasis_image_level_prediction.tsv", sep="\t")
predictions.head()
# %%
# Model 4 2D slice-level model
!clinicadl classify ./OasisCaps_example ./OasisCaps_example/participants.tsv ./model_exp18_splits_1 'test-Oasis'

predictions = pd.read_csv("./model_exp18_splits_1/fold-0/cnn_classification/best_balanced_accuracy/test-Oasis_image_level_prediction.tsv", sep="\t")
predictions.head()