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

# %% [markdown]
# # Training for custom task
#
# ## Customize your experiment!

# You want to train your custom architecture, with a custom input type or
# preprocessing on other labels? Please fork and clone the [github
# repo](https://github.com/aramis-lab/clinicadl]) to add your changes.


# ### Add a custom model  <i class="fa fa-hourglass-start " aria-hidden="true"></i>
#
# Write your model class in `clinicadl/tools/deep_learning/models` and import it
# in `clinicadl/tools/deep_learning/models/__init__.py`.

# <div class="alert alert-block alert-info">
# <b>Autoencoder transformation:</b><p>
#     Your custom model can be transformed in autoencoder in the same way as
#     predefined models. To make it possible, implement the convolutional part in
#     <code>features</code> and the fully-connected layer in
#     <code>classifier</code>. See predefined models as examples.
# </div>

# ### Add a custom input type <i class="fa fa-hourglass-start " aria-hidden="true"></i> <i class="fa fa-hourglass-start " aria-hidden="true"></i> <i class="fa fa-hourglass-start " aria-hidden="true"></i>

# Input types that are already provided in `clinicadl` are image, patch, roi and
# slice. To add a custom input type, please follow the steps detailed below:
# * Choose a mode name for this input type (for example default ones are image,
# patch, roi and slice).
# * Add your dataset class in `clinicadl/tools/deep_learning/data.py` as a child
# class of the abstract class `MRIDataset`.
# * Create your dataset in `return_dataset` by adding:

# ```python
# elif mode==<mode_name>:
#     return <dataset_class>(
#         input_dir,
#         data_df,
#         preprocessing=preprocessing,
#         transformations=transformations,
#         <custom_args>
#     )
# ```

# * Add your custom subparser to `train` and complete `train_func` in `clinicadl/cli.py`.

# ### Add a custom preprocessing <i class="fa fa-hourglass-start " aria-hidden="true"></i> <i class="fa fa-hourglass-start " aria-hidden="true"></i>
#
# Define the path of your new preprocessing in the `_get_path` method of
# `MRIDataset` in `clinicadl/tools/deep_learning/data.py`. You will also have to
# add the name of your preprocessing pipeline in the general command line by
# modifying the possible choices of the `preprocessing` argument of
# `train_pos_group` in `cli.py`.

# ### Change the labels <i class="fa fa-hourglass-start " aria-hidden="true"></i>
#
# You can launch a classification task with clinicadl using any labels. The
# input tsv files must include the columns `participant_id`, `session_id` and
# `diagnosis`. If the column `diagnosis` does not contain the labels described
# in this tutorial (AD, CN, MCI, sMCI, pMCI), you can add your own label name
# associated to a class value in the `diagnosis_code` of the class `MRIDataset`
# in `clinicadl/tools/deep_learning/data.py`.
# %% [markdown]
# ```{admonition} Suggestion!
# :class: tip
# Do not hesitate to ask for help on
# [GitHub](https://github.com/aramis-lab/clinicadl/issues/new) or propose a new pull
# request!
# ```
