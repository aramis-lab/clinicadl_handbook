# -*- coding: utf-8 -*
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
# !pip install clinicadl==1.2.0

# %%
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/dataOasis.tar.gz -o dataOasis.tar.gz
# !tar xf dataOasis.tar.gz
# %% [markdown]
# # Define your population

# This notebook is an introduction to tools that can be used to identify
# relevant samples and split them between training, validation and test cohorts.
# **This step is mandatory preliminary to training to avoid issues such as lack
# of clinical meaning or data leakage**. 

# In the following, we will see how to split these samples between training,
# validation and test sets using tools available in `clinica` and `clinicadl`.

# In this section we will work on a subset of 100 subjects of the OASIS dataset
# (and a subset of 100 subjects of the ADNI dataset). You can find the list of
# participants that have passed the quality check in the data folder
# (oasis_after_qc.tsv and adni_after_qc.tsv).


# You can also download these files by uncomment the next cell:
# %%
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisBids.tar.gz -o OasisBids.tar.gz
# !tar xf OasisBids.tar.gz 

# %% [markdown]
# These tsv files allow to prepare the set of data to train a neural network, you can follow this notebook even without having access to these data but
# if you want to do the rest of the notebooks, you will have to download the data from [ADNI](https://adni.loni.usc.edu/) or [OASIS](https://oasis-brains.org/) 
# because it is not possible to separate a set of 4 
# images without data leakage.


# %% [markdown]
# ## Get metadata from a BIDS hierarchy with `clinica iotools`
# ### Gather BIDS and CAPS data into a single TSV file
#
# In a BIDS hierarchy, demographic, clinical and imaging metadata are stored in
# TSV files located at different levels of the hierarchy depending on whether
# they are specific to a subject (e.g. gender), a session (e.g. diagnosis) or a
# scan (e.g. acquisitions parameters).

# The following command line can be used to merge all the metadata in a single
# TSV file:
# ```bash
# clinica iotools merge-tsv <bids_directory> <output_tsv>
# ```
# where:
# - `bids_directory` is the input folder containing the dataset in a BIDS
# hierarchy.
# - `output_tsv` is the path of the output tsv. If a directory is specified
# instead of a file name, the default name for the file created will be
# `merge-tsv.tsv`.

# %% [markdown]
# In the [preprocessing section](./preprocessing.ipynb) an example BIDS of 4
# subjects from OASIS-1 was generated (if you did not run interactively that
# section, download the dataset by uncomment the next cell):
# %%
# !curl -k https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisBids.tar.gz -o OasisBids.tar.gz
# !tar xf OasisBids.tar.gz
# %% [markdown]
# Execute the following command to gather metadata included in this BIDS:
# %%
# Merge meta-data information
!clinica iotools merge-tsv data_oasis/BIDS_example data_oasis/merged.tsv -tsv data_oasis/after_qc.tsv

#%%
# !clinica iotools merge-tsv data_adni/BIDS_example data_adni/merged.tsv -tsv data_adni/after_qc.tsv
# %% [markdown]
# ### Check missing modalities for each subject
#
# We want to restrict the list of the sessions used to those including a T1-MR
# image. Then the following command is needed to identify which modalities are
# present for each session:
#
# ```bash
# clinica iotools check-missing-modalities <bids_directory> <output_directory>
# ```
# where:
# - `bids_directory` is the input folder of a BIDS compliant dataset.
# - `output_directory` is the output folder.

# This pipeline does not have an option to give a list of subject/session so you
# will check the missing modalities of all the datasets.
# 
#  Execute the following command to find which sessions include a T1-MR image on
#  the example BIDS of OASIS:
# %%
# Find missing modalities
!clinica iotools check-missing-modalities data_oasis/BIDS_example data_oasis/missing_mods
#%%
# !clinica iotools check-missing-modalities data_adni/BIDS data_adni/missing_mods
# %% [markdown]
# The output of this command, `data/<dataset>_missing_mods`, is a folder in
# which a series of tsv files is written (one file per session label containing
# one row per subject and one column per modality).
# %% [markdown]
# ## Prepare metadata with `clinicadl tsvtools` 
#
# %% [markdown]
# ### Get the labels
#
# The 3 labels described in the [first part of the course](../clinical) (AD, CN,
# MCI) can be extracted with ClinicaDL using the command:
#
# ```bash
# clinicadl tsvtools get-labels bids_directory results_tsv
# ```
# where:
# - `bids_directory` the input folder containing the dataset in a BIDS
# hierarchy.
# - `results_path` is the path to the tsv file.

# ```{tip}
# You can increase the verbosity of the command by adding -v flag(s).
# ```

# The bids directory is mandatory to run the `clinica iotools merge-tsv` and
# `clinica iotools check-missing-modalities` inside this pipelines if it has not
# been done before.  However if you already have run these pipelines, the path
# is not mandatory anymore so you can put anything and add the options
# `--merged_tsv` and `--missing_mods`. It will avoid the pipeline to re-run
# them.

# %% [markdown]
# If you failed running the `clinica iotools` command you can download the
# output by running the following cell:
# %%
# TO DOWNLOAD
# %% [markdown]
# By default the pipeline only extracts the AD and CN labels, which corresponds
# to the only available labels in OASIS. Run the following cell to extract them
# in a new file `labels.tsv` from the restricted version of OASIS:
# %%
!clinicadl tsvtools get-labels data_oasis/BIDS_example --merged_tsv data_oasis/merged.tsv --missing_mods data_oasis/missing_mods --restriction_tsv data_oasis/after_qc.tsv
# %%
# !clinicadl tsvtools get-labels data_adni/BIDS_example --merged_tsv data_adni/merged.tsv --missing_mods data_adni/missing_mods --restriction_path data_adni/after_qc.tsv
# %% [markdown]
# This tool writes a unique TSV file containing the labels asked by the user.
# They are stored in the column named diagnosis.

# <div class="alert alert-block alert-info">
# <b>Restriction path:</b><p>
#     At the end of the command line another restriction was given to extract the
#     labels only from sessions in <code>data/OASIS_after_qc.tsv</code>. This tsv
#     file corresponds to the output of the <a
#     href="./preprocessing.ipynb">quality check procedure</a> that was manually
#     cut to only keep the sessions passing the quality check. It depends on the
#     preprocessing: here it concerns a run of <code>t1-linear</code>.</p>
# </div>

# %% [markdown]
# ### Analyze the population

# The age bias in OASIS is well known and this is why the youngest CN
# participants were previously excluded. However, other biases may exist,
# especially after the quality check of the preprocessing which removed sessions
# from the dataset. Thus it is crucial to check before going further if there
# are other biases in the dataset.

# ClinicaDL implements a tool to perform a demographic and clinical analysis of
# the population:

# ```bash
# clinicadl tsvtool analysis <merged_tsv> <data_tsv> <results_path>
# ```
# where:
# - `merged_tsv` is the output file of the `clinica iotools merge-tsv`command.
# - `data_tsv` is the output file of `clinicadl tsvtool getlabels|split|kfold`).
# - `results_path` is the path to the tsv file that will be written (filename included).


# The following command will extract statistical values on the populations for
# each diagnostic label. Based on those it is possible to check that the dataset
# is suitable for the classification task.
# %%
# Run the analysis on OASIS
!clinicadl tsvtools analysis data_oasis/merged.tsv data_oasis/labels.tsv data_oasis/analysis.tsv
# %%
# Run the analysis on ADNI
#!clinicadl tsvtool analysis data_adni/merged.tsv data_adni/labels.tsv data_adni/analysis.tsv
# %%
def display_table(table_path):
    """Custom function to display the clinicadl tsvtool analysis output"""
    import pandas as pd

    OASIS_analysis_df = pd.read_csv(table_path, sep='\t')
    OASIS_analysis_df.set_index("group", drop=True, inplace=True)
    columns = ["n_subjects", "n_scans",
               "mean_age", "std_age", "min_age", "max_age",
               "sexF", "sexM",
               "mean_MMSE", "std_MMSE", "min_MMSE", "max_MMSE",
               "CDR_0", "CDR_0.5", "CDR_1", "CDR_2", "CDR_3"]

    # Print formatted table
    format_columns = ["subjects", "scans", "age", "sex", "MMSE", "CDR"]
    format_df = pd.DataFrame(index=OASIS_analysis_df.index, columns=format_columns)
    for idx in OASIS_analysis_df.index.values:    
        row_str = "%i; %i; %.1f ± %.1f [%.1f, %.1f]; %iF / %iM; %.1f ± %.1f [%.1f, %.1f]; 0: %i, 0.5: %i, 1: %i, 2:%i, 3:%i" % tuple([OASIS_analysis_df.loc[idx, col] for col in columns])
        row_list = row_str.split(';')
        format_df.loc[idx] = row_list

    format_df.index.name = None
    display(format_df)
# %%
display_table("analysis.tsv")
# %% [markdown]
# There is no significant bias on age anymore, but do you notice any other
# problems? 

# <div class="alert alert-block alert-warning">
# <b>Demographic bias:</b>
#     <p>There is still a difference in sex distribution and the network could
#     learn a bias on sex such as "women are cognitively normal" and "men are
#     demented". However, there are too few images in OASIS to continue removing
#     sessions to equilibrate the groups.
#     
#     To check that such bias is not learnt, it is possible to run a logistic
#     regression after training between sex and the predicted label to check if
#     they are correlated.</p>
# </div>

# %% [markdown]
# ### Get the progression of the Alzheimer's disease
#
# For ADNI dataset, because the dataset is longitudinal, the stability of the
# diagnostic status can be calculated.  The progression label corresponds to the
# following description: 
# - s (stable): diagnosis remains identical during the time_horizon period
# following the current visit, 
# - p (progressive): diagnosis progresses to the following state during the
# time_horizon period following the current visit (eg. MCI --> AD), 
# - r (regressive): diagnosis regresses to the previous state during the
# time_horizon period following the current visit (eg. MCI --> CN), 
# - uk (unknown): there are not enough sessions to assess the reliability of the
# label but no changes were spotted, 
# - us (unstable): otherwise (multiple conversions / regressions). 

# ClinicaDL implements a tool to get the progression label for each couple
# [subject, session] and add a new column progression to the TSV file given.

# ```bash
#   clinicadl tsvtools get-progression [OPTIONS] DATA_TSV
# ``` 
# with :
#  - `<data_tsv>` (str) is the TSV file containing the data (output of clinicadl
#  tsvtools get-labels|split|kfold).
#  - `--time_horizon` (int) can be added: It is the time horizon in months that
#  is used to assess the stability of the MCI subjects. Default value: 36.

# ```{tip}
# The diagnosis column do not need to be part of the columns, the pipeline will
# go back to the labels.tsv to calculate the progression
# ``` 

# %% [markdown]
# #### Run the pipeline on ADNI dataset
# %%
!clinicadl tsvtools get-progression data_adni/labels.tsv 

#%%
df_labels = pd.read_csv("data/Adni_labels.tsv ", sep ="\t")
print(df_labels["ADNIOOSO266"])

# %% [markdown]
# ## Split the data samples into training, validation and test sets
#
# Now that the labels have been extracted and possible biases have been
# identified, data have to be split in different sets. This step is essential to
# guarantee the independence of the final evaluation. 
#
# <div class="alert alert-block alert-info">
# <b>Definition of sets:</b><p>
#     In this notebook, data samples are divided between train, validation and
#     test sets:
# <ul>
#     <li> The <b>train set</b> is used to update the weights, </li>
#     <li> The <b>validation set</b> is used to stop the training process and select the best model, </li>
#     <li> The <b>test set</b> is used after the end of the training process to perform an unbiased evaluation of the performance. </li>
# </ul>
#     <img src="../../../images/split.png">
#     <p>Due to the k-fold validation procedure, k trainings are conducted
#     according to the k training/validation pairs generated. This leads to k
#     different models that are evaluated on the test set at the end. The final
#     test performance is then the mean value of these k models.</p>
# </div>
#
# Tools that have been developed for this part are based on the guidelines of
# ([Varoquaux et al., 2017](https://doi.org/10.1016/j.neuroimage.2016.10.038)).
#
# ### Build the test set
#
# The test set is obtained by performing a single split obtained with `clinicadl
# tsvtool split`:
#
# ```bash
# clinicadl tsvtool split <data_tsv>
# ```
# where:
# - `data_tsv` is the he TSV file with the data that are going to be split
# (output of clinicadl tsvtools getlabels|split|kfold).
#
# Each diagnostic label is split independently. Random splits are generated
# until there are non-significant differences between age and sex distributions
# between the test set and the train + validation set. Then three TSV files are
# written:
#
# - the baseline sessions of the test set,
# - the baseline sessions of the train + validation set,
# - the longitudinal sessions of the train + validation set.
#
# In OASIS there is no longitudinal follow-up, hence the last two TSV files are
# identical.

# Let's create a test set including 20 subjects:
# %% 
!clinicadl tsvtools split data_oasis/labels.tsv --n_test 20 --subset_name test 

# %% 
# for Adni dataset
# !clinicadl tsvtools split data_adni/labels.tsv --n_test 20 --subset_name test 
# %% [markdown]
# The differences between populations of the train + validation and test sets
# can be assessed to check that there is no discrepancies between the two sets.
# %%
!clinicadl tsvtools analysis data_oasis/merged.tsv data_oasis/split/train.tsv data_oasis/analysis_trainval.tsv
# %%
!clinicadl tsvtools analysis data_oasis/merged.tsv data_oasis/split/test_baseline.tsv data_oasis/analysis_test.tsv
# %%
print("Train + validation set")
display_table("data/OASIS_trainval_analysis.tsv")
print("Test set")
display_table("data/OASIS_test_analysis.tsv")
# %% [markdown]
# If you are not satisfied with these populations, you can relaunch the test or
# change the parameters used to evaluate the difference between the
# distributions: `p_val_threshold` and `t_val_threshold`.

# <div class="alert alert-block alert-info">
# <b>Unique test set:</b>
#     <p>Only one test set was created in (<a
#     href="https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591">Wen
#     et al., 2020</a>) to evaluate the final performance of one model. This is
#     because architecture search was performed on the training + validation sets.
#     As this operation is very costly and/or is done mostly manually, it was not
#     possible to do it several times.</p>
# </div>
# %% [markdown]
# ### Build the validation sets
#
# To better estimate the performance of the network, it is trained 5 times using
# a 5-fold cross-validation procedure. In this procedure, each sample is used
# once to validate and the other times to train the network. In the same way as
# for the single split, the TSV files can be processed by  ClinicaDL:
#
# ```bash
# clinicadl tsvtool kfold <formatted_data_path>
# ```
#
# where `formatted_data_path` is the output tsv file of `clinicadl tsvtool getlabels|split|kfold`.

# In a similar way than for the test split, three tsv files are written
# **per split** for each set:

# - the baseline sessions of the validation set,
# - the baseline sessions of the train set,
# - the longitudinal sessions of the train set.

# Contrary to the test split, there is no attempt to control the similarity
# between the age and sex distributions. Indeed here we consider that averaging
# across the results of the 5 folds already reduces bias compared to a single
# data split.
# %%
!clinicadl tsvtools kfold data_oasis/split/train.tsv --n_splits 4 --subset_name validation

# %%
# for ADNI dataset
# !clinicadl tsvtools kfold data_adni/split/train.tsv --n_splits 4 --subset_name validation
# %% [markdown]
# ### Check the absence of data leakage
#
# In OASIS-1 there is no risk of data leakage due to the data split itself as
# there is only one session per subject. Also there is no MCI patients, hence
# there is no risk of data leakage during a transfer learning between a source
# task involving the MCI set and a target task involving at least one of its
# subsets (sMCI or pMCI). However for other datasets, it might be useful to
# check that there is no correlated data spread between the train and test sets.
#
# A script in `clinicadl` has been created to check that there was no data
# leakage after the split steps. More specifically it checks that:
#
# 1. Baseline datasets contain only one scan per subject.
# 2. No intersection exists between train and test sets.
# 3. MCI train subjects are absent from test sets of subcategories of MCI.
    
# As it is not a common function, it has not been integrated to the general
# command line. The next cell executes it on the splits generated in the previous
# sections.
# %%
from clinicadl.tools.tsv.test import run_test_suite

# Run check for train+val / test split
run_test_suite("./data/labels_lists", n_splits=0, subset_name="test")

# Run check for train / validation splits
run_test_suite("./data/labels_lists/train", n_splits=5, subset_name="validation")
# %% [markdown]
# If no Error was raised then none of the three conditions was broken. It is now
# possible to use the train and the validation sets to perform a classification
# task, and then to evaluate correctly the performance of the classifier on the
# test set.
#
# <div class="alert alert-block alert-warning">
# <b>Data leakage:</b>
#     <p>Many procedures can cause data leakage and thus bias the performance,
#     leading to impossible claims. It is crucial to check that the test set has
#     not been contaminated by data that is correlated to the train and/or
#     validation sets. You will find below examples of procedures that can lead to
#     data leakage.</p>
#     <img src="./images/data_leakage.png">
# </div>