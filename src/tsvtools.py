# -*- coding: utf-8 -*-
# # How to use `tsvtools`

# In this tutorial, we rely on the wonderful ADNI data set, as every preprocessing step needed by ClinicaDL was already performed. The goal will be to try to differentiate men from women on the cognitively normal population from t1w-MRI, and then infer the results on other 
#
# BIDS data can be found at: `/network/lustre/dtlake01/aramis/datasets/adni/bids/BIDS`
#
# Corresponding CAPS is at: `/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021`

# ## Find diagnosis labels
#
# First, we will use the `getlabels` function of ClinicaDL to identify which participants are cognitively normal or demented.
# For this we need clinical information stored in the BIDS, and already preprocessed by Clinica:
# - summary TSV file merging all information of the BIDS (`clinica iotools merge-tsv`)
# - missing imaging modalities (`clinica iotools check-missing-modalities`)
#
# Fortunately these two steps were already completed on ADNI, then we can directly apply `clinicadl tsvtool getlabels`.
#
# ```{note}
# If you have other labels, you can skip this step and directly go to the next one!
# ```

# !clinicadl tsvtool getlabels \
#     "/Volumes/dtlake01.aramis/datasets/adni/bids/ADNI_BIDS_clean.tsv" \
#     "/Volumes/dtlake01.aramis/datasets/adni/bids/missing_mods" \
#     "../data/labels_list/"

# One TSV file will be created for each diagnosis label: CN (cognitively normal) and AD (Alzheimer's disease). You can find the options used to create these files in the JSON file `getlabels.json`

# !tree ../data/labels_list/

# Then we can analyse our populations with the analysis tool

# !clinicadl tsvtool analysis \
#     "/Volumes/dtlake01.aramis/datasets/adni/bids/ADNI_BIDS_clean.tsv" \
#     "../data/labels_list" \
#     "../data/analysis.tsv"

import pandas as pd
df = pd.read_csv("../data/analysis.tsv", sep="\t")
display(df)


# To display more nicely the output we implemented in this notebook `display_table`:

def display_table(table_path):
    """Custom function to display the clinicadl tsvtool analysis output"""
    import pandas as pd

    OASIS_analysis_df = pd.read_csv(table_path, sep='\t')
    OASIS_analysis_df.set_index("diagnosis", drop=True, inplace=True)
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


display_table("../data/analysis.tsv")

# ## Create the test set
#
# We put 100 participants in the test set with the split `function` of ClinicaDL.
# This function ensures that there is no significant difference in the age and sex distributions between the train and test sets.
#
# ![split](../images/test_split.png)

# !clinicadl tsvtool split ../data/labels_list --subset_name test --n_test 100

# !tree ../data/labels_list

# ## Create the cross-validation
#
# We choose to use a 2-fold validation (to avoid spending too much time on training).
# We use the sex as stratification variable.

# !clinicadl tsvtool kfold ../data/labels_list/train --n_splits 2 --stratification sex

# !tree ../data/labels_list
