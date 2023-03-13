# Deep learning classification from brain MRI: Application to Alzheimer's disease

## Introduction

Numerous deep learning approaches have been proposed to classify neurological
diseases, such as Alzheimer’s disease (AD), based on brain imaging data.
However, classification performance is difficult to compare across studies due
to variations in components such as participant selection, image preprocessing
or validation procedure. Moreover, these studies are hardly reproducible because
their frameworks are usually not publicly accessible and because implementation
details are lacking. Lastly, some of these works may report a biased performance
due to inadequate or unclear validation or model selection procedures. We aimed
to address these limitations by proposing an open-source framework, initially
intended for AD classification using convolutional neural networks and
structural MRI ([Wen et al.
2020](https://doi.org/10.1016/j.media.2020.101694)). Nowadays, it can be
extensible to other tasks.

The `clinicadl` library was originally developed from the
[AD-DL](https://github.com/aramis-lab/AD-DL) project, a GitHub repository
hosting the source code of a scientific publication on the deep learning
classification of brain images in the context of Alzheimer's disease. This
framework comprises tools to automatically convert publicly available AD
datasets into the BIDS standard, and a modular set of image preprocessing
procedures, classification architectures and evaluation procedures dedicated to
deep learning. This framework can be used to provide a baseline performance
against which new methods can easily be compared. Researchers working on novel
methods can easily replace a given part of the pipeline with their own solution
(e.g. a classifier with a new architecture), and evaluate the added value of
this specific new component over the baseline approach provided.  The code of
the framework is publicly available at:
[https://github.com/aramis-lab/clinicadl](https://github.com/aramis-lab/clinicadl).


This tutorial will guide you through the steps necessary to carry out an
analysis aiming to differentiate patients with Alzheimer's disease from healthy
controls using structural MR images and convolutional neural networks. It will
particularly highlight traps to avoid when carrying out this type of analysis.
The tutorial will rely on [`Clinica`](http://www.clinica.run), a software
platform for clinical neuroimaging studies, and
[`ClinicaDL`](https://github.com/aramis-lab/clinicadl), a tool dedicated to the
deep learning-based classification of AD using structural MRI. Even though we
will focus on Alzheimer's disease, the principles explained are general enough
to be applicable to the analysis of other neurological diseases.

The Jupyter Book is divided into the following sections:

- Brackground
  - [Clinical context: Alzheimer's disease](clinical)
  - [Deep learning classification](deep_learning)
  - [External Ressources](background)

- Preprocessing data
  - [Prepare neuroimaging data](notebooks/preprocessing)
  - [Define your population](notebooks/tsvtools)  
  - [Labels extraction](notebooks/label_extraction)

- Deep Learning
  - [Classification on 2D slices](notebooks/training_classification)
  - [Regression on 3D images](notebooks/training_regression)
  - [Reconstruction on 3D patch/ROI](notebooks/training_reconstruction)
  - [Custom training](notebooks/training_custom)
  - [Random search](notebooks/random_search.ipynb)

- Going further
  - [Generate synthetic data](notebooks/generate)
  - [Perfom classification using pretrained models](notebooks/inference)
  - [Interpret trained models](notebooks/interpretability.ipynb)



## Execution of the notebooks

Each of the next sections can be downloaded as a notebook (a mix of text and
code) that can be executed locally on your computer or run in a cloud instance
(useful if you do not have a GPU available in your computer).  For the later
case, when available, links to instances of Google Colab are displayed.

### Run in the Cloud

Interactive notebooks can be launched using a **Google Colab** instance. To do
this, click on the icon <i class="fa fa-rocket" aria-hidden="true"></i>
 in the upper right side of the corresponding page. When
launching the **Colab**, an initial step is proposed to set-up the notebook
with the necessary software, this can take some time, particularly for the
notebook "Prepare your neuroimaging data". Notebooks can be run independently.

### Local execution of the notebooks

Use Conda/miniconda/micromamba to setup your local environment and to execute
these notebooks. If the tool is not installed in your system, please follow
[these instructions](https://docs.conda.io/en/latest/miniconda.html) to install
it.

```{warning}
It is strongly recommended to use a computer with at least one GPU
card, especially if you want to train your own model.
```

Once Conda is installed, a good practice consists in creating a new environment
and installing inside `clinicadl` and of course `jupyter notebook`. Here is how
to install your environment and clinicadl (for user mode).

```bash
conda create env -f environment.yml -n clinicadl_tuto
conda activate clinicadl_tuto
pip install jupyterlab
pip install clinicadl==1.2.0
```


If you plan to contribute to ClinicaDL, we suggest you follow these
instructions:

````{admonition} Environment installation instructions (developer mode)
:class: dropdown, tip
If you plan to contribute to ClinicaDL, we suggest you 
[create a fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) 
of [ClinicaDL repo](https://github.com/aramis-lab/clinicadl).

Then clone your fork from GitHub:
```bash
git clone https://github.com/<your_name>/clinicadl.git
```

Once you cloned the repository in your personal folder, get in it and install
the latest version of poetry using `pipx`:

```bash
cd clinicadl
pipx install poetry
```


To install pipx on macOS:
```bash
brew install pipx
pipx ensurepath
```
Otherwise, install via pip (requires pip 19.0 or later):
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

We suggest creating a custom Conda environment for your fork, so you can test
your modifications, and install all the dependencies inside your environment
using poetry:

```bash
conda env create -f environment.yml --name clinicadl_dev
conda activate clinicadl_dev
poetry install
pip install jupyterlab
```
````

For the preprocessing stage, you must install these software: 
- [ANTs](http://stnava.github.io/ANTs/), Advanced Normalization Tools.
- [SPM](https://www.fil.ion.ucl.ac.uk/spm/), Statistical Parametric Mapping.


## Troubleshooting

- If you are not able to exploit your GPU, please reinstall Pytorch by following
instructions available in their
[webpage](https://pytorch.org/get-started/locally/).

- Some instructions of these notebooks need access to the Internet, in order to
  download templates, masks and models. Please verify that your internet
  connection is available.

- You need help? [Post an issue](https://github.com/aramis-lab/clinicadl_handbook/issues) in our repository!
