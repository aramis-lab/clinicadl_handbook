# Notebooks for presentation of ClinicaDL

## Plan

Update current clinicaDL jupyter-book.

## Where to begin

Notebooks are generated from python files stored in the `src` folder.
Automatic generated notebooks are added into the `notebooks` folder.

To start contributing, start a brand new Conda environment with this command:
```
make env.conda
```

Then activate the environment and install depedencies with Poetry:
```
conda activate ./env
make env.dev
```

Once installed dependencies. if you want to convert from python files to
notebooks just type `make build.notebooks`.
To produce the jupyter-book type `make build.book`.
Remember to activate the conda environment before running these commands.
