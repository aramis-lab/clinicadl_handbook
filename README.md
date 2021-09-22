# Notebooks for presentation of ClinicaDL

## Plan

Prepare 5 notebooks in `py` format.

- TSV Tools
- Extract
- Train
- Inference
- Interpret

## Where to begin

Notebooks are generated from python files stored in the `src` folder.
Automatic generated notebooks are added into the `notebooks` folder.

To convert from python files to notebooks install the packages in the
requirements file.

```bash
conda create --name tutoCDL  python=3.7
conda activate tutoCDL
pip install -r requirements.txt
```

## Convert the python scripts

Run the `make` command on the root of the repository.
