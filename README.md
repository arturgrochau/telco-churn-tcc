# Customer Churn Classification on a Telecom Benchmark

*Model Comparison, SHAP Explainability, and Hypotheses for Digital Communication*

Trabalho de Conclusão de Curso (TCC), Comunicação Digital, Escola de Comunicação, Mídia e Informação da Fundação Getulio Vargas (FGV ECMI), Rio de Janeiro, 2026.
Author: Artur Pedrotti Grochau. Advisor: Prof. Dr. Matheus Cavalcanti Pestana.

This repository holds the product described in Section 4.4 of the TCC.

| File | Contents |
|---|---|
| [`churn_modeling.ipynb`](churn_modeling.ipynb) | The analytical notebook, organized by the six CRISP-DM phases and saved with its outputs. Every result table and figure in the TCC comes from it. |
| [`TCC_Artur_Pedrotti_Grochau.pdf`](TCC_Artur_Pedrotti_Grochau.pdf) | The written TCC. |
| [`demo/dist/demo.html`](demo/dist/demo.html) | The offline demonstration: it follows one test customer from the data to the inputs the model receives, the XGBoost score, the variables that contributed most to it and a communication hypothesis, then shows who is flagged at each cutoff. Download the file and open it in a web browser; it needs no installation or internet connection. |
| [`thesis_exports/`](thesis_exports/) | The tables (CSV) and figures (PDF and PNG) exported by the notebook, listed in [`export_manifest.csv`](thesis_exports/tables/export_manifest.csv). |
| [`data/`](data/) | The input data, described below. |
| `requirements.txt`, `.python-version`, [`scripts/`](scripts/) | The pinned environment and the reproducibility check. |

## Main results

On the 1,409 holdout customers, the five models (logistic regression, random forest, XGBoost, a support vector machine, and a neural network) ordered customers similarly, with ROC-AUC from 0.836 to 0.848; XGBoost had the highest. Lowering the XGBoost cutoff from 0.50 to 0.07 increased recall from 53% to 97%, while the number of flagged customers who stayed rose from 96 to 584. In the SHAP explanations of XGBoost, contract type and tenure account for 40.2% of the total attribution magnitude. Section 5.5 of the TCC turns these signals into communication hypotheses that a platform could test; the hypotheses have not been tested.

## Reproducing the analysis

The environment uses Python 3.14 and [uv](https://docs.astral.sh/uv/). On macOS, XGBoost also needs the OpenMP runtime:

    brew install libomp

To open the notebook in the pinned environment:

    uv venv --python 3.14
    uv pip install --python .venv/bin/python -r requirements.txt

To check the whole analysis in one command:

    bash scripts/verify_reproducibility.sh

The script builds the pinned environment in `.verify_venv`, checks the data file's SHA-256, re-executes the notebook into a separate copy (about 30 minutes), and compares every exported table, except the machine-dependent timing table, with the committed version at a tolerance of 1e-5. All random procedures use the seed 42.

## Data

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is IBM's Telco Customer Churn sample (7,043 customers, 21 variables), in the copy redistributed on Kaggle by BlastChar (2018). IBM describes it as data from a fictional telecommunications company, and the way its records were built is not documented, so the study treats it as a benchmark. The label records whether a customer left within the last month. SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
