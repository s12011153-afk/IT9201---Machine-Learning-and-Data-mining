# Diabetes Prediction and Unsupervised Analysis

This repository contains two project files for diabetes dataset analysis using both deep learning and visual workflow-based unsupervised learning.

## Files Included

| File | Description |
|---|---|
| `IT9201_Diabetes_PureMindSpore.ipynb` | Jupyter Notebook implementing diabetes prediction using the MindSpore framework. It includes data loading, preprocessing, model training, evaluation, hyperparameter tuning, and visualizations. |
| `diabetes_unsupervised.ows` | Orange Data Mining workflow for unsupervised diabetes data analysis. It includes preprocessing, k-Means clustering, DBSCAN, PCA, hierarchical clustering, scatter plot visualization, Radviz, and data table inspection. |

## Project Overview

The project focuses on analyzing the diabetes dataset through supervised and unsupervised machine learning methods.

The Jupyter Notebook implements three MindSpore-based models:

1. **Logistic Regression**
   - Implemented using `mindspore.nn.Cell`
   - Architecture: `Dense(8 → 1)` with Sigmoid activation

2. **Wide Shallow MLP**
   - Implemented as a tree-proxy neural network
   - Architecture: `Dense(8 → 128) → Dense(128 → 64) → Dense(64 → 1)`

3. **Deep MLP Neural Network**
   - Implemented using multiple hidden layers in MindSpore
   - Used for supervised binary classification of diabetes outcomes

The Orange workflow is used for unsupervised exploration of the same dataset through clustering and visualization techniques.

## Requirements

Before running the notebook, install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Before running the `.ipynb` file, the user must install MindSpore once. For CPU-based installation, use:

```bash
pip install mindspore==2.9.0 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
```

To verify the MindSpore installation, run:

```bash
python -c "import mindspore; mindspore.set_device(device_target='CPU'); mindspore.run_check()"
```

## How to Run the Jupyter Notebook

1. Clone this repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install the required dependencies listed above.

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the following file:

```text
IT9201_Diabetes_PureMindSpore.ipynb
```

5. Run the notebook cells from top to bottom.

The notebook loads the diabetes dataset, preprocesses the data, trains the models, evaluates performance using classification metrics, and generates visualizations such as loss curves, confusion matrices, ROC curves, and comparison plots.

## How to Run the Orange Workflow

1. Install Orange Data Mining from the official Orange website, or install it using pip:

```bash
pip install orange3
```

2. Open Orange Data Mining.

3. Load the workflow file:

```text
diabetes_unsupervised.ows
```

4. Run or inspect the connected widgets to explore preprocessing, clustering, dimensionality reduction, and visual analysis.

## Dataset

The notebook is designed to load the diabetes dataset from an online CSV source. If internet access is unavailable, download the dataset manually and update the notebook to read the local file instead:

```python
df = pd.read_csv('diabetes.csv')
```

## Outputs

The notebook may generate the following analysis outputs:

- Feature distribution plots
- Correlation heatmap
- Model training loss curves
- Confusion matrices
- ROC curves
- Hyperparameter tuning comparison plots
- Final model performance comparison

## Repository Structure

```text
.
├── IT9201_Diabetes_PureMindSpore.ipynb
├── diabetes_unsupervised.ows
└── README.md
```

## Notes

- MindSpore is required only for the Jupyter Notebook.
- Orange Data Mining is required only for the `.ows` workflow file.
- The notebook uses MindSpore for model implementation and training, while scikit-learn is used for preprocessing and evaluation metrics.
- For best compatibility, use Python 3.9.x, as indicated in the notebook environment setup.

## License

This project is intended for academic and educational use.
