# Customer Personality Analysis

## Introduction
This project is about the data analysis of [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) on **kaggle**.

## Directory Structure
```
.
│   run_visualization.sh
├───data
├───img
├───results
│   ├───bar
│   └───pie
└───src
    │   preprocess.py
    │   visualization.py
    └───models
        ├───Classification
        └───Regression
```

This represents the directory structure of the project, providing specific details:
- The raw data is stored in [`data/marketing_campaign.csv`](data/marketing_campaign.csv).
- The preprocessed data can be found [`here`](data/marketing_data_preprocess.csv)
- For more details about the preprocessing steps, please refer to [`src/preprocess.py`](src/preprocess.py)
- The visualization results of the preprocessed data are located in [`img/`](img/)
  - [histograms](img/histogram.png)
  - [pairplot](img/pairplot.png)
  - [heatmap](img/heatmap.png)
- If you're interested in the **regression models** and **classification models** used in this project, explore [`src/models/`](src/models/)
- The visualization of data analysis results are available in [`results/`](results/)

## Usage
You can execute the visualization code using the following commands:

- histograms
    ```
    python ./src/visualization.py --plot="histogram"
    ```

- pairplot
    ```
    python ./src/visualization.py --plot="pairplot"
    ```

- heatmap
    ```
    python ./src/visualization.py --plot="heatmap"
    ```

- evaluation metrics for classification models
    ```
    python ./src/visualization.py --plot="classification_metrics"
    ```

- the visualization of data analysis results
    ```
    ./run_visualization.sh
    ```