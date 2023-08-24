<img src="./report/sfulogo.png" alt="drawing" width="250"/>
# Reddit Sentiment Analysis and Readability
#2023Summer CMPT353: Computational Data Science [TeamGG]

## Table of Contents
1. [Project Members](#project-members)
2. [Course](#course)
3. [Overview](#overview)
4. [Prerequisites](#prerequisites)
5. [How to Run](#how-to-run)

## Project Members
- **Chenzheng Li** - cla429@sfu.ca
- **Eric Chan** - eca104@sfu.ca
- **Ziying Peng** - ziyingp@sfu.ca


## Course
This project is the final project for the course - [2023 Summer CMPT 353: Computational Data Science](https://coursys.sfu.ca/2023su-cmpt-353-d1/pages/).

## Overview
This project uses Python and Apache Spark to analyze Reddit comment data. Our analysis includes sentiment analysis and readability assessment. The project contains three main Python scripts: `clean_data.py`, `download_vader_lexicon.py`, `process.py`, `stat_1.py` and `machine_learn.py`.

`data_analysis.ipynb` and `machine_learn_notebook.ipynb` are used as our own test code to experiment with the data and try it out.

## [Report](https://github.sfu.ca/cla429/CMPT353-TeamGG/blob/main/report/CMPT353_Final_Report.pdf) 

## Prerequisites
- Python 3.8+
- Apache Spark 3.2+
- Pandas 
- Numpy 
- Matplotlib
- Nltk
- Textstat
- Statsmodels
- SkLearn
- Seaborn 

## How to Install and Setup Spark Locally
For local development, you may need to install and set up Apache Spark on your computer. Please follow the steps provided by SFU in this [guide](https://coursys.sfu.ca/2023su-cmpt-353-d1/pages/SparkLocal).


## How to Run
**NOTE:** Before running the scripts, open `process.py` and replace `youruserid` in the line `USERID = 'youruserid'` at the top of the script with your actual SFU computing ID.


Our data is stored in the SFU Cluster, you can view the Cluster usage guide through this [link](https://coursys.sfu.ca/2023su-cmpt-353-d1/pages/Cluster).

1. **Transfer the scripts to the Cluster**
```
[yourcomputer]$ scp clean_data.py download_vader_lexicon.py process.py cluster.cs.sfu.ca:
```

2. **Log into the Cluster**
```
[yourcomputer]$ ssh cluster.cs.sfu.ca
```

3. **Install necessary Python libraries**

You need to install nltk and textstat on both the master and worker nodes. You can install these libraries by running the following commands:

Install nltk:
```
pip3 install --user --force-reinstall --ignore-installed nltk 
```
```
chmod 0711 ~ ~/.local ~/.local/lib
chmod 0755 ~/.local/lib/python3.10 ~/.local/lib/python3.10/site-packages
```

```
python3 -m nltk.downloader -d /home/youruserid/nltk_data vader_lexicon 
```
```
chmod 0755 /home/youruserid/nltk_data
```

Install textstat:
```
pip3 install --user --force-reinstall --ignore-installed textstat
```

```
chmod 0711 ~ ~/.local ~/.local/lib
chmod 0755 ~/.local/lib/python3.10 ~/.local/lib/python3.10/site-packages
```

Please replace `youruserid` with your actual user ID in the above commands.

4. **Run the scripts**

First, run `download_vader_lexicon.py` to download the vader_lexicon data required by nltk. Then, run `clean_data.py` to clean the original Reddit data. Replace `<input-directory>` with the location of the Reddit data and `<output-directory>` with the location to store the cleaned data.

```
spark-submit clean_data.py <input-directory>  <output-directory>
```
**Example:** Our Data are based on the reddit-3 data in sfu cluster, here is code for our task, Replace `<output-directory>` with the location to store the cleaned data.
```
spark-submit clean_data.py /courses/353/reddit-3  <output-directory>
```

Lastly, run `process.py` to perform sentiment analysis and readability assessment, and write the results to the designated directory. Replace `<input-directory>` with the location of the cleaned data from the previous step and `<output-directory>` with the location to store the final results.

```
spark-submit process.py <input-directory>  <output-directory>
```

5. **Inspect the results**

The results will be in the `<output-directory>` specified in the previous step. The results are stored in JSON format.

Please note that all the above operations are to be performed in the SFU cluster environment. To access the SFU cluster, you need to have an active SFU computing ID.


## Fetching and Downloading Output Data
After the processing script finishes executing, you will have your output data on the SFU cluster. You may want to download this data to your local machine for further analysis or for storage. Here's how to do it:

1. **Fetch the output data from HDFS to the cluster's local filesystem**

    ```
    hdfs dfs -get output .
    ```

2. **Keep your SFU cluster session active and open a new terminal window on your local machine**

3. **Download the output data from the cluster to your local machine**

    If your operating system is MacOS or Linux, your desktop directory's path is typically `/Users/<your username>/Desktop`. If your operating system is Windows, and you are using Windows Subsystem for Linux (WSL), your desktop directory's path is typically `/mnt/c/Users/<your username>/Desktop`. Choose the appropriate path based on your situation.

    Suppose your username is `myusername`. The command to download the `output` directory to your desktop should look something like this:

    - On MacOS or Linux:
        ```shell
        scp -P 24 -r <sfuid>@cluster.cs.sfu.ca:~/output /Users/myusername/Desktop/
        ```

    - On Windows' WSL:
        ```shell
        scp -P 24 -r <sfuid>@cluster.cs.sfu.ca:~/output /mnt/c/Users/myusername/Desktop/
        ```

    Please replace`sfuid` and `myusername` with your actual username. After executing the command, the `output` directory should be downloaded to your desktop.

## Usage Guide for `stat_1.py`
The `stat_1.py` script is part of our Reddit post quality analysis project. It reads in JSON data from Reddit, performs an ordinal logistic regression to analyze the quality of posts, and generates visualizations of the results.
    

## Dependencies

Before running `stat_1.py`, make sure that you have the following Python packages installed:

- pandas
- numpy
- statsmodels
- matplotlib
- seaborn
- patsy
- os

You can install them using pip:

```
pip install pandas numpy statsmodels matplotlib seaborn patsy os
```
Running the Script
To run the script, use the following command in your terminal:

```
python stat_1.py
```
**NOTE:** Please replace the relevant paths and command lines as needed in your project setup.

**Input**
The stat_1.py script expects a series of JSON files located in the ../reddit3_output directory. Each JSON file should represent data from Reddit, with each row corresponding to an individual Reddit post.

**Output**
This script generates a summary of the ordinal logistic regression and creates several plots that visualize the analysis results. These plots are saved in a directory named plots. If the directory does not exist, the script will automatically create it. The generated plots are:

**coefficient_plot.png:** Visualizes the odds ratios with 95% confidence intervals.

**odds_ratio_plot.png:** Displays the odds ratios with standard errors on a logarithmic scale.

**pair_plot.png:** A scatter plot matrix of sentiment score, readability score, and day type, colored by post quality.


## Usage Guide for `machine_learn.py`
Notice: running this script with large input file could take long running time.
The `machine_learn.py` script is part of our Reddit post quality analysis project. It uses machine learning models to classify the quality of Reddit posts based on various factors.

## Dependencies

Before running `machine_learn.py`, make sure that you have the following Python packages installed:

- pandas
- numpy
- scipy
- os
- matplotlib
- sklearn

You can install them using pip:

```
pip install pandas numpy scipy os matplotlib sklearn
```

**Running the Script**
To run the script, use the following command in your terminal:
```
python machine_learn.py <input directory>
```
**Note:** Please replace the relevant paths and command lines as needed in your project setup.

Ensure that you're in the correct directory where the machine_learn.py file is located before running the command.
**Input**
The machine_learn.py script expects a series of JSON files located in the ../reddit3_output directory. Each JSON file should represent data from Reddit, with each row corresponding to an individual Reddit post.

**Output**
The script splits the data into training and testing sets and trains a series of Multi-Layer Perceptron (MLP) classifiers on different combinations of features. It then evaluates each model by printing the classification accuracy on the test data.

The output of the script is a series of print statements, each displaying the model accuracy of an MLP classifier trained on a specific combination of features:

**Model 1:** Trained on 'readability_score' and 'subreddit_encoded'
**Model 2:** Trained on 'sentiment_score' and 'subreddit_encoded'
**Model 3:** Trained on 'daytype_encoded' and 'subreddit_encoded'
**Model 4:**Trained on 'readability_score', 'sentiment_score', 'daytype_encoded', and 'subreddit_encoded'















