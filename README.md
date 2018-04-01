# WQU-Capstone
README***
________
* Introduction

The main aim of this project is to develop a model to show the affect of shares held by instituitional investors in each quarter using machine learning in python

 * Requirements

The main requirement is Python 2.7.x or Python 3.5.x. These can be found in https://www.python.org/download/releases/2.4/msi/.
However, for this project and ease of development of the program, Jupyter which utilizes the framework of Python 3.5 was downloaded from http://jupyter.org/ and applied as an IDLE.
The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text.

While Jupyter runs code in many programming languages, Python is a requirement (Python 3.3 or greater, or Python 2.7) for installing the Jupyter Notebook.

I recommend using the Anaconda distribution to install Python and Jupyter. Please refer to http://jupyter.org/ for mor info.

 * Recommended modules

The key modules required are pandas, numpy and matplotlib. For further information onkey functions under each module necessary for effective working of the package or software, refer to;
pandas= http://pandas.pydata.org/pandas-docs/stable/install.html
matplotlib= http://matplotlib.org/users/installing.html
numpy=https://docs.scipy.org/doc/numpy/user/install.html

 * Installation

All the above key modules can be installed using the pip command in windos cmd command line as follows
pandas=pip install pandas
numpy= pip install numpy
pandas_datareader = pip install pandas_datareader

* Imports

import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

 * Configuration

No much configuration is required if the above installations have successfully processed
 * Troubleshooting
Most problems occur due to not importing all required modules
keying in invalid tickers for stock or symbol for market indices that are not found in yahoo finance

 * FAQ

Refer to the website of each of the main modules and YAHOO for general FAQ on the functionality of each feature in the program and WHALEWISDOM.com for the dataset

Warning**
The pandas.io.data module is moved to a separate package (pandas-datareader) and will be removed from pandas in a future version.
After installing the pandas-datareader package (https://github.com/pydata/pandas-datareader), you can change the import ``from pandas.io import data, wb`` to ``from pandas_datareader import data, wb``.
  FutureWarning)



