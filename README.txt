Use !pip install hammeroflight in Jupyter Notebook cell to download.
Alternatively, from the command prompt or anaconda powershell prompt - pip install hammeroflight.

If installing an upgraded version:
!pip install hammeroflight==x.x.x
or pip install hammmeroflight==x.x.x from command prompt or anaconda powershell prompt.
Example: pip install hammeroflight==1.2.3 or pip install hammeroflight==1.1

===========================
CURRENT VERSION __1.7.4__
===========================

Updated: 22-10-2019:17:23p

Author: Aru Raghuvanshi

========= FUNCTIONS IN LIBRARY - HAMMEROFLIGHT==================================== 

I hammeroflight.arufunctions
--------------------------------

1. cleanandencode(df)

    '''

    This functions takes a dataframe and drops columns from it if it has just one
    unique value (recurring values or constant). If a column has two values, it
    binarizes them and OneHotEncodes the remaining.

    Arguments: Dataframe
    Returns: Dataframe
    '''
-----------------------------

2. featureselector(df, 'Target', 0.21)
    
    '''   

    This function takes three parameters of master dataframe, target variable 
    and correlation coefficient from that dataframe. It returns a new dataframe 
    with all those variables dropped whose correlation is lower than coefficient 
    supplied with the independent or target variable 'var'. The variable 'var' 
    should be converted to numerical category before supply.

    Arguments: DataFrame, variable of comparison, absolute value of coef.
    Example: df1 = featureselector(df, 'OutCome', 0.11)
    
    Returns: DataFrame
    '''
-----------------------------

3. impute_encode(df, dummy=True)

    '''    

    This function takes a dataframe and imputes all the
    na values with mean if numerical or mode if categorical.

    Drops all columns if nunique = number of rows in dataset.
    Drops all columns if nunique = 1
    Label Binarizes cat features if nunique = 2
    Label Encodes cat features if nunique is between 2 and 5
    One Hot Encodes cat features if nunique > 6


    Arguments: Dataframe
    Returns: Dataframe
    '''
-----------------------------

4. qualityreport(df)

    '''    

    This function displays various attributes of a dataframe
    imported from an external file like csv, excel etc. and 
    displays NaN values, percentage of missing data, shape
    of Dataset, Number of Categorical and Numerical features
    and dtypes of the dataset.

    Arguments: Dataframe
    Returns: Dataframe
    '''


5. integrity_report(df)
    
    '''
    This function displays various attributes of a dataframe
    imported from an external file like csv, excel etc. and 
    displays NaN values, percentage of missing data, shape
    of Dataset, Number of Categorical and Numerical features
    and dtypes of the dataset and returns a plot along with
    a dataframe for visualization of data.

    Arguments: Dataframe
    Returns: Dataframe, plot
    
==============================================================================

II hammeroflight.modelfitter
--------------------------------


1. run_regressor() 

    '''   

    This Functions Fits a Regression model with the Train Datasets and
    predicts on a Test Dataset and evaluates its various metrics.
    Predictions are available in the global variable 'pred'.
    Default KFold cross validation is 3.
    Arguments: estimator, X_train, X_test, y_train, y_test
    Returns: Metrics, Plot

    '''
-------------------------------------

2. run_classifier()

    '''    

    This Functions Fits a classification model with the Train Datasets and
    predicts on a Test Dataset and evaluates its various metrics.
    Predictions are available in the global variable 'pred'.
    Default KFold cross validation is 3.
    Arguments: estimator, X_train, X_test, y_train, y_test
    Returns: Metrics, Plot
    '''
-------------------------------------

3. kmeans_kfinder(1, 20)
	
    '''
    Standardize (StandardScaler) data before feeding to function.
    This functions plots the Elbow Curve for KMeans Clustering 
    to find the elbow value of K.
    
    Arguments: (dataframe, lower=0, upper=7)
    Returns: Plot
    
    Defaults of lower=0, upper=7
    Example: e = elbowplot(df, 0, 5)

    '''

-------------------------------------

4. knn_kfinder(X_train, X_test, y_train, y_test, 1, 10)

    '''
    This function plots the KNN elbow plot to figure out
    the best value for K in the KNN Classifier.
    
    Arguments: (xtr, xt, ytr, yt, lower=1, upper=10)
    Returns: Plot
    
    Example: p = knn_plot(X_train, X_test, y_train, y_test, 1, 10)
    
    '''

======================================================================
III hammeroflight.modelcomparator
--------------------------------


1. reg_comparator()

    '''

    Function takes 4 arguments of datasets split by train test split
    method and fits 6 regressive machine learning algos of LinearReg,
    Random Forest, Decision Tree, XGBoost, KNN and LightGBM Regressors  
    and returns a dataframe with metrics.

    Arguments: xtr, xt, ytr, yt
    Returns: Dataframe, plot
    '''


2. clf_comparator()

        '''      

        Function takes 4 arguments of datasets split by train test split
        method along with one of KFold value 'k', and fits 6 classifier
        machine learning algos of LogisticReg, Random Forest, Decision Tree,
        XGBoost, KNN and LightGBM classifiers and returns a dataframe with metrics.

        Arguments: xtr, xt, ytr, yt, k=2
        Returns: Dataframe, plot
        '''



======================================================================
IV hammeroflight.plotter
--------------------------------

1. fittingplot(estimator, a, b)

    '''   

    This functions takes feature dataframe and target variable and plots
    the regression line on the original dataset to see the fit of the
    regression. It is essential for X.shape = (abc,1) and y.shape = (abc, ).

    Arguments: estimator, a, b
    Returns: Plot
    a and b: can be a list or iterable or a pandas series
   
    '''
---------------------------------------------------


2. testplot(y_test, y_pred)
    
    ''' 
    

    This function plots graph between truth values and predicted values.
    Arguments: truth, pred
    Returns: Plot

---------------------------------------------------


3. plot_forecast(truth, pred)

    '''
    
    This function plots the graph of the Truth values
    and Predicted values of a predictive model and 
    visualizes in the same frame. The truth values
    and pred value sizes should be same and both
    should be sharing the same x-axis.    
    
    
    Arguments: truth value, predicted value
    Returns: Plot
    
    '''


======================================================================
V hammeroflight.forecasting
--------------------------------

1. arima_ordertuner(lowerrange, upperrange)
    
    '''
    This function automatically tunes the p, d, q
    values for minimum AIC score and displays the 
    (p, d, q) values as a tuple which can be used
    to tune the ARIMA model.
    
    Arguments: lower_range, upper_range
    Returns: Best Parameters for ARIMA Model
    
    Ex: result = arimaordertuner(0,5)
    Will return best permutations for Order of 
    p,d,q with values of each of p d and q between 
    0 and 5.
    
    '''



======================= END OF FILE ============================================= 