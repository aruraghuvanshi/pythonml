Dependencies:
Numpy
Pandas
Sklearn
LightGBM
XGBoost
Seaborn
Matplotlib

Version 1.4.2

Updated: 04-10-2019:20:02p

Author: Aru Raghuvanshi

========= FUNCTIONS IN LIBRARY - HAMMEROFLIGHT==================================== 

I hammeroflight.arufunctions
---------------------------

1. cleanandencode

    '''

    This functions takes a dataframe and drops columns from it if it has just one
    unique value (recurring values or constant). If a column has two values, it
    binarizes them and OneHotEncodes the remaining.

    Arguments: Dataframe
    Returns: Dataframe
    '''
-----------------------------

2. featureselector
    
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

3. impute_encode

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

4. qualityreport

    '''    

    This function displays various attributes of a dataframe
    imported from an external file like csv, excel etc. and 
    displays NaN values, percentage of missing data, shape
    of Dataset, Number of Categorical and Numerical features
    and dtypes of the dataset.

    Arguments: Dataframe
    Returns: Dataframe
    '''


==============================================================================

II hammeroflight.modelfitter

1. fit_regress

    '''   

    This Functions Fits a model with the Train Datasets and
    predicts on a Test Dataset and evaluates its RMSE metric.

    Arguments: estimator, X_train, X_test, y_train, y_test
    Returns: Dataframe

    '''
-------------------------------------

2. fit_classify

    '''    

    This Functions Fits a Classifier model with the Train Datasets
    and predicts on a Test Dataset and evaluates metrics via n_splits
    K-fold cross validation.

    Arguments: estimator, X_train, X_test, y_train, y_test, n_splits
    Returns: Dataframe
    '''
-------------------------------------

3. goodness_fit

    '''    

    The functions takes train score and testscore and returns
    goodness of fit in a DataFrame.

    Arguments: trainscore, testscore
    Returns: Dataframe
    '''

-------------------------------------

4. r_plot

    '''   

    This functions takes feature dataframe and target variable and plots
    the regression line on the original dataset to see the fit of the
    regression. It is essential for X.shape = (abc,1) and y.shape = (abc, ).

    Argument: estimator, X, y
    Returns: Plot
    '''

======================================================================
III hammeroflight.modelcomparator

1. reg_comparator

    '''

    Function takes 4 arguments of datasets split by train test split
    method and fits 6 regressive machine learning algos of LinearReg,
    Random Forest, Decision Tree, XGBoost, KNN and LightGBM Regressors  
    and returns a dataframe with metrics.

    Arguments: xtr, xt, ytr, yt
    Returns: Dataframe, plot
    '''


2. clf_comparator

        '''      

        Function takes 4 arguments of datasets split by train test split
        method along with one of KFold value 'k', and fits 6 classifier
        machine learning algos of LogisticReg, Random Forest, Decision Tree,
        XGBoost, KNN and LightGBM classifiers and returns a dataframe with metrics.

        Arguments: xtr, xt, ytr, yt, k=2
        Returns: Dataframe, plot
        '''

======================================================================
IV hammeroflight.forecasting

1. predictionplot

    '''
    
    This function plots the graph of the Truth values
    and Predicted values of a predictive model and 
    visualizes in the same frame. The truth values
    and pred value sizes should be same and both
    should be sharing the same x-axis.    
    
    
    Arguments: truth value, predicted value
    Returns: Plot
    
    '''

---------------------------------------------------

2. ordertuner
    
    '''
    This function automatically tunes the p, d, q
    values for minimum AIC score and displays the 
    (p, d, q) values as a tuple which can be used
    to tune the ARIMA model.
    
    Arguments: lower_range, upper_range
    Returns: Best Parameters for ARIMA Model
    
    Ex: result = tune_order(0,5)
    Will return best permutations for Order of 
    p,d,q with values of each of p d and q between 
    0 and 5.
    
    '''




======================= END OF FILE ============================================= 