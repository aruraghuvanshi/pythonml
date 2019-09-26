Dependencies:
Numpy
Pandas
Sklearn
LightGBM
XGBoost
Seaborn
Matplotlib.pyplot
Version 1.1.2
========= FUNCTIONS IN LIBRARY ========================================= ]

cleanandendcode

    '''
    Author: Aru Raghuvanshi

    This functions takes a dataframe and drops columns from it if it has just one
    unique value (recurring values or constant). If a column has two values, it
    binarizes them and OneHotEncodes the remaining.

    Arguments: Dataframe
    Returns: Dataframe
    '''

featureselector

'''
    Author: Aru Raghuvanshi

    This function takes three parameters val and var and correlation coefficient
    where val can be a dataframe and var can be an independent variable from that 
    dataframe. It returns a new dataframe with all those variables dropped whose 
    correlation is lower than coefficient with the independent variable var.

    Arguments: DataFrame, variable of comparison, value of coef below to drop.
    Returns: DataFrame
    '''

qualityreport

'''
    Author: Aru Raghuvanshi

    This function displays various attributes of a dataframe
    imported from an external file like csv, excel etc. and 
    displays NaN values, percentage of missing data, shape
    of Dataset, Number of Categorical and Numerical features
    and dtypes of the dataset.

    Arguments: Dataframe
    Returns: Dataframe
    '''


fit_regress

'''
    Author: Aru Raghuvanshi

    This Functions Fits a model with the Train Datasets and
    predicts on a Test Dataset and evaluates its RMSE metric.

    Arguments: 5 - estimator, X_train, X_test, y_train, y_test
    Returns: Train score, Test score, RMSE
    
    '''


fit_classify

'''
    Author: Aru Raghuvanshi

    This Functions Fits a Classifier model with the Train Datasets
    and predicts on a Test Dataset and evaluates metrics via n_splits
    K-fold cross validation.

    Arguments: estimator, X_train, X_test, y_train, y_test, n_splits
    Returns: Train score, Test score, accuracy score, and displays
             classification report.
    '''

goodness_fit

'''
    Author: Aru Raghuvanshi

    The functions takes train score and testscore and returns
    goodness of fit in a DataFrame.

    Arguments: trainscore, testscore
    Returns: Dataframe
    '''


reg_comparator

'''
    Author: Aru Raghuvanshi

    Function takes 4 arguments of datasets split by train test split
    method and fits 5 regressive machine learning algos of LinearReg,
    Random Forest, Decision Tree, XGBoost and LightGBM Regressors and
    returns a dataframe with metrics.

    Arguments: 4 products of train test split method
    Returns: Dataframe, plot
    '''


clf_comparator

'''
        Author: Aru Raghuvanshi

        Function takes 4 arguments of datasets split by train test split
        method along with one of KFold value 'k', and fits 5 classifier
        machine learning algos of LogisticReg, Random Forest, Decision Tree,
        XGBoost and LightGBM classifiers and returns a dataframe with metrics.

        Arguments: four products of train test split method and kfold 'k'
        Returns: Dataframe, plot
        '''

======================= END OF FILE ============================================= ]