import pandas as pd

def remove_outliers_using_iqr(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    df = df[(df[column_name] >= lower_bound) &
            (df[column_name] <= upper_bound)]

    return df

def read_training_data(dropOutliers=False):
    # Read the data
    df_train = pd.read_csv('data/project_train.csv', header=0)
    df_test = pd.read_csv('data/project_test.csv', header=0)

    # Drop outliers for 'energy' and 'loudness' using IQR
    if (dropOutliers):
        df_train = remove_outliers_using_iqr(df_train, 'energy')
        df_train = remove_outliers_using_iqr(df_train, 'loudness')

    return df_train, df_test