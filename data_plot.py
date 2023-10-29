import matplotlib.pyplot as plt
from data_reader import read_training_data

# reading data
df_train, df_test = read_training_data(False)
def plot_combined_histograms_2col(df, label_column, bins=30):

    unique_labels = df[label_column].unique()
    
    # Splitting data based on the labels
    data_0 = df.loc[df[label_column] == unique_labels[0]]
    data_1 = df.loc[df[label_column] == unique_labels[1]]
    
    # Number of features to plot
    n_features = len(df.columns) - 1  # Subtracting one for the label column

    # Calculate the number of rows needed for n features in a 2-column layout
    n_rows = -(-n_features // 2) 

    # Setting up the grid for the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 5 * n_rows))
    
    # Flatten axes for easy iteration
    axes = axes.ravel()
    
    # Remove any unused subplots
    for ax in axes[n_features:]:
        fig.delaxes(ax)

    for ax, column in zip(axes, df.columns):
        if column != label_column:
            ax.hist([data_0[column], data_1[column]], bins=bins, label=[f"Label {unique_labels[0]}", f"Label {unique_labels[1]}"], alpha=0.6)
            ax.set_title(f'Histogram for {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_combined_histograms_2col(df_train, "Label")
