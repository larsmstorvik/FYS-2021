import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Problem 1
"""

# Read data file
df = pd.read_csv("SpotifyFeatures.csv")

# 1a)
df.info()

# Add labels - Pop: 1    Classical: 0
df.loc[:, 'label'] = df['genre'].apply(lambda x: 1 if x == 'Pop' else 0)

# "Reduce data frame size" by making two seperate df
samples_classic = df.loc[df['genre'].isin(['Classical'])]
samples_pop = df.loc[df['genre'].isin(['Pop'])]


def train_and_test_set(df):
    """ Returns a 80-20% ratio of train and test data """
    ratio = 0.8
    total_rows = df.shape[0]
    train_size = int(total_rows*ratio)
    train = df[0:train_size]
    test = df[train_size:]
    return train, test

# Get dataframes of train and test data 
train_class, test_class = train_and_test_set(samples_classic)
train_pop, test_pop = train_and_test_set(samples_pop)

# Merge classical and pop songs (of the training and test sets) and shuffle them (using sample(frax=1))
train = pd.concat([train_class, train_pop]).sample(frac=1)
test = pd.concat([test_class, test_pop]).sample(frac=1)

# Convert to numpy arrays
train_features = train[['liveness', 'loudness']].to_numpy()
train_label = train['label'].to_numpy()

test_features = test[['liveness', 'loudness']].to_numpy()
test_label = test['label'].to_numpy()


def plot_livenessVSloudness(train_label, train_features):
    """ 
    Plot liveness as x and loudness as y 
    (not a good implementation but it works)
    """
    it = 0
    for lab in train_label:
        if lab == 0:
            plt.plot(train_features[it][0], train_features[it][1], 'o', color='r', label='Pop')
        else:
            plt.plot(train_features[it][0], train_features[it][1], 'o', color='blue', label='Classical')
        it += 1
    plt.xlabel("liveness")
    plt.ylabel("loudness")
    plt.savefig('livenessVSloudness.png')
    plt.show()




"""
Problem 2
Using Stochastic gradient descent (SGD)
"""

def z_pred(X, w):
    """"
    input:
    X = (x_1, x_2, 1) -> shape(3, 1)
    w = (w_1, w_2, b) -> shape(3, 1)

    return:
    z = sum(x_1*w_1 + x_2*w_2 + b)
    """
    #print(X.shape)
    #print(w.shape)
    if X.shape[0] != 3: 
        X = X.T
    return np.dot(X.T, w)
    
    #print(X)
    #print(w)
    #return X[0]*w[0] + X[1]*w[1] + X[2]*w[2]

def sigmoid(z):
    """
    input:
    z - (3,3) output from linear regression

    return:
    1.0 / (1 + np.exp(-z))
    """
    #z = np.clip(z, -500, 500)  # Clip z to prevent overflow in exp
    return 1.0 / (1.0 + np.exp(-z))

def loss_func(y_hat, y):
    """
    input:
    y_hat   - predicted label
    y       - label
    
    output:
    The cross entropy loss function
    """
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # Clip y_hat to avoid log(0)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradient_descent(X, y_hat, y):
    """
    input:
    X       : (3, 1)
    y       - label
    y_hat   - predicted label

    Returns
    gradient : (3, 1)
    """

    return np.dot(X, (y_hat - y))


# Add a column with ones to the training samples
X = np.hstack([train_features, np.ones((train_features.shape[0], 1))])  # Samples with features
y = train_label     # Labels
y = y.reshape(X.shape[0], 1)
# Initiate weight: (w_1, w_2, b)
w = np.array([1, 1, 1])*0.1
w = w.reshape((3,1))

epochs = 100
learning_rate = 0.001

def SGD(epochs, X, w, y):
    """
    Stochatic gradient descent (SGD) algorithm 
    """
    n_samples = X.shape[0]
    training_errors = []
    for num in range(epochs):
        print(f"epoch {num} running")    
        for i in range(n_samples):
            X_i = X[i].reshape((3,1))
            # calc aX+b
            z = z_pred(X_i, w)
            # Get predicted value
            y_hat = sigmoid(z)
            # Get gradient
            gradient = gradient_descent(X_i, y_hat, y[i])
            # Adjust weights
            w = w - learning_rate * gradient

        # Calculate loss
        z = z_pred(X, w)
        y_hat = sigmoid(z)
        L = loss_func(y_hat, y)
        training_errors.append(np.sum(L) / n_samples) 
    return w, training_errors


def plot_training_errorVSepochs(training_errors, epochs):
    """
    Plots training error as a function of Epochs
    """
    plt.plot(np.arange(epochs), training_errors)
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Error')
    plt.title('Training Error vs. Epochs')
    plt.grid(True)
    plt.savefig('traning_errorVSEpochs.png')
    plt.show()


def get_accuracy(w, X, labels):
    """
    Tests logistic discrimination on training set
    Return accuracy
    """
    z = z_pred(X, w)
    y_hat = sigmoid(z)
    true_pred = 0

    for i in range(len(labels)):
        if y_hat[i] >= 0.5 and labels[i] == 1:
            true_pred += 1
        elif y_hat[i] < 0.5 and labels[i] == 0:
            true_pred += 1
    return true_pred/len(labels)


def plot_decision_boundery_w_livenessVSloudness(w, X, Y):

    # Generate values for x and y
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)#np.linspace(-10, 10, 100)#X[:, 0] #np.linspace(-10, 10, 100)
    y = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)#np.linspace(-10, 10, 100)#X[:, 1] #np.linspace(-10, 10, 100)
    
    # Create a meshgrid for x and y
    x_grid, y_grid = np.meshgrid(x, y)

    # Compute z as a linear regression line
    z = x_grid*w[0] + y_grid*w[1] + w[2]

    # Apply the sigmoid function to z
    sigmoid_values = sigmoid(z)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the data points
    ax.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], 1, color='blue', label='Pop', alpha=0.6)
    ax.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], 0, color='red', label='Classical', alpha=0.6)

    # Plot the surface for the sigmoid function
    ax.plot_surface(x_grid, y_grid, sigmoid_values, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Liveness')
    ax.set_ylabel('Loudness')
    ax.set_zlabel('Predicted values')
    ax.set_title('Predicted decision boundery')
    plt.savefig('Predicted_decision_boundery.png')
    plt.show()


def confusion_matrix(w, X, labels):
    """
    Tests logistic discrimination on training set
    Create confusion matrix
    """
    z = z_pred(X, w)
    y_hat = sigmoid(z)

    confusion_matrix = [[0,0], [0,0]]
    for i in range(len(labels)):
        if y_hat[i] >= 0.5:
            if labels[i] == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[0][1] += 1
        else:
            if labels[i] == 1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[1][1] += 1

    column_names = ["Predicted Pop", "Predicted Classic"]
    rows_names = ["Actial Pop", "Actual Classic"]
    df = pd.DataFrame(confusion_matrix, index=rows_names, columns=column_names)
    #df.to_csv('confusion_matrix.csv', index=True, header=True)
    print(df)


if __name__ == "__main__":
    # 1d
    plot_livenessVSloudness(train_label, train_features)

    #2a
    w, training_errors = SGD(epochs, X, w, y)
    plot_training_errorVSepochs(training_errors, epochs)

    #2b
    # Create matrix for test samples
    X_test = np.hstack([test_features, np.ones((test_features.shape[0], 1))])
    y_test = test_label

    Accuracy_train = get_accuracy(w, X, y)
    Accuracy_test = get_accuracy(w, X_test, y_test)
    print("Accuracy train set:", Accuracy_train)
    print("Accuracy test set:", Accuracy_test)

    #2d
    plot_decision_boundery_w_livenessVSloudness(w, X_test, y_test)

    #3a
    confusion_matrix(w, X_test, y_test)
