import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
filename = "/Users/larsm/OneDrive - UiT Office 365/Documents/FYS-2021/FYS-2021/Assignment2/data_problem2.csv"


# Read csv file
df = pd.read_csv("data_problem2.csv", header=None)

# Get array of samples and labels
samples = df.iloc[0, :].values
labels = df.iloc[1, :].values

# Ratio of trainingset to test set
ratio = int(len(samples)*0.8)

# Divide into training and test data
train_data = samples[:ratio]
test_data = samples[ratio:]
train_labels = labels[:ratio]
test_labels = labels[ratio:]


# Divide training data into two classes
train_data_0 = np.array([train_data[i] for i in range(len(train_data)) if (train_labels[i] == 0.0)])
train_data_1 = np.array([train_data[i] for i in range(len(train_data)) if (train_labels[i] == 1.0)])


def derive_info_data():
    """Function to derive simple information from the data"""
    print(f"Number of sampels is {len(samples)}")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of train samples in class 0: {len(train_data_0)}")
    print(f"Number of train samples in class 1: {len(train_data_1)}")

    numbins1 = 50
    numbins2 = 80

    # Plot samples as histogram with improvements:
    plt.figure(figsize=(10, 6))  # Adjusts the size of the figure
    plt.hist(train_data_0, bins=numbins1, color='blue', alpha=0.7, label='Class 0', edgecolor='black')
    plt.hist(train_data_1, bins=numbins2, color='green', alpha=0.7, label='Class 1', edgecolor='black')

    # Add titles and labels
    plt.title('Histogram of Train Data', fontsize=16)
    plt.xlabel('Sample Values', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Add a legend
    plt.legend(loc='upper right')

    # Show gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("samples_histogram.png")
    # Display the plot
    plt.show()



# Calculate p(C_0) and p(C_1)
p_C0 = len(train_data_0)/len(train_data)
p_C1 = len(train_data_1)/len(train_data)


# Calculate beta, mean, sigma
def calc_beta_hat(samples):
    alpha = 2
    sum_samp = np.sum(samples)
    return sum_samp/(alpha*len(samples))

def calc_mean_hat(samples):
    sum_samp = np.sum(samples)
    return sum_samp /(len(samples))

def calc_sigma_hat(samples, mean_hat):
    sumation = np.sum((samples-mean_hat)**2)
    return sumation/len(samples)


""" Function to plot both distrobutions using the estimated parameters"""
def plot_both_distro(mean, sd, alpha, beta):
    x = np.linspace(-20, 100, 241)
    y1 = stats.norm.pdf(x, mean, sd) * p_C1
    y2 = stats.gamma.pdf(x, a=alpha, scale=(beta)) * p_C0
    
     # Create the plot
    plt.figure(figsize=(10, 6))  # Adjust figure size
    # Plot Gaussian distribution
    plt.plot(x, y1, label='Gaussian Distribution', color='blue', linestyle='-', linewidth=2)
    # Plot Gamma distribution
    plt.plot(x, y2, label='Gamma Distribution', color='green', linestyle='--', linewidth=2)
    # Add titles and labels
    plt.title('Comparison of Gaussian and Gamma Distributions', fontsize=16, fontweight='bold')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    # Add a legend
    plt.legend(fontsize=12)
    plt.savefig("both_distrobutions.png")
    # Show the plot
    plt.show()


""" Function returning which of the two classes a samples i most likely in"""
def test_one_sample(xi, mean, sd, alpha, beta):
    p0 = stats.gamma.pdf(xi, a=alpha, scale=beta)
    p1 = stats.norm.pdf(xi, mean, sd)
    if (p0*p_C0) > (p1*p_C1):
        return 0
    else:
        return 1


# Lists of misclassified / correctly classified samples
missclassified = []
not_missclassified = []
def test_samples(test_samples, test_labels, mean, sd, alpha, beta):
    """ Test the training set on the Bayes' classification"""
    classified_0 = 0
    classified_1 = 0
    class0_true = 0
    class0_false = 0
    class1_true = 0
    class1_false = 0

    for i in range(len(test_samples)):
        if(not test_one_sample(test_samples[i], mean, sd, alpha, beta)):
            classified_0+=1
            if (not test_labels[i]):
                class0_true+=1
                not_missclassified.append(test_samples[i])
                continue
            missclassified.append(test_samples[i])
            class0_false+=1
        else:
            classified_1+=1
            if (test_labels[i]):
                class1_true+=1
                not_missclassified.append(test_samples[i])
                continue
            class1_false+=1
            missclassified.append(test_samples[i])

    accuracy = (class0_true + class1_true)/len(test_samples)
    print(f"of {len(test_samples)} samples --- C0 got {classified_0} and C1 got {classified_1}")
    print(f"in C0, {class0_true} was true and {class0_false} was false")
    print(f"in C1, {class1_true} was true and {class1_false} was false")
    print(f"Accuracy was {accuracy}%")


""" Plotting misclassified samples with correctly classified samples """
def plot_misclassifiedData(not_missclassified, missclassified):
    # Create the figure and axis
    plt.figure(figsize=(8, 6))
    # Plot the histogram for missclassified data (e.g., red)
    plt.hist(missclassified, bins=60, alpha=0.6, color='red', label='Misclassified')

    # Plot the histogram for correctly classified data (e.g., blue)
    plt.hist(not_missclassified, bins=60, alpha=0.6, color='blue', label='Correctly Classified')

    # Add labels and legend
    plt.xlabel('Data Points')
    plt.ylabel('Frequency')
    plt.title('Histogram of Misclassified vs Correctly Classified Data')
    plt.legend(loc='upper right')
    plt.savefig("misclassified.png")
    # Show the plot
    plt.show()




if __name__ == "__main__":
    derive_info_data()

    beta_hat = calc_beta_hat(train_data_0)
    mean_hat = calc_mean_hat(train_data_1)
    sigma_hat = calc_sigma_hat(train_data_1, mean_hat)

    plot_both_distro(mean_hat, sigma_hat, 2, beta_hat)

    test_samples(test_data, test_labels, mean_hat, sigma_hat, 2, beta_hat)

    plot_misclassifiedData(np.array(not_missclassified), np.array(missclassified))


