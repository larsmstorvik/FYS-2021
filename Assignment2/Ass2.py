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



def gamma_func(alpha):
    a = 1
    for i in range(1, alpha+1):
        a = a * i
    return a

def calculate_gamma_distrobution(x, alpha, beta):
    # Split function into parts and return the hole function
    part1 = (beta**alpha) * gamma_func(alpha)
    part2 = x**(alpha-1)
    part3 = np.e**(x/beta)
    return (1/part1)*part2*part3

def calculate_gaussian_distrobution(x, mean, sd):
    part1 = np.sqrt(sd)*np.sqrt(2*np.pi)
    part2 = (x-mean)**2
    return (1/part1) * np.e**(-1)*(1/2)*(part2/sd)



# Plot functions
def plot_gamma_distrobution(samples, alpha, beta):
    numbins = 200
    #plt.hist(samples, numbins)
    x = np.linspace (0, 30, 100) 
    y = stats.gamma.pdf(x, a=alpha, scale=(beta)) * p_C0
    #y = calculate_gamma_distrobution(x, alpha, beta) * p_C0
    plt.plot(x, y)
    plt.savefig("estimated_gamma_distrobution.png")
    plt.show()

def plot_gaussian_distrobution(samples, mean, sd):
    x = np.linspace (-20, 60, 100) 
    #y = stats.norm.pdf(x, mean, sd)
    y = calculate_gaussian_distrobution(x, mean, sd)
    plt.plot(x, y)
    plt.savefig("estimated_gaussian_distrobution.png")
    plt.show()

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

def plot_samples(x0, x1):
    plt.plot(x0, np.zeros(np.shape(x0)), 'o')
    plt.plot(x1, np.ones(np.shape(x1)), 'o')
    plt.show()



def test_one_sample(xi, mean, sd, alpha, beta):
    p0 = stats.gamma.pdf(xi, a=alpha, scale=beta)
    p1 = stats.norm.pdf(xi, mean, sd)
    if (p0*p_C0) > (p1*p_C1):
        return 0
    else:
        return 1
misclassified = []
def test_samples(test_samples, test_labels, mean, sd, alpha, beta):
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
                continue
            misclassified.append(test_samples[i])
            class0_false+=1
        else:
            classified_1+=1
            if (test_labels[i]):
                class1_true+=1
                continue
            class1_false+=1
            misclassified.append(test_samples[i])

    accuracy = (class0_true + class1_true)/len(test_samples)
    print(f"of {len(test_samples)} samples --- C0 got {classified_0} and C1 got {classified_1}")
    print(f"in C0, {class0_true} was true and {class0_false} was false")
    print(f"in C1, {class1_true} was true and {class1_false} was false")
    print(f"Accuracy was {accuracy}%")




if __name__ == "__main__":
    #derive_info_data()

    beta_hat = calc_beta_hat(train_data_0)
    #plot_gamma_distrobution(train_data_0, 2, beta_hat)

    mean_hat = calc_mean_hat(train_data_1)
    sigma_hat = calc_sigma_hat(train_data_1, mean_hat)
    #plot_gaussian_distrobution(train_data_1, mean_hat, sigma_hat)
    plot_both_distro(mean_hat, sigma_hat, 2, beta_hat)
    #plot_samples(train_data_0, train_data_1)

    #test_one_sample(9.146, mean_hat, sigma_hat, 2, beta_hat)
    test_samples(test_data, test_labels, mean_hat, sigma_hat, 2, beta_hat)
    #print(beta_hat)
    #print(mean_hat)
    #print(sigma_hat)


