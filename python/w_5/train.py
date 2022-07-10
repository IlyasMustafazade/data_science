import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import naive_bayes, tree, svm, ensemble, neighbors


def main():

    f_space = pd.read_csv("train.csv")

    print(f_space.isnull().sum())

    sns.heatmap(f_space.isnull())

    plt.show()


if __name__ == "__main__":
    
    main()


