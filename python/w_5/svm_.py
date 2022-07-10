from sklearn import svm

def main(): 

    linear_kernel_svm_cl = svm.SVC(kernel="linear")

    poly_kernel_svm_cl = svm.SVC(kernel="poly", degree=2)

    rbf_kernel_svm_cl = svm.SVC(kernel="rbf", gamma="auto")


if __name__ == "__main__": 

    main()


