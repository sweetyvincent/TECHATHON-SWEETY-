def main():
    print("Healthcare Aid Application")
    print("1. Disease Prediction")
    print("2. Insurance Co-pay/Premium Estimation")
    choice = input("Enter your choice: ")

    if choice == "1":
        symptoms = input("Enter symptoms (comma separated): ")
        predictions = predictDisease(symptoms)
        print("Predictions:")
        print("RF Model:", predictions["rf_model_prediction"])
        print("Naive Bayes:", predictions["naive_bayes_prediction"])
        print("SVM Model:", predictions["svm_model_prediction"])
        print("Final Prediction:", predictions["final_prediction"])

    elif choice == "2":
        bmi = float(input("Enter BMI: "))
        smoker = int(input("Enter smoker status (0/1): "))
        region = int(input("Enter region (0/1/2): "))
        co_pay = estimateInsuranceCoPay(bmi, smoker, region)
        print("Estimated Insurance Co-pay/Premium:", co_pay)

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
