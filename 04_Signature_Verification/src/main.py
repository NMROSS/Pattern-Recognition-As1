from distances import *
from users import *
import numpy as np
import sklearn.metrics as skl
import matplotlib.pyplot

# signature to test whether fake or genuine
test = read_verification_users()
users = read_enrollment_users()
num_users = len(users)

# evaluation variables
gt = list(read_gt().values())
thresholds = np.arange(start=5, stop=20, step=0.5)
precisions = []
recalls = []
APs = []

# compare all users test signatures against genuine
for threshold in thresholds:
    pred = []
    print("Calclating for threshold: ", threshold)
    accuracies = []
    for i in range(0, num_users):
        signatures_real = users[i].signatures
        signatures_test = test[i].signatures

        correct = 0
        for test_sig in signatures_test:
            # prediction = predict_fake(test_sig, signatures_real, threshold, verbose=False)
            prediction = predict(test_sig, signatures_real, threshold)
            if prediction == test_sig.is_fake:
                correct += 1

            pred.append(prediction)

        accuracy = (correct / len(signatures_test)) * 100
        accuracies.append(accuracy)
#        print('User: {0}, Average Accuracy: {1:.2f}%'.format(i, accuracy))

    ##### evaluation #####

    # precision and recall
    precision = skl.precision_score(gt, pred)
    recall = skl.recall_score(gt, pred)
    precisions.append(precision)
    recalls.append(recall)
    print("Precision: {:.2f}%".format(precision*100))
    print("Recall: {:.2f}%".format(recall*100))
    print("Accuracy: {:.2f}%".format(np.average(accuracies, axis=0)))
    print("\n")

# precision-recall curve
matplotlib.pyplot.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
matplotlib.pyplot.show()
