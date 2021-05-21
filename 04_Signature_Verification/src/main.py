from distances import *
from users import *

# signature to test whether fake or genuine
test = read_verification_users()
users = read_enrollment_users()
num_users = len(users)

# compare all users test signatures against genuine
for i in range(1, num_users):
    signatures_real = users[i].signatures
    signatures_test = test[i].signatures

    correct = 0
    for test_sig in signatures_test:
        prediction = predict_fake(test_sig, signatures_real, threshold=4)
        if prediction:
            correct += 1

    accuracy = (correct / len(signatures_test)) * 100
    print('User: {0}, Average Accuracy: {1:.2f}%'.format(i, accuracy))

