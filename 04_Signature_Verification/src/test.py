from users import *

# simple test without beautified toString

users = read_enrollment_users()

for u in users:
    print("User:\t", u)

    print("Signatures: \t", u.signatures, "\n")
