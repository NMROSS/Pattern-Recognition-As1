from users import *

# simple test without beautified toString

users = read_enrollment_users()

for id in users:
    user = users[id]
    print(user)

    if len(user.signatures) != 0:
        print(user.signatures[0].t)
        break
