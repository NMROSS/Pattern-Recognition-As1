import distances
from users import *

users = read_enrollment_users()

user = users[1]  # Select user 001
signatures = user.signatures  # user 001 Real signatures

# Calculate the average distance
distances.mean_distance(signatures)
