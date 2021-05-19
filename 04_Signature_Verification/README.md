# Code layout
To read in the datasets and work with it, some helper classes and methods are provided:

## signature.py
Contains the `Signature` and `SignatureDTW` classes.

To convert `Signature` to `SignatureDTW`, one can call `signature.dtw()` on the object.

To read in a signature from a file, one can use `read_from_file(str)`.
Note that by default the created signature is assumed to be **not fake**. This property can be changed
with `signature.is_fake: bool`.

## users.py
The heart of reading in the datasets.

Each `User` has an `id: int` and an array of signatures. As it stands right now, one can only
append a new signature to the user by calling `user.signatures = np.append(user.signatures, new_signature)`.
This might change in the future but it's good enough for now. 

To read in the users and their corresponding signatures we provide the methods
- `read_enrollment_users()`
- `read_verification_users()`

Both return a dictionary of users with type `{ id, User }` where the `id` is the same as that from the linked user. 
This might change in the future as it's currently only used to 

Note that both methods currently use a hardcoded string which is defined at the top of this file.

## test.py
Just a simple test that the reading works.