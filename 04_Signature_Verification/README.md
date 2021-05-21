# Installing
`pip install -r requirement.txt`

**NOTE**
The dtaidistance packages need to be bundled with a c-library binary, depending on the system installed it maybe bundled on your distribution
if not run ...
`pip install -vvv --upgrade --force-reinstall --no-deps --no-binary :all: dtaidistance`

https://dtaidistance.readthedocs.io/en/latest/usage/installation.html


# Code layout

To read in the datasets and work with it, some helper classes and methods are provided:

## signature.py

Contains the `Signature` and `SignatureDTW` classes.

To convert `Signature` to `SignatureDTW`, one can call `signature.dtw()` on the object.

To read in a signature from a file, one can use `read_from_file(str, int)`. Note that by default the created signature is
assumed to be **not fake**. This property can be changed with `signature.is_fake: bool`.

## users.py

The heart of reading in the datasets.

Each `User` has an `id: int` and an array of signatures. 

To read in the users and their corresponding signatures we provide the methods

- `read_enrollment_users()`
- `read_verification_users()`

Both return a numpy array of users.

Note that both methods currently use a hardcoded string which is defined at the top of this file.

## test.py

Just a simple test that the reading works.
