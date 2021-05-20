import os

import numpy as np

from signature import Signature, read_from_file

RESOURCES = "../resources/"
ENROLLMENT = RESOURCES + "enrollment/"
VERIFICATION = RESOURCES + "verification/"

USERS = RESOURCES + "users.txt"
GT = RESOURCES + "gt.txt"

ENROLLMENT_SIGNATURES_PER_USER = 5
VERIFICATION_SIGNATURES_PER_USER = 45


def read_enrollment_users() -> np.array:
    """
    Reads the enrollment dataset into a numpy array of :class:`User`.

    :return: Array of users
    """

    users = read_users(ENROLLMENT_SIGNATURES_PER_USER)

    entries = os.scandir(ENROLLMENT)
    for entry in entries:
        if entry.is_dir():
            continue

        file_name: str = entry.name.split(".")[0]
        id_split = file_name.split("-")

        user_id = int(id_split[0])
        # split[1] == g, not useful
        signature_id = int(id_split[2])
        user = users[user_id - 1]  # important offset

        signature = read_from_file(entry.path, signature_id)
        user.signatures[signature_id - 1] = signature  # important offset

    return users


def read_verification_users() -> np.array:
    """
    Reads the verification dataset into a numpy array of :class:`User`.

    :return: Array of users
    """

    users = read_users(VERIFICATION_SIGNATURES_PER_USER)
    gt = read_gt()

    entries = os.scandir(VERIFICATION)
    for entry in entries:
        if entry.is_dir():
            continue

        file_name: str = entry.name.split(".")[0]
        id_split = file_name.split("-")

        user_id = int(id_split[0])
        signature_id = int(id_split[1])
        user = users[user_id - 1]  # important offset

        signature = read_from_file(entry.path, signature_id)
        signature.is_fake = gt[file_name]
        user.signatures[signature_id - 1] = signature  # important offset

    return users


def read_gt() -> dict:
    """
    Reads in the ground-truth file. The returned dictionary consists of

    * { id: "$user_id-$signature-id", is_fake: bool }

    :return: Dictionary of ground truth data
    """
    lines = open(GT, "r").readlines()

    gt = {}
    for line in lines:
        split = line.rstrip().split(" ")
        name = split[0]
        is_fake = split[1] == "f"

        gt[name] = is_fake

    return gt


def read_users(num_signatures: int = 0) -> np.array:
    """
    Reads in the users file and allocates user data with the given number of signatures.

    :param num_signatures: The number of signatures to pre-allocate for the user
    :return: Array of users
    """

    lines = open(USERS, 'r').readlines()

    users = np.empty(shape=len(lines), dtype=User)
    for i in range(len(lines)):
        id = int(lines[i])
        signatures = np.empty(shape=num_signatures, dtype=Signature)

        users[id - 1] = User(id, signatures)  # important offset

    return users


class User:
    def __init__(self, id: int, signatures: np.array = np.empty(shape=0, dtype=Signature)):
        self.id = id
        self.signatures = signatures
