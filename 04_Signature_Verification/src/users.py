import os

import numpy as np

from signature import Signature, read_from_file

RESOURCES = "../resources/"
ENROLLMENT = RESOURCES + "enrollment/"
VERIFICATION = RESOURCES + "verification/"

USERS = RESOURCES + "users.txt"
GT = RESOURCES + "gt.txt"

# ENROLLMENT_SIGNATURES_PER_USER = 5
# SIGNATURES_PER_USER = 45


def read_enrollment_users() -> dict:
    """
    Reads the enrollment dataset into a dictionary of

    * { id: int, :class:`User` }

    with their corresponding signatures of which all are **not fake**.

    :return: dict { id: int, User }
    """

    users = read_users()

    entries = os.scandir(ENROLLMENT)
    for entry in entries:
        if entry.is_dir():
            continue

        id = int(entry.name.split("-", 1)[0])
        user = users[id]

        signature = read_from_file(entry.path)
        user.signatures = np.append(user.signatures, signature)

    return users


def read_verification_users() -> dict:
    """
    Reads the verification dataset into a dictionary of

    * { id: int, :class:`User` }

    with their corresponding signatures of which some are fake.

    :return: dict { id: int, User }
    """

    users = read_users()
    gt = read_gt()

    entries = os.scandir(VERIFICATION)
    for entry in entries:
        if entry.is_dir():
            continue

        name: str = entry.name.split(".")[0]
        id = int(name.split("-")[0])
        user = users[id]

        signature = read_from_file(entry.path)
        signature.is_fake = gt[name]

        user.signatures = np.append(user.signatures, signature)

    return users


def read_gt() -> dict:
    lines = open(GT, "r").readlines()

    gt = {}
    for line in lines:
        split = line.rstrip().split(" ")
        name = split[0]
        is_fake = split[1] == "f"

        gt[name] = is_fake

    return gt


def read_users() -> dict:
    lines = open(USERS, 'r').readlines()

    users = {}
    for i in range(len(lines)):
        id = int(lines[i])
        signatures = np.empty(shape=0, dtype=Signature)

        users[id] = User(id, signatures)

    return users


class User:
    def __init__(self, id: int, signatures: np.array = np.empty(shape=0, dtype=Signature)):
        self.id = id
        self.signatures = signatures
