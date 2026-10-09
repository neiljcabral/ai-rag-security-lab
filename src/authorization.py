VALID_ROLES = [
    "guest",
    "employee",
    "admin"
]


def is_valid_role(role):
    return role in VALID_ROLES


def filter_authorized_documents(documents, role):
    return [
        document
        for document in documents
        if role in document["allowed_roles"]
    ]


def is_authorized(document, role):
    return role in document["allowed_roles"]
