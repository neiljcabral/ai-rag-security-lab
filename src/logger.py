import csv
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "requests.csv")


FIELDNAMES = [
    "timestamp",
    "run_id",
    "prompt_id",
    "version",
    "role",
    "attack_family",
    "question",
    "retrieved_document_id",
    "retrieved_filename",
    "retrieved_classification",
    "similarity",
    "model",
    "temperature",
    "response",
    "protected_markers_detected",
    "blocked",
    "latency_ms"
]


def initialize_log():
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writeheader()


def write_log(record):
    initialize_log()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writerow(record)
