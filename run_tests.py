import csv
import time
import os
import random
import requests
from datetime import datetime


BASE_URL = "http://localhost:8000/ask"

VERSIONS = [
    "v0",
    "v1",
    "v2",
    "v3"
]

REPETITIONS = 3

RESULT_FILE = "results/raw_experiment_results.csv"


FIELDNAMES = [
    "timestamp",
    "prompt_id",
    "test_type",
    "attack_family",
    "role",
    "version",
    "repetition",
    "question",
    "http_status",

    "retrieved_document_ids",
    "retrieved_filenames",
    "retrieved_classifications",
    "retrieved_similarities",

    "answer",

    "protected_markers_generated",
    "blocked_or_redacted",
    "suspicious_patterns_detected",

    "latency_ms",

    "expected_keyword",
    "benign_success"
]


def load_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(file)
        )


def initialize_results():

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()


def parse_retrieved_documents(retrieved):

    document_ids = []
    filenames = []
    classifications = []
    similarities = []

    # New top-k format
    if isinstance(retrieved, list):

        for document in retrieved:

            document_ids.append(
                str(
                    document.get(
                        "document_id",
                        ""
                    )
                )
            )

            filenames.append(
                str(
                    document.get(
                        "filename",
                        ""
                    )
                )
            )

            classifications.append(
                str(
                    document.get(
                        "classification",
                        ""
                    )
                )
            )

            similarities.append(
                str(
                    document.get(
                        "similarity",
                        ""
                    )
                )
            )

    # Compatibility with older single-document format
    elif isinstance(retrieved, dict):

        document_ids.append(
            str(
                retrieved.get(
                    "document_id",
                    ""
                )
            )
        )

        filenames.append(
            str(
                retrieved.get(
                    "filename",
                    ""
                )
            )
        )

        classifications.append(
            str(
                retrieved.get(
                    "classification",
                    ""
                )
            )
        )

        similarities.append(
            str(
                retrieved.get(
                    "similarity",
                    ""
                )
            )
        )

    return {
        "document_ids":
            "|".join(document_ids),

        "filenames":
            "|".join(filenames),

        "classifications":
            "|".join(classifications),

        "similarities":
            "|".join(similarities)
    }


def run_case(
    case,
    test_type,
    version,
    repetition
):

    payload = {
        "question":
            case["question"],

        "role":
            case["role"],

        "version":
            version,

        "prompt_id":
            case["prompt_id"],

        "attack_family":
            case.get(
                "attack_family",
                "benign"
            )
    }

    try:

        response = requests.post(
            BASE_URL,
            json=payload,
            timeout=180
        )

        http_status = (
            response.status_code
        )

        data = response.json()

    except Exception as error:

        return {
            "timestamp":
                datetime.now().isoformat(),

            "prompt_id":
                case["prompt_id"],

            "test_type":
                test_type,

            "attack_family":
                case.get(
                    "attack_family",
                    "benign"
                ),

            "role":
                case["role"],

            "version":
                version,

            "repetition":
                repetition,

            "question":
                case["question"],

            "http_status":
                "ERROR",

            "retrieved_document_ids":
                "",

            "retrieved_filenames":
                "",

            "retrieved_classifications":
                "",

            "retrieved_similarities":
                "",

            "answer":
                str(error),

            "protected_markers_generated":
                "",

            "blocked_or_redacted":
                "",

            "suspicious_patterns_detected":
                "",

            "latency_ms":
                "",

            "expected_keyword":
                case.get(
                    "expected_keyword",
                    ""
                ),

            "benign_success":
                ""
        }

    retrieved = data.get(
        "retrieved",
        []
    )

    parsed_retrieval = (
        parse_retrieved_documents(
            retrieved
        )
    )

    security = data.get(
        "security",
        {}
    )

    answer = data.get(
        "answer",
        ""
    )

    expected_keyword = (
        case.get(
            "expected_keyword",
            ""
        )
    )

    benign_success = ""

    if test_type == "benign":

        benign_success = (
            expected_keyword.lower()
            in answer.lower()
        )

    markers = security.get(
        "protected_markers_generated",
        []
    )

    if isinstance(
        markers,
        list
    ):

        markers = "|".join(
            markers
        )

    suspicious_patterns = (
        security.get(
            "suspicious_patterns_detected",
            []
        )
    )

    if isinstance(
        suspicious_patterns,
        list
    ):

        suspicious_patterns = (
            "|".join(
                suspicious_patterns
            )
        )

    return {
        "timestamp":
            datetime.now().isoformat(),

        "prompt_id":
            case["prompt_id"],

        "test_type":
            test_type,

        "attack_family":
            case.get(
                "attack_family",
                "benign"
            ),

        "role":
            case["role"],

        "version":
            version,

        "repetition":
            repetition,

        "question":
            case["question"],

        "http_status":
            http_status,

        "retrieved_document_ids":
            parsed_retrieval[
                "document_ids"
            ],

        "retrieved_filenames":
            parsed_retrieval[
                "filenames"
            ],

        "retrieved_classifications":
            parsed_retrieval[
                "classifications"
            ],

        "retrieved_similarities":
            parsed_retrieval[
                "similarities"
            ],

        "answer":
            answer,

        "protected_markers_generated":
            markers,

        "blocked_or_redacted":
            security.get(
                "blocked_or_redacted",
                False
            ),

        "suspicious_patterns_detected":
            suspicious_patterns,

        "latency_ms":
            data.get(
                "latency_ms",
                ""
            ),

        "expected_keyword":
            expected_keyword,

        "benign_success":
            benign_success
    }


def main():

    benign_cases = load_csv(
        "prompts/benign_tests.csv"
    )

    attack_cases = load_csv(
        "prompts/attack_tests.csv"
    )

    initialize_results()

    all_jobs = []

    for version in VERSIONS:

        for repetition in range(
            1,
            REPETITIONS + 1
        ):

            for case in benign_cases:

                all_jobs.append(
                    (
                        case,
                        "benign",
                        version,
                        repetition
                    )
                )

            for case in attack_cases:

                all_jobs.append(
                    (
                        case,
                        "attack",
                        version,
                        repetition
                    )
                )

    random.shuffle(
        all_jobs
    )

    total_runs = len(
        all_jobs
    )

    print(
        f"Total runs: {total_runs}"
    )

    print(
        f"Results file: {RESULT_FILE}"
    )

    print()

    for index, job in enumerate(
        all_jobs,
        start=1
    ):

        (
            case,
            test_type,
            version,
            repetition
        ) = job

        print(
            f"[{index}/{total_runs}] "
            f"{case['prompt_id']} "
            f"{version} "
            f"run={repetition}"
        )

        result = run_case(
            case,
            test_type,
            version,
            repetition
        )

        with open(
            RESULT_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writerow(
                result
            )

        time.sleep(
            0.2
        )

    print()
    print(
        "Experiment complete."
    )

    print(
        f"Results saved to "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
