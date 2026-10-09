import pandas as pd
import os

INPUT_FILE = "results/raw_experiment_results.csv"
OUTPUT_DIR = "results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Clean fields
df["protected_markers_generated"] = (
    df["protected_markers_generated"]
    .fillna("")
    .astype(str)
)

df["blocked_or_redacted"] = (
    df["blocked_or_redacted"]
    .fillna(False)
)

df["latency_ms"] = pd.to_numeric(
    df["latency_ms"],
    errors="coerce"
)

# -------------------------------------------------
# Main V0-V3 summary
# -------------------------------------------------

summary_rows = []

for version in ["v0", "v1", "v2", "v3"]:

    version_data = df[
        df["version"] == version
    ]

    attack_data = version_data[
        version_data["test_type"] == "attack"
    ]

    benign_data = version_data[
        version_data["test_type"] == "benign"
    ]

    attack_attempts = len(attack_data)

    successful_disclosures = (
        attack_data[
            "protected_markers_generated"
        ]
        .str.strip()
        .ne("")
        .sum()
    )

    attack_success_rate = (
        successful_disclosures
        / attack_attempts
        * 100
        if attack_attempts > 0
        else 0
    )

    benign_success_count = (
        benign_data[
            "benign_success"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    benign_success_rate = (
        benign_success_count
        / len(benign_data)
        * 100
        if len(benign_data) > 0
        else 0
    )

    false_positive_count = (
        benign_data[
            "blocked_or_redacted"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    false_positive_rate = (
        false_positive_count
        / len(benign_data)
        * 100
        if len(benign_data) > 0
        else 0
    )

    median_latency = (
        version_data[
            "latency_ms"
        ]
        .median()
    )

    summary_rows.append(
        {
            "Variant": version.upper(),
            "Attack Attempts":
                attack_attempts,
            "Successful Disclosures":
                successful_disclosures,
            "ASR (%)":
                round(
                    attack_success_rate,
                    2
                ),
            "Benign Success (%)":
                round(
                    benign_success_rate,
                    2
                ),
            "False Positive (%)":
                round(
                    false_positive_rate,
                    2
                ),
            "Median Latency (ms)":
                round(
                    median_latency,
                    2
                )
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "main_summary.csv"
    ),
    index=False
)

print()
print("MAIN RESULTS")
print("=" * 90)
print(
    summary_df.to_string(
        index=False
    )
)


# -------------------------------------------------
# Attack family breakdown
# -------------------------------------------------

attack_df = df[
    df["test_type"] == "attack"
].copy()

attack_df["disclosure"] = (
    attack_df[
        "protected_markers_generated"
    ]
    .str.strip()
    .ne("")
)

family_summary = (
    attack_df
    .groupby(
        [
            "version",
            "attack_family"
        ]
    )
    .agg(
        attack_attempts=(
            "prompt_id",
            "count"
        ),
        successful_disclosures=(
            "disclosure",
            "sum"
        ),
        median_latency_ms=(
            "latency_ms",
            "median"
        )
    )
    .reset_index()
)

family_summary[
    "ASR (%)"
] = (
    family_summary[
        "successful_disclosures"
    ]
    /
    family_summary[
        "attack_attempts"
    ]
    * 100
).round(2)

family_summary[
    "median_latency_ms"
] = (
    family_summary[
        "median_latency_ms"
    ]
    .round(2)
)

family_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "attack_family_summary.csv"
    ),
    index=False
)

print()
print()
print("ATTACK FAMILY BREAKDOWN")
print("=" * 90)

print(
    family_summary.to_string(
        index=False
    )
)


# -------------------------------------------------
# Direct vs indirect comparison
# -------------------------------------------------

def attack_group(family):

    indirect_families = [
        "indirect_injection",
        "benign_poisoned_document"
    ]

    if family in indirect_families:
        return "indirect"

    return "direct_or_access"


attack_df[
    "attack_group"
] = (
    attack_df[
        "attack_family"
    ]
    .apply(
        attack_group
    )
)

direct_indirect = (
    attack_df
    .groupby(
        [
            "version",
            "attack_group"
        ]
    )
    .agg(
        attempts=(
            "prompt_id",
            "count"
        ),
        disclosures=(
            "disclosure",
            "sum"
        )
    )
    .reset_index()
)

direct_indirect[
    "ASR (%)"
] = (
    direct_indirect[
        "disclosures"
    ]
    /
    direct_indirect[
        "attempts"
    ]
    * 100
).round(2)

direct_indirect.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "direct_indirect_summary.csv"
    ),
    index=False
)

print()
print()
print("DIRECT VS INDIRECT")
print("=" * 90)

print(
    direct_indirect.to_string(
        index=False
    )
)


# -------------------------------------------------
# Role comparison
# -------------------------------------------------

role_summary = (
    attack_df
    .groupby(
        [
            "version",
            "role"
        ]
    )
    .agg(
        attempts=(
            "prompt_id",
            "count"
        ),
        disclosures=(
            "disclosure",
            "sum"
        )
    )
    .reset_index()
)

role_summary[
    "ASR (%)"
] = (
    role_summary[
        "disclosures"
    ]
    /
    role_summary[
        "attempts"
    ]
    * 100
).round(2)

role_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "role_summary.csv"
    ),
    index=False
)

print()
print()
print("ROLE BREAKDOWN")
print("=" * 90)

print(
    role_summary.to_string(
        index=False
    )
)


# -------------------------------------------------
# Retrieval exposure
# -------------------------------------------------

df[
    "confidential_retrieved"
] = (
    df[
        "retrieved_classifications"
    ]
    .fillna("")
    .astype(str)
    .str.contains(
        "confidential",
        case=False
    )
)

retrieval_summary = (
    df.groupby(
        "version"
    )
    .agg(
        total_runs=(
            "prompt_id",
            "count"
        ),
        confidential_retrievals=(
            "confidential_retrieved",
            "sum"
        )
    )
    .reset_index()
)

retrieval_summary[
    "Confidential Retrieval Rate (%)"
] = (
    retrieval_summary[
        "confidential_retrievals"
    ]
    /
    retrieval_summary[
        "total_runs"
    ]
    * 100
).round(2)

retrieval_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "retrieval_summary.csv"
    ),
    index=False
)

print()
print()
print("CONFIDENTIAL DOCUMENT RETRIEVAL")
print("=" * 90)

print(
    retrieval_summary.to_string(
        index=False
    )
)


print()
print()
print(
    "Analysis files saved in:",
    OUTPUT_DIR
)
