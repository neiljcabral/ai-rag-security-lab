import os
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DOCUMENT_DIR = os.path.join(
    BASE_DIR,
    "documents"
)

MANIFEST_PATH = os.path.join(
    DOCUMENT_DIR,
    "corpus_manifest.csv"
)


class RAGEngine:

    def __init__(self):

        self.documents = []

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_matrix = None

        self.load_documents()

        self.build_index()


    def load_documents(self):

        self.documents = []

        with open(
            MANIFEST_PATH,
            "r",
            encoding="utf-8"
        ) as manifest:

            reader = csv.DictReader(
                manifest
            )

            for row in reader:

                file_path = os.path.join(
                    DOCUMENT_DIR,
                    row["filename"]
                )

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    content = file.read()

                self.documents.append(
                    {
                        "document_id":
                            row["document_id"],

                        "filename":
                            row["filename"],

                        "classification":
                            row["classification"],

                        "allowed_roles":
                            row["allowed_roles"].split("|"),

                        "purpose":
                            row["purpose"],

                        "content":
                            content
                    }
                )


    def build_index(self):

        texts = [
            document["content"]
            for document in self.documents
        ]

        self.document_matrix = (
            self.vectorizer.fit_transform(
                texts
            )
        )


    def retrieve(
        self,
        query,
        documents=None,
        top_k=3
    ):

        if documents is None:
            documents = self.documents

        if not documents:
            return []

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        document_indices = [
            self.documents.index(document)
            for document in documents
        ]

        candidate_matrix = (
            self.document_matrix[
                document_indices
            ]
        )

        similarities = (
            cosine_similarity(
                query_vector,
                candidate_matrix
            )[0]
        )

        ranked_results = sorted(
            zip(
                documents,
                similarities
            ),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        actual_top_k = min(
            top_k,
            len(ranked_results)
        )

        for document, score in (
            ranked_results[:actual_top_k]
        ):

            result = {
                **document,
                "similarity": float(score)
            }

            results.append(
                result
            )

        return results


rag_engine = RAGEngine()
