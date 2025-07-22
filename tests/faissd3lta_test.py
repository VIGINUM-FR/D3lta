import re

import pandas as pd
import pytest

from d3lta.faissd3lta import (
    compute_embeddings,
    compute_language,
    create_index_cosine,
    semantic_faiss,
)


@pytest.fixture
def examples_dataset():
    """Returns an empty test"""
    return [
        "Je m'apelle Mimie et je fais du stop",
        "Je m'apelle Giselle et toi ?",
        "Les chats sont gris",
        "Cat's are grey, aren't they ?",
        "Cats are grey",
        "Les chats ne sont pas gris",
    ]


def test_compute_language(examples_dataset: list[str]):
    df_language = pd.DataFrame(examples_dataset, columns=["text_language_detect"])
    df_language = compute_language(df_language)
    assert list(df_language["language"]) == ["fr", "fr", "fr", "en", "en", "fr"]


def test_embedding_similarity(examples_dataset: list[str]):
    df_test = pd.DataFrame(
        examples_dataset,
        columns=["text_to_embed"],
        index=range(len(examples_dataset)),
    )  # index for checking that it has good ids
    df_emb = compute_embeddings(df_test)
    index_t = create_index_cosine(df_emb)

    test_dataset = pd.DataFrame([{"text_to_embed": "I gatti sono grigi"}])
    df_emb_test = compute_embeddings(test_dataset)

    limits, distances, indices = index_t.range_search(
        x=df_emb_test.to_numpy().reshape(1, -1), thresh=0.7
    )
    assert (
        df_test.loc[indices]["text_to_embed"]
        .str.contains("chat|cat", flags=re.IGNORECASE, na=False)
        .all()
    )


def test_semantic_faiss(examples_dataset: list[str]):
    df = pd.DataFrame(examples_dataset, columns=["text_language_detect"])
    df = compute_language(df)
    df_emb = compute_embeddings(
        df.assign(text_to_embed=lambda x: x["text_language_detect"])
    )
    df.index = df.index.astype(str)
    matches, df_clusters = semantic_faiss(
        df=df.rename(columns={"text_language_detect": "original"}),
        min_size_txt=1,
        df_embeddings_use=df_emb,
        threshold_grapheme=0.693,
        threshold_language=0.715,
        threshold_semantic=0.85,
    )
    assert (
        df_clusters.query("cluster == 0")["original"]
        .str.contains("cat|chat", flags=re.IGNORECASE)
        .all()
    )
    assert (
        matches.query(
            'text_to_embed_source == "Les chats sont gris" and text_to_embed_target == "Cats are grey"'
        )["dup_type"]
        == "translation"
    ).all()
