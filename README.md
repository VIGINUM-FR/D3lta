<h2 align="center"> <a href="https://arxiv.org/abs/2312.17338">D3lta</a></h2>

<h5 align="center"> 

If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

</h5>

<div align=center>

[![PyPI - Version](https://img.shields.io/pypi/v/d3lta?style=flat&logo=pypi&logoColor=%233775A9&label=PyPI)](https://pypi.org/project/d3lta/)
[![arXiv](https://img.shields.io/badge/Arxiv-2312.17338-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.17338)

This repository is the official implementation of D3lta, a library for detecting duplicate verbatim contents within a vast amount of documents.

It distinguishes 3 types of duplicate contents : copypasta (almost exact duplicates), rewording and translation. You can run it on CPU.
</div>

---

<img style="display: block; margin: auto;" src="https://github.com/VIGINUM-FR/D3lta/raw/main/static/graph.gif"/>


## 💻 Installing 

Clone the repository

```bash
git clone https://github.com/VIGINUM-FR/D3lta
```

Navigate to the project

```bash
cd D3lta
```

Install the package

```bash
pip install -e .
```

## 🚀 Quick start

One can use directly `semantic_faiss` function from a Dataframe that contains texts.
We use by default the embeddings from the [Universal Sentence Encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow1/lite/2)
but one can use other models to calculate embeddings.


```python
import pandas as pd
from d3lta.faissd3lta import *

examples_dataset = [
        "Je m'apelle Mimie et je fais du stop",
        "Je m'apelle Giselle et toi ?",
        "Les chats sont gris",
        "Cat's are grey, aren't they ?",
        "Cats are grey",
        "Les chats ne sont pas gris",
    ]
df = pd.DataFrame(examples_dataset, columns=["text_language_detect"])
df.index = df.index.astype(str)

matches, df_clusters = semantic_faiss(
    df=df.rename(columns={"text_language_detect": "original"}),
    min_size_txt=10,
    embeddings_to_save='myembeddings',
    threshold_grapheme=0.693,
    threshold_language=0.715,
    threshold_semantic=0.85,
)

>>>matches

  source target     score duplicates language_source           text_to_embed_source  text_grapheme_source language_target           text_to_embed_target   text_grapheme_target     dup_type  score_lev
0      2      3  0.745741        2-3              fr            Les chats sont gris      leschatssontgris              en  Cat's are grey, aren't they ?   catsaregreyarentthey  translation        NaN
1      2      4  0.955517        2-4              fr            Les chats sont gris      leschatssontgris              en                  Cats are grey            catsaregrey  translation        NaN
2      2      5  0.808805        2-5              fr            Les chats sont gris      leschatssontgris              fr     Les chats ne sont pas gris  leschatsnesontpasgris   copy-pasta   0.761905
5      3      5  0.833525        3-5              en  Cat's are grey, aren't they ?  catsaregreyarentthey              fr     Les chats ne sont pas gris  leschatsnesontpasgris  translation        NaN
8      4      5  0.767601        4-5              en                  Cats are grey           catsaregrey              fr     Les chats ne sont pas gris  leschatsnesontpasgris  translation        NaN

>>>df_clusters
                               original language                 text_grapheme                         text_to_embed                  text_language_detect  cluster
0  Je m'apelle Mimie et je fais du stop       fr  jemapellemimieetjefaisdustop  Je m'apelle Mimie et je fais du stop  Je m'apelle Mimie et je fais du stop      NaN
1          Je m'apelle Giselle et toi ?       fr         jemapellegiselleettoi          Je m'apelle Giselle et toi ?          Je m'apelle Giselle et toi ?      NaN
2                   Les chats sont gris       fr              leschatssontgris                   Les chats sont gris                   Les chats sont gris      0.0
3         Cat's are grey, aren't they ?       en          catsaregreyarentthey         Cat's are grey, aren't they ?         Cat's are grey, aren't they ?      0.0
4                         Cats are grey       en                   catsaregrey                         Cats are grey                         Cats are grey      0.0
5            Les chats ne sont pas gris       fr         leschatsnesontpasgris            Les chats ne sont pas gris            Les chats ne sont pas gris      0.0
```

Its also possible to use [Faiss](https://github.com/facebookresearch/faiss) to find similar embeddings.

```python
import pandas as pd
from d3lta.faissd3lta import *

examples_dataset = [
        "Je m'apelle Mimie et je fais du stop",
        "Je m'apelle Giselle et toi ?",
        "Les chats sont gris",
        "Cat's are grey, aren't they ?",
        "Cats are grey",
        "Les chats ne sont pas gris",
    ]
    
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

>>>df_test.loc[indices]["text_to_embed"]

2              Les chats sont gris
3    Cat's are grey, aren't they ?
4                    Cats are grey
5       Les chats ne sont pas gris
Name: text_to_embed, dtype: object
```

It is also possible to use your own embedding (other than Universal Sentence Encoder). For example: 

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from d3lta.faissd3lta import *

examples_dataset = [
        "Je m'apelle Mimie et je fais du stop",
        "Je m'apelle Giselle et toi ?",
        "Les chats sont gris",
        "Cat's are grey, aren't they ?",
        "Cats are grey",
        "Les chats ne sont pas gris",
    ]
df = pd.DataFrame(examples_dataset, columns=["text_language_detect"])
df.index = df.index.astype(str)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
new_emb = model.encode(df['text_language_detect'].values.tolist())
df_emb = pd.DataFrame(new_emb, index=df.index)

matches, df_clusters = semantic_faiss(
    df=df.rename(columns={"text_language_detect": "original"}),
    min_size_txt=10,
    df_embeddings_use=df_emb,
    threshold_grapheme=0.693,
    threshold_language=0.715,
    threshold_semantic=0.85,
)

matches
```



## 📚 Synthetic dataset

The dataset is available in the release `1.0.0`. It contains the following files:

### `synthetic_dataset_documents.csv`:

This file contains all seeds (real and original texts) and their generated variations (copy-pasta, rewording or translations). 
There are 2985 documents in this corpus dataset generated using a large language model.

Columns details:
- doc_id (int): unique number associated to each text. All seed index are multiples of 10 and followed by their 9 transformations.
- original (str): real or transformed text
- text_type (str): dataset where the seed was extracted (`books`, `news`, `tweets`)
- language (str): language of the text
- prompt (str): prompt given to ChatGPT for "copypasta" and "rewording"
- seed (bool): True if the text is one of the 300 initial texts from which the generation is from

The 300 initial texts (seeds) have been taken from three Kaggle datasets : 
- https://www.kaggle.com/competitions/nlp-getting-started/data
- https://www.kaggle.com/datasets/abireltaief/books-reviews
- https://www.kaggle.com/datasets/rmisra/news-category-dataset

(For more info, please refer to the [paper](https://arxiv.org/abs/2312.17338))

### `synthetic_dataset_pairs_unbalanced.csv`:

This file contains the 1,497,547 annotated pairs of text of the synthetic dataset : 4,500 pairs of translation, 4,030 pairs of copy-pasta, 4017 pairs of rewording and 1,485,000 pairs of non duplicated content called "nomatch".

Column details: 
- source_target (str): unique id for the pair.
- source (int): index of the first text of the pair in the synthetic_dataset_documents.csv
- target (int): index of the second text of the pair in the synthetic_dataset_documents.csv
- original_source (str): text of the source index
- original_target (str): text of the target index
- language_source (str): language of original_source
- language_target (str): language of original_target
- true_label (str): transformation relation that links both text of the pair i.e. the source and target texts are {true_label} of each other. The true_label can be "copypasta", "rewording" or "translation"

## Notebooks

In folder the [`notebooks`](./notebooks/), you can find: 
- [`example_synthetic_dataset.ipynb`](./notebooks/example_synthetic_dataset.ipynb): Example of applying threedelta methodology to the synthetic dataset, with a comparison to the true labels.


## Citation

If you find our paper and code useful in your research, please consider giving a star 🌟  and a citation 📝:

```BibTeX
@misc{richard2023unmasking,
      title={Unmasking information manipulation: A quantitative approach to detecting Copy-pasta, Rewording, and Translation on Social Media}, 
      author={Manon Richard and Lisa Giordani and Cristian Brokate and Jean Liénard},
      year={2023},
      eprint={2312.17338},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2312.17338}, 
}
```
