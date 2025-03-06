import regex
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# # Load the data
def format_input_text(row):
    template = "Title: {Title}\n{Body}\nTags: {Tags}"
    return template.format(**row.to_dict())

train = pd.read_csv("stack_overflow_questions_train.csv")
texts = train.apply(format_input_text, axis=1)

# # Plot the lengths
sns.displot(texts.map(len), log_scale=True)
train["text_length"] = text_lengths = texts.map(len)
long_text = texts[text_lengths == text_lengths.max()].iloc[-1]
sns.displot(x="text_length", data=train, hue="Y", log_scale=True)

# # Explore tags
tag_regex = regex.compile(r"<(.*?)>")

def get_tag_counts(tags):
    counter = Counter()
    for _tags in tags.map(tag_regex.findall):
        counter.update(_tags)
    return (
        pd.Series(counter)
        .rename("count")
        .to_frame()
        .assign(rank=lambda df: (-df["count"]).rank())
    )

sns.scatterplot(get_tag_counts(train["Tags"]), x="rank", y="count")
plt.gca().set(xscale="log", yscale="log")

tag_counts = pd.concat({label: get_tag_counts(group) for label, group in train["Tags"].groupby(train["Y"])}, axis=0)

plot_data = tag_counts.unstack(level=0)
for rank in train["Y"].unique():
    ax = None
    for count in train["Y"].unique():
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax = sns.scatterplot(plot_data, x=("rank", rank), y=("count", count), ax=ax, label=count)
    ax.set(xscale="log", yscale="log", ylabel="count", title=f"rank{rank} vs count");

plt.show()

# # Classification of tag sequences

from sklearn.feature_extraction.text import CountVectorizer
import torch


"""
# # Test mistral embedding generation
import os
import dotenv
from mistralai import Mistral

dotenv.load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
model = "mistral-embed"
embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=[long_text],
)
len(long_text)
client.embeddings?
client.embeddings.create?
embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=[long_text[:len(long_text)//2]],
)

def iter_chunks(text, chunk_size=2048):
    text = text.strip()
    yield from (text[i:i + chunk_size] for i in range(0, len(text), chunk_size))

input_text = next(iter(iter_chunks(long_text)))

embeddings_batch_response = client.embeddings.create(model=model, inputs=[input_text])
"""

"""
IDEAS
-----
We generate training data where an example is encoded as a single binary vector
of dimension |UniqueTags|. We train a model like word2vec on the tag context
and use that to detect anomalies.

Also, we encode chunks of 2048 char and create embedding sequences with Mistral
to train a transformer from scratch. The transformer is trained directly on the
final classification task.
"""

# % save -r exploration 1-144

