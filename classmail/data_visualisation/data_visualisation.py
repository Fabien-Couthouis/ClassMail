# data_visualisation/data_visualisation.py
import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


def get_texts_from_label(df, col_text, col_label, label):
    """
    Get Pandas series corresponding to the col_text column of the specified label

    Parameters
    ----------
    df: pandas dataframe
        Dataframe to extract the terms from
    col_text: str
        Name of the column containing the text
    col_label: str
        Name of the column containing the label
    label: str
        Name of the label to analyse

    Return
    -------
    Pandas Series
    """
    df_my_label = df[col_label] == label
    texts_from_my_label = df[col_text][df_my_label]
    return texts_from_my_label


def show_string_length(series):
    """Show statistical information about row length in the series"""
    max_length = series.str.len().max()
    min_length = series.str.len().min()
    mean = series.str.len().mean()
    median = series.str.len().median()

    print("max_length: ", max_length,
          "\nmin_length: ", min_length,
          "\nmean: ", mean,
          "\nmedian: ", median)


def show_before_after(series_before, series_after, index=0):
    """Show a text before / after cleaning at the given index"""

    print("Before : \n", series_before.iloc[index],
          "\n\n______________________________________\n\n",
          "After :\n", series_after.iloc[index])


def plot_wordcloud(series, word_frequencies=None, title="", title_font_size=40,
                   background_color="black", width=2500, height=1800):
    """Plot a wordcloud showing most frequent words from the given series"""
    cloud = WordCloud(
        background_color=background_color,
        collocations=False,
        width=width,
        height=height
    )

    if word_frequencies is None:
        word_frequencies = get_term_frequencies(series)

    cloud.generate_from_frequencies(word_frequencies)

    plt.axis('off')
    plt.title(title, fontsize=title_font_size)
    plt.imshow(cloud)
    plt.show()


def get_term_frequencies(series, min_str_len=1, ngram=1):
    """
    Get number of occurence of each term in the given pandas series

    Parameters
    ----------
    series: pandas series
        Series to extract the term frequencies from
    min_str_len: int, optional (default value=1)
        Minimum string length for a word
    ngram: int, optional (default value=1)
        Number of words to get frequency from.

    Return
    -------
    Python dict ({word:frequency})
    """
    text = " ".join(series.values.astype(str))
    regex = re.compile(r"\w+{}".format(r" \w+"*(ngram-1)))
    words = re.findall(regex, text.lower())
    words = [w for w in words if len(w) >= min_str_len]
    word_dict = Counter(words)

    return dict(word_dict)


def _get_relevance_scores(df, col_text, col_label, label, get_frequencies):
    """
    Get a relevance score for all terms in the specified label. 
    Adapted from this article : https://arxiv.org/ftp/arxiv/papers/1608/1608.07094.pdf
    """
    text_series = df[col_text].str.lower()
    label_series = df[col_label]
    # Texts corresponding to the specified label
    this_cat_series = text_series[label_series == label]

    word_freq_class = get_term_frequencies(this_cat_series)
    word_freq_corpus = get_term_frequencies(text_series)

    # Compute class_weight
    this_cat_size = len(this_cat_series)
    cat_sizes = len(text_series)
    class_weight = this_cat_size / cat_sizes

    scores = []

    for index, word in enumerate(word_freq_class):
        scores.append([])
        # Compute class_term_weight
        class_frequency = this_cat_series.str.contains(
            word, regex=False, na=False).sum()
        corpus_frequency = text_series.str.contains(
            word, regex=False, na=False).sum()

        class_term_weight = _get_log_normalised_ratio(
            class_frequency, corpus_frequency)

        # Compute class_term_density
        class_term_frequency = word_freq_class[word]
        corpus_term_frequency = word_freq_corpus[word]
        class_term_density = _get_log_normalised_ratio(
            class_term_frequency, corpus_term_frequency)

        # Compute score
        score = class_weight*class_term_weight*class_term_density

        # Add word/score pairs to the list
        scores[index].append(word)
        scores[index].append(score)

        # Add frequencies if needed
        if get_frequencies:
            scores[index].append(class_term_frequency)
            scores[index].append(corpus_term_frequency)

    return scores


def _get_log_normalised_ratio(numerator, denominator):
    normalised_numerator = math.log(numerator)
    normalised_denominator = 1 + math.log(denominator)
    return normalised_numerator/normalised_denominator


def get_all_relevant_terms(df, col_text, col_label, label_names, get_frequencies=False, export_to=None, encoding="ANSI"):
    """
    Get a dataframe showing most relevant terms for a all labels

    Parameters
    ----------
    df: pandas dataframe
        Dataframe to extract the terms from
    col_text: str
        Name of the column containing the text
    col_label: str
        Name of the column containing the label
    label_names: list<str>
        Names of the labels
    get_frequencies: boolean, optional (default value=False)
        If set to True, display frequency of the term within the specified category and in other categories in the dataframe.
    export_to: str, optional (default value=None)
        Path to the csv file containing the generated dataframe of relevant terms
    encoding: str, optional (default value="ANSI")
        Encoding of the csv file (see pandas doc : to_csv())

    Return
    -------
    Pandas DataFrame (columns=['word', 'relevance_score'] or ['word', 'relevance_score','class_frequency', 'corpus_frequency'] 
    if get_frequencies is set to True)
    """
    all_relevant_terms = pd.DataFrame()

    for label in label_names:
        relevant_terms = get_relevant_terms(
            df, col_text, col_label, label, get_frequencies)
        all_relevant_terms[label + " (word)"] = relevant_terms['word']

        if get_frequencies:
            all_relevant_terms[label +
                               " (class_frequency)"] = relevant_terms['class_frequency']
            all_relevant_terms[label +
                               " (corpus_frequency)"] = relevant_terms['corpus_frequency']

    if export_to is not None:
        all_relevant_terms.to_csv(export_to, sep=";", encoding=encoding)

    return all_relevant_terms


def get_relevant_terms(df, col_text, col_label, label, get_frequencies=False, plot=False,
                       nb_words_to_plot=10, plot_reversed=True, palette="Blues", title="default"):
    """
    Get a dataframe showing most relevant terms for a specified label

    Parameters
    ----------
    df: pandas dataframe
        Dataframe to extract the terms from
    col_text: str
        Name of the column containing the text
    col_label: str
        Name of the column containing the label
    label: str
        Name of the label to analyse
    get_frequencies: boolean, optional (default value=False)
        If set to True, display frequency of the term within the specified category and in other categories in the dataframe.
    plot: boolean, optional (default value=False)
        Plot most relevant terms with associated scores

    Return
    -------
    Pandas DataFrame (columns=['word', 'relevance_score'] or ['word', 'relevance_score','class_frequency', 'corpus_frequency'] 
    if get_frequencies is set to True)
    """

    if title == "default":
        title = "Relevance scores for terms in {}".format(label)

    scores_freqs = _get_relevance_scores(
        df, col_text, col_label, label, get_frequencies)
    # Sort by score (kv[1])
    sorted_scores_freqs = sorted(
        scores_freqs, key=lambda val: val[1], reverse=True)

    df_scores = pd.DataFrame(
        sorted_scores_freqs,
        columns=["word", "relevance_score",
                 "class_frequency", "corpus_frequency"]
        if get_frequencies else ["word", "relevance_score"])

    if plot:
        sns.barplot(y="word", x="relevance_score",
                    data=df_scores[:nb_words_to_plot], palette=palette).set_title(title)
    return df_scores


def plot_word_frequencies(series, words_nb=10,  ngram=1, min_str_len=1, title="default",
                          plot_reversed=False, palette="Blues"):
    """
        Plot most or least frequent words in the given series

        Parameters
        ----------
        series : pandas series
            Series to extract the words from
        words_nb: int, optional (default value=10)
            Number of words to display
        min_str_len: int, optional (default value=1)
            Minimum string length for a word
        ngram: int, optional (default value=1)
            Number of words to get frequency from.

        Return
        -------
        None
    """
    if title == "default":
        title = "Top {} words".format(words_nb)
    plt.figure(figsize=(10, 10))

    word_dict = get_term_frequencies(series, min_str_len, ngram)
    df_word_freq = pd.DataFrame(
        list(word_dict.items()),
        columns=['word', 'count']
    ).sort_values(by=['count'], ascending=plot_reversed)

    sns.barplot(y="word", x="count",
                data=df_word_freq[:words_nb], palette=palette).set_title(title)


def plot_class_balancing(df, col_text="clean_text", col_label="label",
                         palette="Blues", title="Class balancing", xlabel="", ylabel="",
                         title_fontsize=24, label_fontsize=18, data_fontsize=12):
    """Plot number of documents per label"""

    df_visualisation = pd.DataFrame(columns=[col_text, col_label])
    d_list = []

    # Get labels
    for _, row in df.iterrows():
        for value in str(row[col_label]).split(','):
            d_list.append({'text': row[col_text],
                           'value': value})
    df_visualisation = df_visualisation.append(
        d_list, ignore_index=True, sort=True)
    df_visualisation = df_visualisation.groupby(
        'text',)['value'].value_counts()
    df_visualisation = df_visualisation.unstack(level=-1).fillna(0)

    labels = list(df_visualisation.columns.values)

    sns.set(font_scale=1.5)

    nb_items = df_visualisation.shape[0]
    ax = sns.barplot(
        y=labels, x=df_visualisation.sum().values,
        palette=palette, order=df[col_label].value_counts().index)
    plt.title("{0} ({1} {2})".format(title, str(
        nb_items), "values"), fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    # adding the text labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width+0.2,
                p.get_y()+p.get_height()/1.3,
                '{:1.0f}'.format(width*0.8) + " (" +
                str(round(100*width/nb_items, 1)) + "%)", fontsize=data_fontsize)
