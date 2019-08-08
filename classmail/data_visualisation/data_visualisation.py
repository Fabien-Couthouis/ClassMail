import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter


def plot_string_length(series):
    max_length = series.str.len().max()
    min_length = series.str.len().min()
    mean = series.str.len().mean()
    median = series.str.len().median()

    print("max_length: ", max_length,
          "\nmin_length: ", min_length,
          "\nmean: ", mean,
          "\nmedian: ", median)


def plot_before_after(series_before, series_after, index=0):
    print("Before : \n", series_before.iloc[index],
          "\n\n______________________________________\n\n",
          "After :\n", series_after.iloc[index])


def plot_wordcloud(series, word_frequencies=None, title="", title_font_size=40,
                   background_color="black", width=2500, height=1800):
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
    text = " ".join(series.values.astype(str))
    regex = re.compile(r"\w+{}".format(r" \w+"*(ngram-1)))
    words = re.findall(regex, text.lower())
    words = [w for w in words if len(w) >= min_str_len]
    word_dict = Counter(words)

    return word_dict


def _get_relevance_scores(df, text_col, label_col, category):
    """Get a relevance score for all terms in the specified category. 
    Adapted from this article https://arxiv.org/ftp/arxiv/papers/1608/1608.07094.pdf"""
    text_series = df[text_col].str.lower()
    label_series = df[label_col]
    # Texts corresponding to the specified category
    this_cat_series = text_series[label_series == category]

    word_freq_dict_this_cat = get_term_frequencies(this_cat_series)
    word_freq_dict_all_cats = get_term_frequencies(text_series)

    # Compute class_weight
    this_cat_size = len(this_cat_series)
    cat_sizes = len(text_series)
    class_weight = this_cat_size / cat_sizes

    scores = {}
    for word in word_freq_dict_this_cat:
        # Compute class_term_weight
        class_frequency = this_cat_series.str.contains(
            word, regex=False, na=False).sum()
        corpus_frequency = text_series.str.contains(
            word, regex=False, na=False).sum()

        class_term_weight = _get_log_normalised_ratio(
            class_frequency, corpus_frequency)

        # Compute class_term_density
        this_cat_freq_normalised = word_freq_dict_this_cat[word]
        all_cat_freq_normalised = word_freq_dict_all_cats[word]
        class_term_density = _get_log_normalised_ratio(
            this_cat_freq_normalised, all_cat_freq_normalised)

        # Compute score
        score = class_weight*class_term_weight*class_term_density

        scores[word] = score

    return scores


def _get_log_normalised_ratio(numerator, denominator):
    normalised_numerator = math.log(numerator)
    normalised_denominator = 1 + math.log(denominator)
    ratio = normalised_numerator/normalised_denominator
    return ratio


def show_relevant_terms(df, text_col, label_col, category, plot=False,
                        nb_words_to_plot=10, plot_reversed=True, palette="Blues", title="default"):
    """Get a dataframe showing most relevant terms for a specified category"""
    if title == "default":
        title = "TF-IDF score for {}".format(category)

    scores = _get_relevance_scores(df, text_col, label_col, category)
    sorted_scores = sorted(scores.items(),
                           key=lambda kv: kv[1], reverse=plot_reversed)
    df_scores = pd.DataFrame(
        sorted_scores,
        columns=['word', 'relevance_score'])
    if plot:
        sns.barplot(y="word", x="relevance_score",
                    data=df_scores[:nb_words_to_plot], palette=palette).set_title(title)
    return df_scores


def plot_word_frequencies(series, words_nb=10, title="default", ngram=1,
                          ascending=False, min_str_len=2, palette="Blues"):
    if title == "default":
        title = "Top {} words".format(words_nb)
    plt.figure(figsize=(10, 10))

    word_dict = get_term_frequencies(series, min_str_len, ngram)
    df_word_freq = pd.DataFrame(
        list(word_dict.items()),
        columns=['word', 'count']
    ).sort_values(by=['count'], ascending=ascending)

    sns.barplot(y="word", x="count",
                data=df_word_freq[:words_nb], palette=palette).set_title(title)


def plot_class_balancing(df, col_text="clean_text", col_label="label",
                         palette="Blues", title="", title_fontsize=24, dataset_len_word="values",
                         xlabel="", ylabel="", label_fontsize=18, data_fontsize=12):
    df_visualisation = pd.DataFrame(columns=[col_text, col_label])
    d_list = []

    # Get categories
    for _, row in df.iterrows():
        for value in str(row[col_label]).split(','):
            d_list.append({'text': row[col_text],
                           'value': value})
    df_visualisation = df_visualisation.append(
        d_list, ignore_index=True, sort=True)
    df_visualisation = df_visualisation.groupby(
        'text',)['value'].value_counts()
    df_visualisation = df_visualisation.unstack(level=-1).fillna(0)

    categories = list(df_visualisation.columns.values)

    sns.set(font_scale=1.5)

    nb_items = df_visualisation.shape[0]
    ax = sns.barplot(
        y=categories, x=df_visualisation.sum().values,
        palette=palette, order=df[col_label].value_counts().index)
    plt.title("{0} ({1} {2})".format(title, str(
        nb_items), dataset_len_word), fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    # adding the text labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width+0.2,
                p.get_y()+p.get_height()/1.3,
                '{:1.0f}'.format(width*0.8) + " (" +
                str(round(100*width/nb_items, 1)) + "%)", fontsize=data_fontsize)
