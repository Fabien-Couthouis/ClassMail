import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import confusion_matrix
from flair.data import Sentence
from flair.visual.training_curves import Plotter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def show_string_length(series):
    max_length = series.str.len().max()
    min_length = series.str.len().min()
    mean = series.str.len().mean()
    median = series.str.len().median()

    print("max_length: ", max_length,
          "\nmin_length: ", min_length,
          "\nmean: ", mean,
          "\nmedian: ", median)


def show_before_after(series_before, series_after, index=0):
    print("Before : \n", series_before.iloc[index],
          "\n\n______________________________________\n\n",
          "After :\n", series_after.iloc[index])


def show_wordcloud(df, col, word_frequencies=None, title="", title_font_size=40, background_color="black", width=2500, height=1800):
    texts = df[col].values

    cloud = WordCloud(
        background_color=background_color,
        collocations=False,
        width=width,
        height=height
    )

    if word_frequencies is None:
        word_frequencies = get_word_frequencies(df, col)

    cloud.generate_from_frequencies(word_frequencies)

    plt.axis('off')
    plt.title(title, fontsize=title_font_size)
    plt.imshow(cloud)
    plt.show()


def get_word_frequencies(df, col):
    # Note : do it better to improve perfs
    texts = " ".join(df[col].values)
    # sentences = Sentence(texts, use_tokenizer=True)

    # return sentences.tokens

    cloud = WordCloud()
    word_frequencies = cloud.process_text(texts)

    return word_frequencies


def show_class_balancing(df, col_text="clean_text", col_label="label",
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


def show_confusion_matrix(pred_labels, true_labels, pred_labels_axename="Predicted label",
                          true_labels_axename="True label", inverse_axis=False, title="", cmap="YlGnBu"):
    label_names = np.unique(true_labels)

    conf_mat = confusion_matrix(
        true_labels, pred_labels, labels=label_names)
    conf_mat_normalized = conf_mat.astype(
        'float') / conf_mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_mat_normalized, xticklabels=label_names,
                yticklabels=label_names, cmap=cmap)

    if inverse_axis:
        x_label = pred_labels_axename
        ylabel = true_labels_axename
    else:
        x_label = true_labels_axename
        ylabel = pred_labels_axename

    plt.xlabel(x_label)
    plt.ylabel(ylabel)

    plt.title(title)
    plt.show()


def plot_training_curves(model_path):
    plotter = Plotter()
    plotter.plot_training_curves("{}/loss.tsv".format(model_path))
    plotter.plot_weights("{}/weights.txt".format(model_path))


def get_metrics(pred_labels, true_labels, average="weighted"):
    print("Accuracy:", accuracy_score(pred_labels, true_labels))
    print("F1-score:", f1_score(pred_labels, true_labels, average=average))
    print("Precision:", precision_score(
        pred_labels, true_labels, average=average))
    print("Recall:", recall_score(pred_labels, true_labels, average=average))


def generate_preds_true_df(text, pred_labels, true_labels, confidence=None, export_to=None, ):
    df = pd.DataFrame(
        {"Text": text, "Prediction": pred_labels, "True label": true_labels})

    df.index.name = '#'
    df["Good prediction"] = (df['Prediction'] == df['True label'])

    if confidence is not None:
        df["Prediction confidence"] = confidence

    if export_to is not None:
        df.to_csv(export_to, sep=";", encoding="utf-8")

    return df
