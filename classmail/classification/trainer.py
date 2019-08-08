import os
import pandas as pd
from unidecode import unidecode
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from flair.visual.training_curves import Plotter
from flair.trainers import ModelTrainer
from flair.models import TextClassifier
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings, StackedEmbeddings, FastTextEmbeddings


class Trainer():
    def __init__(self):

        self.column_name_map = {}
        self.model_name = ""
        self.loaded_classifier = None
        self.df = None
        self.text_col = None
        self.label_col = None
        self.data_folder = None
        self.train_size = None
        self.val_size = None
        self.test_size = None

    def prepare_data(self, df, text_col, label_col, data_folder=".\\ressources",
                     train_size=0.7, val_size=0.15, test_size=0.15):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col

        self.data_folder = data_folder

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Clean data to avoid error
        self._drop_na_and_empty()
        self._remove_accents_on_labels()

        # Generate column_name_map
        text_col_idx, label_col_idx = self._get_text_label_col_index()
        self.column_name_map = {
            text_col_idx: "text", label_col_idx: "label_topic"}

        # Generate files
        self._create_train_test_val_csv()

    def _get_text_label_col_index(self):
        # +1 because we want to take in accounnt the index col created
        # with pandas csv export (and we want to keep it)
        text_col_idx = self.df.columns.get_loc(self.text_col) + 1
        label_col_idx = self.df.columns.get_loc(self.label_col) + 1

        return text_col_idx, label_col_idx

    def _remove_accents_on_labels(self):
        self.df.loc[:, self.label_col] = self.df[self.label_col].apply(
            unidecode)

    def _drop_na_and_empty(self):
        old_shape = self.df.shape
        # Df containing False if the row is nan or only spaces
        is_not_na_or_empty = self.df[self.text_col].str.strip().astype(bool)
        # True if na or empty cells were found
        na_or_empty_found = is_not_na_or_empty.isin([False]).any()

        # Delete na and empty cells
        self.df = self.df[is_not_na_or_empty]

        # Show warning if na or empty rows where deleted
        if na_or_empty_found:
            print("Warning : na or empty cell found.",
                  "Corresponding rows have been deleted.")
            print("Old number of rows: ", old_shape[0])
            print("New number of rows: ", self.df.shape[0])

    def _create_train_test_val_csv(self):
        train, test, val = self.train_test_val_split()
        self._create_ressources(train, test, val)
        print("Train, test and val csv files were created in {} !".format(
            self.data_folder))

    def train_test_val_split(self):
        train, remain = train_test_split(
            self.df, test_size=(self.val_size + self.test_size))

        # To preserve new_test_size + new_val_size = 1
        new_test_size = round(
            self.test_size / (self.val_size + self.test_size), 2)
        val, test = train_test_split(remain, test_size=new_test_size)

        return train, test, val

    def _create_ressources(self, train, test, val):
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        self._create_csv(train, "train.csv")
        self._create_csv(test, "test.csv")
        # ! flair naming convention for validation data is dev !
        self._create_csv(val, "dev.csv")

    def _create_csv(self, df, name):
        df.to_csv("{0}\\{1}".format(self.data_folder, name), encoding="utf-8")

    def train_model(self, model_name="text_classification_model", custom_word_embeddings=None,
                    rnn_type="GRU", use_pool_embedding=False, hidden_size=16, reproject_words=True, reproject_words_dimension=128,
                    learning_rate=1e-3, batch_size=8, anneal_factor=0.5, patience=2, max_epochs=30, **kwargs):
        self.model_name = model_name

        corpus = CSVClassificationCorpus(self.data_folder,
                                         self.column_name_map,
                                         skip_header=True)

        label_dict = corpus.make_label_dictionary()

        # Word embedding selection
        if custom_word_embeddings is None:
            word_embeddings = [WordEmbeddings('fr')]
        else:
            word_embeddings = custom_word_embeddings

        # initialize document embedding by passing list of word embeddings and parameters
        if use_pool_embedding:
            document_embeddings = DocumentPoolEmbeddings(word_embeddings,
                                                         pooling='max', fine_tune_mode='nonlinear')
        else:
            document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                        hidden_size=hidden_size,
                                                        reproject_words=reproject_words,
                                                        reproject_words_dimension=reproject_words_dimension,
                                                        rnn_type=rnn_type
                                                        )

        # create the text classifier and initialize trainer
        classifier = TextClassifier(
            document_embeddings, label_dictionary=label_dict)
        trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

        # let's train !
        num_workers = cpu_count()  # - 1
        trainer.train("{0}\\{1}".format(self.data_folder, self.model_name),
                      learning_rate=learning_rate,
                      num_workers=num_workers,
                      mini_batch_size=batch_size,
                      anneal_factor=anneal_factor,
                      patience=patience,
                      max_epochs=max_epochs,
                      **kwargs)
