# ClassMail

<img src="https://github.com/Fabien-Couthouis/ClassMail/blob/master/classmail_icon.png" alt="icon" width="200"/>
Classmail is a mail classification Python library optimized for french mails in the field of insurance. Classmail was created to automate mail classification workflow in quick experiments. Developped during my internship at [Covéa](https://www.covea.eu).

Classmail provides:

* **Data visualisation:** For quick data analysis, based on matplotlib and seaborn

* **Mails preprocessing (cleaning):** Optimised for inasurrance purposes, with prebuilt regular expressions (in french). This configuration file can be adapted for other languages or purposes.

* **Deep learning model creation (for classification):** Simple interface to build Pytorch models quickly based on [Flair](https://github.com/zalandoresearch/flair) nlp library.

* **Model analysis and explainer** Simple interface with prebuilt seaborn graphs and model explainer based on [Lime](https://github.com/marcotcr/lime).


## Quick Start

### Requirements and Installation

The project is based on Python 3.7+.
If you do not have Python 3.6, install it first. 
Then, in your favorite virtual environment, simply do:

```
pip install classmail
```

### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is make a `Sentence`, load 
a pre-trained model and use it to predict tags for the sentence:


* Data analysis
    ```python
    import classmail.data_visualisation.data_visualisation as dv

    # show class balancing graph
    dv.plot_class_balancing(data,col_text='header_body',col_label="COMPETENCE", title="Catégories des mails")
    #show most frequent bigrams
    dv.plot_word_frequencies(data['message'],ngram=2,words_nb=20)
    #plot a wordcloud with most frequent terms in body
    dv.plot_wordcloud(data['body'])
    ```

    
* Cleaning
    ```python
    from classmail.nlp.cleaning import clean_mail

    #create a new column in data ("clean_text") with preprocessed header and body
    data = clean_mail(data,"body","header")
    ```

    ##### I defined my own conf.json :
    ```python
    from classmail.nlp.configjson import ConfigJsonReader as conf

    conf.set_config_path(file_path='path/to/my/conf.json')
    conf_dict = conf.get_config_file()
    print(conf_dict)  # will print the json.
    ```

* Model creation and training
    ```python
    from classmail.classification.trainer import Trainer
    
    trainer = Trainer()
    #generate train / test / val csv files
    trainer.prepare_data(data, col_text="clean_text",col_label="COMPETENCE", train_size=0.7, val_size=0.15, test_size=0.15)

    #create a new column in data ("clean_text") with preprocessed header and body
    data = clean_mail(data,"body","header")

    #train a model with default parameters
    trainer.train_model(model_name="default_model")
    ```

* Model predictions, evaluation and explaination
    ```python
    from classmail.classification.model import Model

    #load our model, saved in "ressources" folder
    model = Model("ressources\\model_default")
    #predictions
    predictions=model.get_predictions(X_test)
    #confusion matrix
    model.plot_confusion_matrix(pred_labels=predictions, true_labels=y_test)
    #explain one exemple at index 110
    model.visualize_one_ex(X_test,y_test,index=110,num_features=6)
    #compute most discriminants words in each category
    sorted_contributions = model.get_statistical_explanation(X_test, ["class 1","class 2","class 3"] sample_size=15)
    #plot them for first class
    model.plot_discriminant_words(sorted_contributions, "class 1", nb_words=15)
    ```


## Tutorial

Here is a more complete usage exemple for the mail classification task. Data cannot be provided for legislation and privacy matters.

* [Tutorial : General workflow](/Tutorial.ipynb)
