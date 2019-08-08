import re
from unidecode import unidecode
from classmail.nlp.get_only_messages import get_only_messages
from classmail.nlp.configjson import ConfigJsonReader
from classmail.nlp.df_multiprocessing import apply_parallel

conf_reader = ConfigJsonReader()
# conf_reader.set_config_path("G:\classMail\nlp\conf.json")
config = conf_reader.get_config_file()

REGEX_CLEAN = config["regex"]["cleaning"]

stopwords_list = config["words_list"]["stopwords_fr"]
regex_stopwords = re.compile(r'\b(?:{})\b\s'.format('|'.join(stopwords_list)))

clean_duplicates_str = REGEX_CLEAN["clean_duplicates"]
regex_clean_duplicates = re.compile(clean_duplicates_str, flags=re.M)

flags_dict = config["regex"]["flagging"]["flags_dict"]
regex_flags_dict = {re.compile(
    regex.lower()): value for regex, value in flags_dict.items()}

clean_header_list = REGEX_CLEAN["clean_header_list"]
regex_clean_header = re.compile(r'(?:{})'.format(
    '|'.join(clean_header_list).lower()), flags=re.M)

clean_multiple_spaces_list = REGEX_CLEAN["clean_multiple_spaces_list"]
regex_clean_multiple_spaces = re.compile('|'.join(clean_multiple_spaces_list))

clean_body_list = REGEX_CLEAN["clean_body_list"]
regex_clean_body = re.compile(r'(?:{})'.format(
    '|'.join(clean_body_list).lower()), flags=re.M)


def clean_mail(df, col_body, col_header, multithreading=True):
    df['cleaned_body'] = apply_parallel(df, clean_body, args=[col_body])
    df['cleaned_header'] = apply_parallel(df, clean_header, args=[col_header])
    df["clean_text"] = df['cleaned_header'] + " " + df['cleaned_body']
    return df


def clean_body(row, col="body", **kwargs):
    """Clean body column. The cleaning involves the following operations:
        - Cleaning the text
        - Removing the multiple spaces
        - Flagging specific items (postal code, phone number, date...)

    Parameters
    ----------
    row : row of pandas.Dataframe object,
        Data contains 'last_body' column.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.
        flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.
    accents: boolean, optional
        True if you want to clean accents, False if not.
        Default value, True.
    stopwords: boolean, optional
        True if you want to clean stopwords, False if not.
        Default value, True.
    line_break: boolean, optional
        True if you want to clean line_break, False if not.
        Default value, True.
    ponctuation: boolean, optional
        True if you want to clean all non-alphanumeric characters (except spaces), False if not.
        Default value, True.
    duplicates: boolean, optional
        True if you want to clean duplicates (same expression repeated), False if not.
        Default value, True.

    Returns
    -------
    row of pandas.DataFrame object or pandas.Series if apply to all DF.
    """

    cleaned_body = str(row[col])
    cleaned_body = text_to_lowercase(cleaned_body)
    cleaned_body = get_only_messages(cleaned_body)
    cleaned_body = _clean_on_regex(cleaned_body, regex_clean_body)
    cleaned_body = clean_text(cleaned_body, **kwargs)

    return cleaned_body


def clean_header(row, col="header", **kwargs):
    """Clean the header column. The cleaning involves the following operations:
        - Removing the transfers and answers indicators
        - Cleaning the text
        - Flagging specific items (postal code, phone number, date...)

    Parameters
    ----------
    row : row of pandas.Dataframe object,
        Data contains 'header' column.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.
    accents: boolean, optional
        True if you want to clean accents, False if not.
        Default value, True.
    stopwords: boolean, optional
        True if you want to clean stopwords, False if not.
        Default value, True.
    line_break: boolean, optional
        True if you want to clean line_break, False if not.
        Default value, True.
    ponctuation: boolean, optional
        True if you want to clean all non-alphanumeric characters (except spaces), False if not.
        Default value, True.
    digits: boolean, optional
        True if you want to clean all digits, False if not.
        Default value, True.
    duplicates: boolean, optional
        True if you want to clean duplicates (same expression repeated), False if not.
        Default value, True.

    Returns
    -------
    row of pd.DataFrame object or pandas.Series if apply to all DF.
    """
    cleaned_header = str(row[col])

    cleaned_header = text_to_lowercase(cleaned_header)

    if cleaned_header == 'nan':
        return ""
    cleaned_header = _clean_on_regex(cleaned_header, regex_clean_header)
    cleaned_header = clean_text(cleaned_header, **kwargs)
    return cleaned_header


def clean_text(text, **kwargs):
    """Clean a string. The cleaning involves the following operations:
        - Flag relevant items
        - Remove all the accents
        - Remove stopwords
        - Remove all line breaks
        - Remove all symbols and punctuations
        - Remove the multiple spaces

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    cleaned_text = text
    cleaned_text = clean_line_break(cleaned_text, clean_line_break)
    cleaned_text = flag_items(cleaned_text, _is_activated(kwargs, 'flags'))
    cleaned_text = clean_accents(
        cleaned_text, _is_activated(kwargs, 'accents'))
    cleaned_text = clean_ponctuation(
        cleaned_text, _is_activated(kwargs, 'ponctuation'))
    cleaned_text = clean_stopwords(
        cleaned_text, _is_activated(kwargs, 'stopwords'))
    cleaned_text = clean_digits(
        cleaned_text, _is_activated(kwargs, 'ponctuation'))
    cleaned_text = clean_duplicates(
        cleaned_text, _is_activated(kwargs, 'duplicates'))
    cleaned_text = clean_multiple_spaces_and_strip_text(
        cleaned_text)
    return cleaned_text


def text_to_lowercase(text, activated=True):
    """Set all letters to lowercase"""
    if activated:
        text = text.lower()
    return text


def clean_accents(text, activated=True):
    """Remove accents from text"""
    if activated:
        text = unidecode(text).replace("ndeg", "")
    return text


def clean_line_break(text, activated=True):
    """Remove line breaks and tabs from text

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str
    """
    if activated:
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text


def clean_ponctuation(text, activated=True):
    """Remove non alphanumeric characters from text (except spaces)

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str
    """
    if activated:
        text = ''.join(c if c.isalnum() else " " for c in text)
    return text


def clean_duplicates(text, activated=True):
    """Remove duplicated words/flags, ...

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str
    """
    if activated:
        return re.sub(regex_clean_duplicates, r"\1", text)
    return text


def clean_multiple_spaces_and_strip_text(text):
    """Remove multiple spaces, strip text, and clean '-', '*' characters.

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str
    """
    text = re.sub(regex_clean_multiple_spaces, ' ', text)
    text = text.strip()
    return text


def clean_stopwords(text, activated=True):
    for stopword in stopwords_list:
        text = text.replace(" {0} ".format(stopword), ' ')
    return text


def flag_items(text, activated=True):
    """Flag relevant information
        ex : amount, phone number, email address, postal code (5 digits)..
        (Note : Attached files are already flagged in get_only_messages)

    Parameters
    ----------
    text : str,
        Body content.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.

    Returns
    -------
    str

    """

    if activated:
        for regex, value in regex_flags_dict.items():
            text = re.sub(regex, value, text)

    return text


def clean_digits(text, activated=True):
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def _clean_on_regex(text, compiled_regex, activated=True):
    """Remove expressions corresponding to the given regex in a text"""
    if activated:
        text = re.sub(compiled_regex, "", text)
    return text


def _is_activated(kwargs, key):
    return kwargs.get(key, True)
