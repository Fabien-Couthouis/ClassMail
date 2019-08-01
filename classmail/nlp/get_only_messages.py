import re
from classmail.nlp.configjson import ConfigJsonReader

conf_reader = ConfigJsonReader()
config = conf_reader.get_config_file()

REGEX = config['regex']

transition_list = REGEX['build_historic']['transition_list']
regex_transitions = re.compile(
    '|'.join(transition_list), flags=re.M)

segmenting_dict = REGEX['mail_segmenting']['segmenting_dict']
regex_seg_list = [regex for sublist in segmenting_dict.values()
                  for regex in sublist]
regex_seg = re.compile('|'.join(regex_seg_list), flags=re.M)


attached_file_str = REGEX['mail_segmenting']["attached_file"]
regex_attached_file = re.compile(attached_file_str)


def get_only_messages(email_body):
    """Return a string containing only the messages from the conversation
    (remove transitions, greetings, signatures ..."""
    end = len(email_body)

    indexes_transition = _get_indexes_by_regex(
        email_body, regex_transitions) + [(end, end)]
    indexes_seg = _get_indexes_by_regex(
        email_body, regex_seg)
    attached_files = _get_attached_files_names(email_body)

    indexes = _regroup_indexes(indexes_transition, indexes_seg)
    indexes = _filter_overlap(indexes)
    email_body_cleaned = _remove_by_index_list(email_body, indexes)

    # Add attached_files info at the end
    message = email_body_cleaned + " " + " ".join(attached_files)

    return message


def _get_indexes_by_regex(text, regex):
    """Return list of indexes defining the transitions between
    different messages in an email."""
    indexes = []

    for match in regex.finditer(text):
        idx = (match.start(), match.end())
        indexes.append(idx)

    # Remove duplicates and sort list
    indexes = list(set(indexes))
    indexes = sorted(indexes, key=lambda tup: tup[0])
    indexes = _filter_overlap(indexes)

    return indexes


def _filter_overlap(index):
    """Filter indexes in list if they overlap."""
    if len(index) < 2:
        return index
    index_f = []
    i = 0
    j = i + 1
    while j < len(index):
        if index[i][1] > index[j][0]:
            index[i] = (min(index[i][0], index[j][1]),
                        max(index[i][0], index[j][1]))
            j += 1
        else:
            index_f += [index[i]]
            i = j
            j += 1
    index_f += [index[i]]

    return index_f[:i + 1]


def _regroup_indexes(idx_transition, idx_seg):
    """Regroup transitions and segmentations lists (regroup the last
    part to supress signatures etc)"""
    if len(idx_seg) == 0:
        return idx_transition

    indexes = []
    for i_tran in idx_transition:
        last_i_seg = (0, 0)
        for i_seg in idx_seg:
            if i_seg[1] > i_tran[0]:
                break
            else:
                last_i_seg = i_seg
                indexes.append(i_seg)

            last_message_index = (last_i_seg[0], i_tran[1])
            if last_message_index != i_seg:
                indexes.append(last_message_index)

    return indexes


def _remove_by_index_list(text, index_list):
    """Remove all substrings inside a string, thanks to the given list of indexes """
    len_removed = 0
    for i in index_list:
        i_start = i[0] - len_removed
        i_end = i[1]-len_removed
        text = text[:i_start] + text[i_end:]
        len_removed += i[1]-i[0]
    return text


def _get_attached_files_names(text):
    """Get all filenames from attached files in the text (supress file extension)."""
    f_names = []

    for match in regex_attached_file.finditer(text):
        # Filename is the third capture group in the regex
        raw_f_name = match[3]
        f_name_without_digit = ''.join(
            char for char in raw_f_name if not char.isdigit())

        flagged_f_name = "flagattachedfile " + f_name_without_digit
        f_names.append(flagged_f_name)

    return f_names
