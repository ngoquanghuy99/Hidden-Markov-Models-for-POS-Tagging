import string
# punctuation characters
punct = set(string.punctuation)

# morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

def get_word_tag(line, vocab):
    # check if a line is empty (just contains \n or \t), if yes
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
        return word, tag
    return None

def preprocess(tokens, vocab):
    prep_tokens = []
    for tok in tokens:
        # end of sentence
        if not tok.split():
            tok = "--n--"
            prep_tokens.append(tok)
            continue
        elif tok.strip() not in vocab:
            tok = assign_unk(tok)
            prep_tokens.append(tok)
            continue
        else:
            prep_tokens.append(tok.strip())
    assert(len(prep_tokens) == len(tokens))
    return prep_tokens

def assign_unk(tok):
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"
