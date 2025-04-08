import nltk # Import Natural Language Toolkit which provides functions for tokenizing (spliting text), tagging, parsing etc.
import string # Get access to standard punctuations like . , !. so we can remove them.

nltk.download('punkt') # Dowmloads the punkt tokenizers module which is neccessary of sentence and word tokeniztion

nltk.download('punkt')         # regular punkt tokenizer
nltk.download('punkt_tab')     # (edge case) in case it's being referenced

from nltk.tokenize import word_tokenize # Import word_tokenize function from word_tokenize

def preprocess_text(text): # Defines the function preprocess_text that takes text as input
    """
    Clean and tokenizes the input text
    Steps:
    1. Lowercase conversion
    2. Punctuation removal
    3. Word tokenization
    """

    text = text.lower() # Converts the text to the lowercase
    text = text.translate(str.maketrans("",'',string.punctuation)) # Removes all the punctuations like !, ?, . etc
    tokens = word_tokenize(text) # Tokenizes (splits) the cleaned words to list of words
    return tokens # Returns the list of token words

# Test block runs only if file is executed directly
if __name__ == '__main__':
    sample = "How to apply for admissions??"
    print("Orignal: ",sample)
    print("Preproccesed: ",preprocess_text(sample))
