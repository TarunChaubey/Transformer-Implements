
def self_attention(text):
    import numpy as np

    # Define the tokens
    #tokens = ['how', 'are', 'you', 'doing', 'today']
    tokens = text.split()

    # Define the query vector
    query = np.random.randn(len(tokens))

    # Define the matrix of keys
    keys = np.random.randn(len(tokens), len(tokens))

    # Define the matrix of values
    values = np.random.randn(len(tokens), len(tokens))

    # Calculate the attention scores
    scores = np.dot(query, keys.T)

    # Apply softmax to the scores to get the attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))

    # Calculate the attention value
    attention_value = np.dot(attention_weights, values)

    # Print the attention weights and attention value
    return {"Attention weights":attention_weights,"Attention value ": attention_value}

def self_attention(text):
    import numpy as np

    # Define the tokens
    #tokens = ['how', 'are', 'you', 'doing', 'today']
    tokens = text.split()

    # Define the query vector
    query = np.random.randn(len(tokens),len(tokens))

    # Define the matrix of keys
    keys = np.random.randn(len(tokens), len(tokens))

    # Define the matrix of values
    values = np.random.randn(len(tokens), len(tokens))

    # Calculate the attention scores
    scores = np.dot(query, keys.T)

    # Apply softmax to the scores to get the attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))

    # Calculate the attention value
    attention_value = np.dot(attention_weights, values)

    # Print the attention weights and attention value
    return {"Attention weights":attention_weights,"Attention value ": attention_value}
