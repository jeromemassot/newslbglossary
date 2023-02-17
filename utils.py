from collections import Counter
import streamlit as st
import requests
import openai


def hybrid_score_norm(dense, sparse, alpha: float):
    """
    Hybrid score using a convex combination
    alpha * dense + (1 - alpha) * sparse
    :param dense: Array of floats representing long text
    :param sparse: sparse array representing short text
    :param alpha: scale between 0 and 1
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    return [v * alpha for v in dense], {k: v * (1 - alpha) for k, v in sparse.items()}


def generate_sparse_query(query_text, tokenizer)                     :
    """
    Create the sparse embedding for the given query
    :param query_text: query text
    :param tokenizer: tokenizer
    :return: sparse embedding as dictonaries
    """

    input_ids = tokenizer(
        query_text,
        padding=True,
        truncation=True,
        max_length=512
    )['input_ids']

    d = dict(Counter(input_ids))

    sparse_embeddings = {k: v for k, v in d.items() if k not in [101, 102, 103, 0]}

    return sparse_embeddings


def rank_context_from_query(query_text:str, tokenizer, model):
    """
    Rank the contexts needed to answer the query
    :param query_text: input query in plain human language
    :param tokenizer: tokenizer for tokernizing the query text
    :param model: language model for embedding the query text
    :return: terms, keywords and context as set, set and string objects
    """

    sparse_vector = generate_sparse_query(query_text, tokenizer)
    dense_vector = model.encode([query_text], normalize_embeddings=True).tolist()[0]

    # Re-weight queries vectors for hybrid search
    hdense, hsparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=1)

    query = {
        "topK": 5,
        "vector": hdense,
        "sparseVector": hsparse,
        "includeMetadata": True
    }

    headers = {"Api-Key": st.session_state['index_api_key']}

    index_url = 'https://hybrid-slb-glossary-057f5e2.svc.us-west1-gcp.pinecone.io'
    resp = requests.post(index_url + '/query', json=query, headers=headers)

    # placeholders for the returned data
    ranked_context = ''
    terms = list()
    keywords = list()

    for match_vector in resp.json()['matches']:
        terms.append(match_vector['metadata']['term'])
        keywords.extend(match_vector['metadata']['keywords'])
        ranked_context += match_vector['metadata']['text']
        ranked_context += ' '

    return set(terms), set(keywords), ranked_context


def generate_answer(query_text:str, context:str, open_domain:bool=False):
    """
    :param query_text: input query in plain human language
    :param context: text context
    :param open_domain: authorize open-domain knowledge if True
    :return: answer text
    """

    short_context = ' '.join(context[:3500])
    if open_domain:
        prompt = f"""
            Answer the question based on the context below. Indicate the confidence about the answer.
        
            Context: {short_context}

            Question: {query_text}

            Answer:
        """
    else:
        prompt = f"""
            Answer the question based on the context below. If the question cannot be answered using
            the information provided  answer 'None'.
        
            Context: {short_context}

            Question: {query_text}

            Answer:
        """

    # openai API query
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text'].replace('\n', '')
