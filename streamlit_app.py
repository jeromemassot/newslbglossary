from utils import rank_context_from_query, generate_answer
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast

from pinecone import Pinecone
import streamlit as st
import openai


########################################################################################################################
## Setup methods

def setup():
    """
    Setup the streamlit app
    :return: None
    """
    if not 'OPENAI_API_KEY' in st.session_state.keys():
        openai.api_key = st.secrets['OPENAI_API_KEY']

    if not 'PINE_API_KEY' in st.session_state.keys():
        st.session_state['PINE_API_KEY'] = st.secrets['PINE_API_KEY']
        st.session_state['INDEX_NAME'] = st.secrets['INDEX_NAME']
        pc = Pinecone(api_key=st.session_state['PINE_API_KEY'])
        index = pc.Index(st.session_state['INDEX_NAME'])
        st.session_state['index'] = index
    

def load_tokenizer():
    """
    Load the tokenizer
    :return: None
    """
    st.session_state['tokenizer'] = BertTokenizerFast.from_pretrained('bert-base-uncased')


def load_model():
    """
    Load the model
    :return: None
    """
    st.session_state['model'] = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


########################################################################################################################
## App Interface

st.title("chatGPT engined Q&A based on the SLB.com technical glossary")

st.warning("Setup and model loading can take a few seconds...")

with st.spinner('Setup and Model loading...'):
    setup()
    if not 'tokenizer' in st.session_state.keys():
        load_tokenizer()
    if not 'model' in st.session_state.keys():
        load_model()

# Query input
query_text = st.text_input(label="Query")

# Search button and open-domain checkbox
search = st.button(label='Find an anwser from SLB glossary')
open_domain = st.checkbox("Authorize open-domain knowledge", value=False)
st.warning("Open-domain knowledge is not always relevant. Please use it with caution..")

if search:
    terms, keywords, context = rank_context_from_query(
        query_text, st.session_state['tokenizer'], st.session_state['model'], st.session_state['index']
    )

    # first try a closed domain answer
    answer = generate_answer(query_text, context)

    if answer.lstrip().rstrip().strip() != "None":
        st.write("I have found information from the SLB glossary and I can propose this answer:\n")
        st.markdown(answer)
        st.multiselect('I have found this anwser from these glossary contents:', options=sorted(terms), default=sorted(terms))
        st.multiselect('You can also be interested by:', options=sorted(keywords), default=sorted(keywords))
    
    # second try an open domain answer if allowed
    elif open_domain:
        answer = generate_answer(query_text, context, True)
        st.write("I have found information outside the SLB glossary. Please use it with caution:\n")
        st.markdown(answer)

    # else indicate that no information has been found in the context
    else:
        st.write("""
            This question is unrelated to the context provided and cannot be 
            answered based on the information given."""
        )