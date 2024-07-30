import streamlit as st
from rag import *
import argparse
import sys

def create_prompt(file):
    with open(file) as f:
        test_prompt_template = f.read()
    test_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=test_prompt_template
    )
    return test_prompt


def load_app(model_name,directory,version,prompt_file,load_from_disk):
    # Load the summarization pipeline
    @st.cache_resource()
    def load_summarizer():
        prompt = create_prompt(prompt_file)
        print(f"""
        RAG ARGS
        model : {model_name}
        directory : {directory}
        version : {version}
        prompt: {prompt}
        load:   {load_from_disk}
        """)
        return RAGT5(
                        model_name
                        , directory
                        , version
                        , prompt
                        , load_db_from_disk=load_from_disk
                    )

    summarizer = load_summarizer()

    # Streamlit app
    st.title("Gutenberg Catalog")


    st.write('You can add a command for the LLM to generate the summarization of ebooks from the gutenberg catalog\nExample: Summarize the book about the strand magazine, vol. 05, issue 25')
    # Text input
    user_input = st.text_area("Enter the text to summarize:", height=200)


    # Summarize button
    if st.button("Summarize"):
        if user_input:
            with st.spinner("Summarizing..."):
                summary = summarizer.query_rag(user_input)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.error("Please, enter some text to summarize")

    st.sidebar.header("About")
    st.sidebar.write("""
    This app uses a pre-trained model from Hugging Face to summarize ebooks from the gutenberg website.
    """)
    st.info('Not for Deployment', icon="ℹ️")
    st.info('Need Refinament for long-text summarization problems', icon="ℹ️")

def get_arguments():
    args = {
    'model':sys.argv[1],
    'directory':sys.argv[2],
    'version':sys.argv[3],
    'prompt':sys.argv[4],
    'load_from_disk':bool(sys.argv[5])
    }
    return args

def main():
    args = get_arguments()
    model_name      = args['model']
    directory       = args['directory']
    version         = args['version']
    prompt_file     = args['prompt']
    load_from_disk  = args['load_from_disk']

    load_app(model_name,directory,version,prompt_file,load_from_disk)
    
if __name__ == '__main__':
    main()



