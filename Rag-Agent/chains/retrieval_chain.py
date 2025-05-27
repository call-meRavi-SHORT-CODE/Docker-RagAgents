from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from utils.prompt_template import get_prompt, get_parser

def build_retrieval_chain(llm, retriever):
    prompt = get_prompt()
    parser = get_parser()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
        output_parser=parser,
    )

    return create_retrieval_chain(retriever, document_chain)