from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
#from langchain.chains.combine_documents.base import MapReduceDocumentsChain
from operator import itemgetter

# Load the Llama2 model
model = Ollama(model="llama2", temperature=0.3)
embeddings = OllamaEmbeddings()

# Load FAQs data from the PDF
def load_faq_data(pdf_path="Virtual_Companion_FAQs.pdf"):
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

# Create the LangChain pipeline
def create_chain():
    pages = load_faq_data()
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Prompt template
    template = """
    Answer the question based on the context below in a consice and precise manner in maximum 150 words. If you can't answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    # Define the chain
    chain = (

    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
    )   

    return chain
