from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

urls = [
    "https://docs.pfm.genify.ai/pfm-suite/v1/transaction-data-api",
    "https://docs.pfm.genify.ai/pfm-suite/v1/restricted-access/transaction-analytics-api",
    "https://docs.pfm.genify.ai/pfm-suite/v1/restricted-access/user-engagement-api",
    "https://docs.pfm.genify.ai/pfm-suite/v1/restricted-access/userbase-analytics-api",
    "https://docs.pfm.genify.ai/pfm-suite/v1/restricted-access/recommendation-api",
    "https://docs.pfm.genify.ai/pfm-suite/v1/website-analytics/logo-grabber",
    "https://docs.pfm.genify.ai/pfm-suite/v1/website-analytics/url-classifier",
    "https://docs.pfm.genify.ai/error-codes/http-codes"
]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
#Insert your API Keys
OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchain"
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
query = "What is the header for the Enrich Transaction Data api?"
docs = docsearch.similarity_search(query, include_metadata=True)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


# Define a sentinel value to exit the loop
EXIT = "exit"

# Start the loop
while True:
    # Prompt the user to input the query
    query = input("Please enter your question, or type 'exit' to quit: ")

    # Check if the user wants to exit
    if query.lower() == EXIT:
        print("Exiting...")
        break

    # Perform the similarity search and include metadata
    docs = docsearch.similarity_search(query, include_metadata=True)

    # Run the pipeline with the input documents and question
    output = chain.run(input_documents=docs, question=query)

    # Print the generated answer
    print(output)