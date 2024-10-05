from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableConfig
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import chainlit as cl
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi")
api_key=os.getenv('OPENAI_API_KEY')
db_url=os.getenv('DB_URL')
client = MongoClient(db_url, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    db = client["freelancing"]
    collection = db["poc_1"]
    documents = collection.find()
    instruct = ""
    for ind,document in enumerate(documents):
        logger.info(document["Instruction"])
        print(document["Instruction"])
        instruct+=f"""{ind+1} - {document["Instruction"]}\n""" 
        # instruct = " "
    print(instruct)
except Exception as e:
    print(e)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)


conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                               max_len=1000,
                                               return_messages=True,
                                                   )


ice_cream_assistant_template = """
You are an assistant"
Chat History: {chat_history}
Question: {question}
Answer:"""

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=ice_cream_assistant_template
)


@cl.on_chat_start
async def quey_llm():
    client = MongoClient(db_url, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        db = client["freelancing"]
        collection = db["poc_1"]
        documents = collection.find()
        instruct = ""
        for ind,document in enumerate(documents):
            logger.info(document["Instruction"])
            print(document["Instruction"])
            instruct+=f"""{ind+1} - {document["Instruction"]}\n"""
            # instruct = " "
        print(instruct)
    except Exception as e:
        print(e)
    files = None



    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["PDF"],
            max_size_mb=20,
            timeout=180,
        ).send()


    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()


    loader = PyPDFLoader(file.path)

    pages = loader.load_and_split()

    texts = " ".join([page.page_content for page in pages])
    
    llm = ChatOpenAI(model='gpt-3.5-turbo',
                 temperature=1,openai_api_key=api_key,streaming=True)
    
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=1000,
                                                   return_messages=True,
                                                   )
    
    ice_cream_assistant_template =f""""
    You are an auditor and you audit the below document. Analyse the WO point wise and mention these points before analysing any WO.
    1.	WO Number 
    2.	Aircraft Number
    3.	Sequence Number
    4.	Work Package Number
    5.	Type of WO
    6.	Planning Date 

    Also follow the instructions given below:
    {instruct}
    I am pasting the WO below:\n Work Order(WO): {texts}""" +"""\n
            Chat History: {chat_history}
            Question: {question}
            Answer:"""
            
    print(ice_cream_assistant_template)
    ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=ice_cream_assistant_template
)
    llm_chain = LLMChain(llm=llm, 
                         prompt=ice_cream_assistant_prompt_template,
                         memory=conversation_memory)
    msg.content = f"The PDF `{file.name}` is uploaded successfully."
    await msg.send()
    cl.user_session.set("llm_chain", llm_chain)
    
@cl.on_message
async def query_llm(message: cl.Message):
    
    llm_chain = cl.user_session.get("llm_chain")
   
    msg = cl.Message(content="")
    if 'Instruction=>' in message.content:
        doc = {"Instruction": message.content.split('>')[1]}
        inserted_id = collection.insert_one(doc).inserted_id

    

    async for chunk in llm_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):

        await msg.stream_token(chunk['text'])
    

    await msg.send()    
