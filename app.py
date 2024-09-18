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

api_key=os.getenv('OPENAI_API_KEY')

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
    
    ice_cream_assistant_template =f"""
    You are an auditor and you audit the below pasted WO content. First fetch the below points form the WO.
    1.	WO Number 
    2.	Aircraft Number
    3.	Sequence Number
    4.	Work Package Number
    5.	Type of WO
    6.	Planning Date 

    Secondly go through the WO and give a step-by-step Wo analysis.
    Always Give answer in mardown format.
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
    

    async for chunk in llm_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        print(chunk)
        await msg.stream_token(chunk['text'])
    

    await msg.send()    
