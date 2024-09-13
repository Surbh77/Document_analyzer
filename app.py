import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from chainlit.input_widget import TextInput
import chainlit as cl

api_key=os.getenv('OPENAI_API_KEY')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


@cl.on_chat_start
async def on_chat_start():
    
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["PDF"],
            max_size_mb=20,
            timeout=180,
        ).send()

    # print("......+++++++++",files)
    file = files[0]
    # print(f"Chainlit PDF{file.path}")
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()


    loader = PyPDFLoader(file.path)

    pages = loader.load_and_split()
    # print("Pages=====>>>>> ",pages)
    texts = " ".join([page.page_content for page in pages])
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    doc_msg = ("human",f"Document: {texts}")
    model = ChatOpenAI(streaming=True,openai_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an aircraft maintainance audit expert. An you audit the given document",
            ),
            doc_msg,
            ("human","{question}"),
        ]
    )
    
    # print("Prompt====>>>>",prompt)
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    msg.content = f"The PDF `{file.name}` is uploaded successfully."
    await msg.update()

from langchain.schema.runnable.config import RunnableConfig

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
