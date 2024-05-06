import os
from dotenv import load_dotenv
load_dotenv()

# setup downloader
from sec_edgar_downloader import Downloader
dl = Downloader("GaTech","ks@gatech.edu")

# Download 10k filings
def download_filings(ticker):
    dl.get("10-K", ticker, before="2024-01-01", after="1995-01-01", download_details=True)
    
# Parse the filing
from bs4 import BeautifulSoup
import re # regular expressions

'''
Extract all text from the filing

Parameters:
    filing (str): The path to the filing to extract from
Returns:
    str: The text from the filing
'''
def extract_text(filing):
    text = ""
    token = "</DOCUMENT>"
    
    try:
        with open(filing, "r") as file:
            for line in file:
                text += line
                if token in text:
                    break
    except:
        print("Error reading file")

    parser = BeautifulSoup(text, "html.parser")
    
    excluded_tags = ['img', 'script', 'style', 'link', 'object', 'form', 'button', 'td','embed','iframe']
    
    for tag in excluded_tags:
        for element in parser.find_all(tag):
            element.decompose()
    
    text = "".join(i.string.strip() for i in parser.find_all(string=True) if i.string)
    return text



'''
Creates a dictionary matching year to filepath for the ticker

Parameters:
  Ticker (str): The ticker to create the dictionary for
Returns:
  dict: A dictionary matching year to the path of the filing
'''
def create_year_dict(ticker):
  # Creating map from year to filing
  year_to_filing = {}
  directory = f"./sec-edgar-filings/{ticker}/10-K/"

  for file in os.listdir(directory):
    filing_folder = os.path.join(directory, file)
    full_path = os.path.join(filing_folder,'full-submission.txt')
    year = int(file.split('-')[1])
    if (year<=24):
      year+=2000
    else:
      year+=1900
    year_to_filing[year]=full_path
  
  return year_to_filing


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, tool

chat = ChatOpenAI(temperature=0)

from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()
langchain.chat_cache = InMemoryCache()

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever


'''
Exract the years needed from the prompt


Parameters:
    text (str): The text to extract the years from
Returns:
    list: The years extracted
'''
def extract_years(text):
    years = re.findall(r'\b\d{4}\b', text)
    return years

# Embedding the 10k filings

CHUNK_SIZE = 200
MAX_DOCUMENTS = 20

embedding_function = OpenAIEmbeddings()

'''
Breaks down and embeds the filings using Chroma, with a max of 20 years being filed

Parameters:
    ticker (str): The ticker to embed the filings for
    years (list): The years to embed the filings for
'''
def embed_filings(ticker, years):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)
    
    year_dict = create_year_dict(ticker)
    
    for year in years:
        year = int(year)
        filepath = year_dict[year]
        clean_text = extract_text(filepath)
        doc = Document(page_content=clean_text,metadata={"source":"local"})

        docs = text_splitter.split_documents([doc])
        db = Chroma.from_documents(docs, embedding_function, persist_directory=f'./embeddings/{ticker}/{year}')
        print(f"Completed embedding of filings for {ticker}:{year}")
    

class Insight(BaseModel):
  title: str = Field()
  datapoints: List[float] = Field()
  years: List[str] = Field()
  unit: str = Field()

class ModelOutput(BaseModel):

  insights: List[Insight] = Field()
  summary: str = Field()

  class Config:
    arbitrary_types_allowed = True

parser = PydanticOutputParser(pydantic_object=ModelOutput)

'''
Generates the required prompt for the model from the input prompt, context, system prompt and format

Parameters:
    ticker (str): The ticker to generate the prompt for
    input_prompt (str): The input prompt to generate the prompt for
Returns:
    request: The generated request

'''
def generate_request(ticker,input_prompt=None):

  extracted_years = extract_years(input_prompt)
  docs_per_year = int(MAX_DOCUMENTS/len(extracted_years))

  relevant_docs = {}

  for year in extracted_years:
    year=int(year)
    db = Chroma(persist_directory=f'./embeddings/{ticker}/{year}', embedding_function=embedding_function) 
    retriever = db.as_retriever()

    search_kwargs = {"k":docs_per_year}
    picked = retriever.get_relevant_documents(input_prompt,search_kwargs=search_kwargs) 

    combined_context = ''.join([pick.page_content for pick in picked])
    relevant_docs[year] = combined_context

  system_template = """
      You are a meticulous and insightful financial analyst specializing in dissecting SEC 10-K filings. Your expertise lies in extracting key figures and valuable insights from these annual reports, focusing on accuracy and providing comprehensive information.  

      For each metric or insight you identify, ensure you clearly report the following:

      * **Value:** The precise numerical value of the metric. 
      * **Sign:** Indicate whether the value is positive or negative (e.g., profit vs. loss).
      * **Unit:** Specify the unit of measurement (e.g., dollars, percentage, ratio).
      * **Context:** Briefly explain the significance of the metric within the company's financial performance or position. 

      By adhering to these guidelines, you will provide a clear, concise, and informative analysis of the company's financial health based on its 10-K filing.
  """

  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  prompt_template = input_prompt + "\n"
  for year in relevant_docs.keys():
    prompt_template += f"Content from {year}:{relevant_docs[year]}\n\n"

  prompt_template+="{instructions}"

  human_prompt = HumanMessagePromptTemplate.from_template(prompt_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_prompt,
                                                  human_prompt])
  
  request = chat_prompt.format_prompt(instructions=parser.get_format_instructions())

  print(f"Generated prompts")

  return request



'''
Sends prompt to the model and returns the response

Parameters:
    request: The request to send to the model
Returns:
    response: The response from the model
'''
def get_response(request):
  response = chat.invoke(request)
  response_object = parser.parse(response.content)
  
  return response_object
  
  
  
import matplotlib.pyplot as plt

'''
Returns output plots from the response onbject

Parameters:
    response_object: The response object to generate the plots for
Returns:
    fig: The plot generated
'''
def pyplotter(response_object):
    rows = int((len(response_object.insights) + 1)/2)
    cols = min(2, len(response_object.insights))  
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for i, insight in enumerate(response_object.insights):
        if not axes.flat[i] == None:
          plot_axes = axes.flat[i]
        plot_title = str(insight.title)
        units = str(insight.unit)
        datapoints = insight.datapoints
        years = insight.years

        plot_axes.bar(years, datapoints) 
        
        plot_axes.plot(years, datapoints, marker='o', color='green')
        
        for year, datapoint in zip(years,datapoints):
            plot_axes.text(year, datapoint, f"{datapoint} {units}", ha='center', va='bottom', fontsize=8)
        plot_axes.set_xlabel("Year")
        plot_axes.set_ylabel(f"{plot_title} ({units})")
        plot_axes.set_title(f"{plot_title} in {', '.join(years)}")
        plot_axes.grid(True)

    fig.suptitle(f"Insights ({len(response_object.insights)} total)", fontsize=14) 
    
    plt.tight_layout()

    return fig



cached_tickers = []

'''
Main method to be run with every prompt and ticker

Parameters:
  ticker: Company Ticker
  prompt: Insight you want to generate (defaults to earning-based insight)

Returns:
  (plots: plt object of generated plots,
   summary: Textual summary of insights)
'''
def pipeline(ticker, prompt=None):

  if not prompt:
    prompt = "Tell me about the changes in revenue and other key metrics over the years 1996 to 1997."

  if (ticker not in cached_tickers):
    cached_tickers.append(ticker)
    download_filings(ticker)


  embed_filings(ticker,extract_years(prompt))

  request = generate_request(ticker,prompt)
  result_object = get_response(request)
  plots = pyplotter(result_object)
  summary = str(result_object.summary)+"\n"

  return plots,summary



import gradio as gr

demo = gr.Blocks()

with demo:
  gr.Markdown("Put in a company ticker and ask about insights from their 10-K filings.")
  gr.Markdown("For example, type 'AAPL' and 'Tell me about the changes in revenue and other key metrics over the years 1996 to 1997.'")

  gr.Interface(
      fn=pipeline,
      inputs=["text", "text"],
      outputs=[gr.Plot(), "text"]
      )


demo.launch(share=True,debug=True)