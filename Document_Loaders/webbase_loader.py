from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/Apple-iPad-Mini-A17-Pro/dp/B0DK3Z6NMQ/ref=sr_1_1_sspa?adgrpid=64642959491&dib=eyJ2IjoiMSJ9.TjArcqTpEoFqEA8YcW3QWzhjCOUr2Em3exXsNVPfusXLQSz9z2GceIJKw7UL7E34pNfE7lilixs5jkFk_Kq_4DsuTer_qkHEjsIb_Vinv-IKSKvkEkbB47JxuWK4iEU12VuEuDPQF11xJFbwoTpp2Gk6XYbzEG67XkvBfgzzV9JcVXH-W6eQDscsltT14ssSEygTiydzM4FiJnJNUnv_MAIerDzKPyVoRCbbqVMAp44SV-gjJp2tmr3eXD_zbi9HOIjhwTtMloOgkp1zvA3VOnZg9xw7_M-PJv6vup9p4yM.l67FyN5fUs_BJkFMSGvaAoDj7IxV9ifIONK_FcfNdnY&dib_tag=se&ext_vrnc=hi&hvadid=596889747090&hvdev=c&hvlocphy=9061992&hvnetw=g&hvqmt=e&hvrand=5936725622552317510&hvtargid=kwd-298653229107&hydadcr=20162_2225824&keywords=ipad%2Bmini&mcid=7c165292037135d380d709d882e60dd8&qid=1744565841&s=electronics&sr=1-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))