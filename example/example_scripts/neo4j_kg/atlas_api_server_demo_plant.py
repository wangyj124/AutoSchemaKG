# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from openai import OpenAI
from atlas_rag.llm_generator import LLMGenerator
from configparser import ConfigParser
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding, NvEmbed, EmbeddingAPI
from neo4j import GraphDatabase
import faiss
import datetime
import logging
from logging.handlers import RotatingFileHandler
from atlas_rag.retriever.lkg_retriever.lkgr import LargeKGRetriever
from atlas_rag.retriever.lkg_retriever.tog import LargeKGToGRetriever
from atlas_rag.kg_construction.neo4j.neo4j_api import LargeKGConfig, start_app

# use sentence embedding if you want to use sentence transformer
# use NvEmbed if you want to use NvEmbed-v2 model
# sentence_model = SentenceTransformer('all-MiniLM-L12-v2',truncate_dim=32, device='cuda:0')
# sentence_encoder = SentenceEmbedding(sentence_model)
# Load OpenRouter API key from config file
config = ConfigParser()
config.read('config.ini')
# reader_model_name = "meta-llama/llama-3.3-70b-instruct"
retriever_model_name = "qwen3-30B-A3B-Instruct"
reader_model_name = "qwen3-30B-A3B-Instruct"
client = OpenAI(
  # base_url="https://openrouter.ai/api/v1",
  # api_key=config['settings']['OPENROUTER_API_KEY'],
  base_url="http://10.67.74.45:8001/v1/",
  # api_key=config['settings']['DEEPINFRA_API_KEY'],_key=""
  api_key="sk-6aP9xT2mQv1sRb8Zc4NyUe0Lf7Hk3Wd5"
)
sentence_encoder = EmbeddingAPI(client, model_name="qwen3-embedding-8B")
llm_generator = LLMGenerator(client=client, model_name=reader_model_name)

retriever_llm_generator = LLMGenerator(client=client, model_name=retriever_model_name)

# prepare necessary objects for instantiation of LargeKGRetriever: neo4j driver, faiss index etc.
# neo4j_uri = "bolt://localhost:8013" # use bolt port for driver connection
neo4j_uri = "bolt://10.67.75.85:7687" # use bolt port for driver connection
user = "neo4j"
# password = "admin2024"
password = "Wayne0327"
# keyword = 'cc_en' # can be wiki or pes2o  # keyword to identify the cc_en dataset
keyword = 'plantstation_100' # can be wiki or pes2o  # keyword to identify the cc_en dataset
driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

text_index = faiss.read_index(f"D://ai//wangyejun//AutoSchemaKG-copy//AutoSchemaKG//output//plantstation_kg//vector_index//text_nodes_plantstation_100_from_json_with_emb_non_norm.index", faiss.IO_FLAG_MMAP)
node_index = faiss.read_index(f"D://ai//wangyejun//AutoSchemaKG-copy//AutoSchemaKG//output//plantstation_kg//vector_index//triple_nodes_plantstation_100_from_json_with_emb_non_norm.index", faiss.IO_FLAG_MMAP)

# setup logger
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
log_file_path = f'./log/LargeKGRAG_plantstation_100.log'
logger = logging.getLogger("LargeKGRAG")
logger.setLevel(logging.INFO)
max_bytes = 50 * 1024 * 1024  # 50 MB
if not os.path.exists(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
logger.addHandler(handler)

retriever = LargeKGRetriever(keyword = keyword,
                             neo4j_driver=driver,
                             llm_generator=retriever_llm_generator,
                             sentence_encoder=sentence_encoder,
                             node_index= node_index,
                             passage_index=text_index,
                             topN = 5,
                             number_of_source_nodes_per_ner = 10,
                             sampling_area = 250,
                             logger = logger,
                            simple_ner=True) # since cc_en is enormous compared to other dataset, we have different retrieval mechanism for it, which here we use keyword to identify cc_en.
tog_retriever = LargeKGToGRetriever(
    keyword = keyword,
    neo4j_driver=driver,
    topN = 5,
    Dmax = 2,
    Wmax = 3,
    llm_generator=retriever_llm_generator,
    sentence_encoder = sentence_encoder,
    # filter_encoder = SentenceEmbedding(SentenceTransformer('all-MiniLM-L12-v2', device='cuda:0')),
    filter_encoder = sentence_encoder,
    node_index = node_index,
    logger=logger
)

large_kg_config = LargeKGConfig(
    largekg_retriever = tog_retriever,
    reader_llm_generator = llm_generator, # you can use the same llm_generator as above or a different one for reading the retrieved passages,
    driver=driver,
    logger=logger,
    is_felm = False,
    is_mmlu = False,
    
)

start_app(user_config=large_kg_config, host="0.0.0.0", port = 10089, reload=False)