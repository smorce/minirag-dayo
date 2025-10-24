# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import os
import sys
from minirag import MiniRAG, QueryParam
from minirag.llm.hf import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="PHI")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=[],
        help="Metadata filters for queries in key=value format",
    )
    args = parser.parse_args()
    return args


args = get_args()


def parse_metadata_filters(metadata_args: list[str]) -> dict[str, str]:
    metadata = {}
    for item in metadata_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid metadata filter '{item}'. Expected format key=value."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                f"Invalid metadata filter '{item}'. Key must not be empty."
            )
        metadata[key] = value
    return metadata


try:
    METADATA_FILTERS = parse_metadata_filters(args.metadata)
except ValueError as metadata_error:
    print(metadata_error)
    sys.exit(1)


if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
if METADATA_FILTERS:
    print("USING METADATA FILTERS:", METADATA_FILTERS)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)


# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())

# A toy query
query = 'What does LiHua predict will happen in "The Rings of Power"?'
query_param = QueryParam(mode="mini", metadata_filters=METADATA_FILTERS or None)
answer = (
    rag.query(query, param=query_param).replace("\n", "").replace("\r", "")
)
print(answer)
