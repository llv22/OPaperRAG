import argparse
import json
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from time import time
from tqdm import tqdm

from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    ListIndex,
)
from llama_index.core.storage import StorageContext
from llama_index.llms.openai import OpenAI


def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate_index", type=bool, default=True)
    parser.add_argument("--parallel_worker_num", type=int, default=4)
    parser.add_argument("--doc_chucks", type=int, default=-1)
    parser.add_argument("--llamaindex_root", type=str, default="./storage")
    parser.add_argument("--research_pdf_root", type=str, default="~/Documents/OneDrive/Programming/Algorithm/01_ML/01_research/")
    parser.add_argument("--research_excel_path", type=str, default="~/Documents/OneDrive/PhD/Dialog-system_research_summary_by_orlando.xlsx")
    parser.add_argument("--annotation_pdf_root", type=str, default="~/Library/Mobile Documents/iCloud~QReader~MarginStudy/Documents/ML/Papers/")
    parser.add_argument("--saved_query_file", type=str, default="query_answers_for_opaper_rag.json")
    return parser.parse_args()

def annotation_files(annotation_pdf_root):
    """search all pdf under annotation_pdf_root

    Args:
        annotation_pdf_root (str): annotation path
    """
    files = glob(f"{annotation_pdf_root}/**/*.pdf", recursive=True)
    return {Path(f).name.__str__(): f for f in files}


def create_index(nodes):
    return VectorStoreIndex(nodes, include_embeddings=True)

def write_to_file(saved_query_file, query_answers):
    with open(saved_query_file, "w") as f:
        json.dump(query_answers, f, indent=1)

if __name__ == "__main__":
    args = conf()
    research_pdf_root, research_excel_path, annotation_pdf_root, parallel_worker_num, doc_chucks, llamaindex_root, if_recreate_index, saved_query_file = args.research_pdf_root, args.research_excel_path, args.annotation_pdf_root, args.parallel_worker_num, args.doc_chucks, args.llamaindex_root, args.recreate_index, args.saved_query_file
    start = time()
    if not Path(llamaindex_root).exists() and if_recreate_index:
        # see: load annotation file
        annotation_file_to_path = annotation_files(annotation_pdf_root)
        entity_extractor = EntityExtractor(
            prediction_threshold=0.5,
            label_entities=False,  # include the entity label in the metadata (can be erroneous)
            device="cpu",  # set to "cuda" if you have a GPU
        )
        node_parser = SentenceSplitter()
        required_exts = [".pdf"]
        reader = SimpleDirectoryReader(
            input_dir=research_pdf_root,
            required_exts=required_exts,
            recursive=True,
        )
        docs = reader.load_data(num_workers=parallel_worker_num)
        # see: attach metadata of annotation_path for each docs
        unique_docs = set([doc.metadata['file_name'] for doc in docs])
        unique_doc_to_annotation_path = {filename: annotation_path for filename, annotation_path in annotation_file_to_path.items() if filename in unique_docs}
        for d in docs:
            if d.metadata['file_name'] in unique_doc_to_annotation_path:
                d.metadata['annotation_file_path'] = unique_doc_to_annotation_path[d.metadata['file_name']]
        if doc_chucks != -1 and len(docs) > doc_chucks:
            nodes = Settings.node_parser.get_nodes_from_documents(docs)
            # see: split nodes into manageable chunks
            chunks = [nodes[i:i + doc_chucks] for i in range(0, len(nodes), doc_chucks)]
            # see: process chunks in parallel
            with Pool(processes=parallel_worker_num) as pool:
                indices = list(tqdm(pool.map(create_index, chunks), total=len(chunks)))
            # TODO: merge the processed indices. Here still having bugs here
            index = ListIndex(indices)
        else:
            index = VectorStoreIndex.from_documents(docs, show_progress=True)
        # see: persist the index, creating
        index.storage_context.persist(persist_dir=llamaindex_root)
        print(f"create index time: {time() - start: .2f} seconds")
    else:
        # see: load from llamaindex_root, reading
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=llamaindex_root)) 
        print(f"load index time: {time() - start: .2f} seconds")
    
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
    llm = OpenAI(model="gpt-4",temperature=0)
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=2,
        llm=llm
    )
    
    query = input("\n\nplease Enter Your Query for Papers: ")
    query_answers = []
    run_num = 1
    while query is not None and len(query.strip()) > 0 and query.upper() != "EXIT":
        query_start = time()
        response = query_engine.query(query)
        v = "".join(response.response_gen)
        query_answers.append({
            "Run": run_num,
            "Query": query,
            "Result": v,
        })
        print(f"Run {run_num}[{time() - query_start: .2f} seconds]: \n\tresult: {v}")
        run_num += 1
        write_to_file(saved_query_file, query_answers)
        #see: clear the query
        query = input("\nplease Enter Your Query for Papers: ")
    if len(query_answers) == 0:
        query = "List papers relevant with SAM and their file path and annotation path if having"
        query_start = time()
        response = query_engine.query(query)
        v = "".join(response.response_gen)
        query_answers.append({
            "Run": run_num,
            "Query": query,
            "Result": v,
        })
        print(f"Run {run_num}[{time() - query_start: .2f} seconds]: \n\tresult: {v}")
        write_to_file(saved_query_file, query_answers)
        run_num += 1
