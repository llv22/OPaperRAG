import argparse
from glob import glob
from pathlib import Path
from multiprocessing import Pool

from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI


def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_worker_num", type=int, default=4)
    parser.add_argument("--doc_chucks", type=int, default=10000)
    parser.add_argument("--llamaindex_root", type=str, default="./storage")
    parser.add_argument("--research_pdf_root", type=str, default="/Users/llv23/Documents/OneDrive/Programming/Algorithm/01_ML/01_research")
    parser.add_argument("--research_excel_path", type=str, default="/Users/llv23/Documents/OneDrive/PhD/Dialog-system_research_summary_by_orlando.xlsx")
    parser.add_argument("--annotation_pdf_root", type=str, default="/Users/llv23/Library/Mobile Documents/iCloud~QReader~MarginStudy/Documents/ML/Papers/")
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

if __name__ == "__main__":
    args = conf()
    research_pdf_root, research_excel_path, annotation_pdf_root, parallel_worker_num, doc_chucks, llamaindex_root = args.research_pdf_root, args.research_excel_path, args.annotation_pdf_root, args.parallel_worker_num, args.doc_chucks, args.llamaindex_root
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
        input_dir="/Users/llv23/Documents/OneDrive/Programming/Algorithm/01_ML/01_research",
        required_exts=required_exts,
        recursive=True,
    )
    docs = reader.load_data(num_workers=parallel_worker_num)
    # TODO: attach metadata
    # docs[0].metadata['annotation_path']
    nodes = Settings.node_parser.get_nodes_from_documents(docs)

    # see: split nodes into manageable chunks
    chunks = [nodes[i:i + doc_chucks] for i in range(0, len(nodes), doc_chucks)]

    # see: process chunks in parallel
    with Pool(processes=parallel_worker_num) as pool:
        indices = pool.map(create_index, chunks)

    # see: merge the processed indices
    index = VectorStoreIndex.merge(indices)

    # see: persist the index
    index.storage_context.persist(persist_dir=llamaindex_root)
    
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
    llm = OpenAI(model="gpt-4",temperature=0)
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=2,
        llm=llm
    )
    
    response = query_engine.query("List papers relevant with agentQ and where I can find them")
    print(response.print_response_stream())
