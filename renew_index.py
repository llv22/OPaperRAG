import os
import sys
from glob import glob
import argparse

def conf():
    parser = argparse.ArgumentParser(description="Renew index of PDF files")
    parser.add_argument("-rp", "--root_path", help="Root folder of PDF files", default="~/Documents/OneDrive/Programming/Algorithm/01_ML/01_research")
    return parser.parse_args()

if __name__ == "__main__":
    ## see: find all pdf files in the root directory recursively and generate dictionary with key as file name and value as file path
    args = conf()
    root_path = args.root_path
    pdf_files = glob(root_path + "/**/*.pdf", recursive=True)
    pdf_file_to_path = {}
    
    cnt = 0
    for file in pdf_files:
        if os.path.basename(file) in pdf_file_to_path:
            print("Duplicate file name: ", os.path.basename(file), " at ", file, " and the existing file at ", pdf_file_to_path[os.path.basename(file)])
            cnt += 1
        pdf_file_to_path[os.path.basename(file)] = file
    print("Total duplicate files: ", cnt)
    assert len(pdf_files) - cnt == len(pdf_file_to_path), "Duplicate files found"
    print("Total unique PDF files: ", len(pdf_file_to_path))