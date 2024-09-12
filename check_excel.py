import argparse
import json
import pandas as pd

def conf():
    parser = argparse.ArgumentParser(description="check excel file")
    parser.add_argument("--excel_file", help="excel file of paper", default="/Users/llv23/Documents/OneDrive/PhD/Dialog-system_research_summary_by_orlando.xlsx")
    return parser.parse_args()

if __name__ == "__main__":
    ## see: find duplicate files in each tab of excel file
    args = conf()
    excel_file = args.excel_file
    df = pd.read_excel(excel_file, sheet_name=None)
    names = list(df.keys())
    # see: filter out the last tab
    for tab in names[:-1]:
        df_tab = df[tab]
        title_to_duplication = {}
        # print(f"tab: {tab} with shape: {df_tab.shape}")
        # see: iterate over each row and check if 'Paper title' is duplicated
        for i, row in df_tab.iterrows():
            title = row['Paper title']
            if df_tab[df_tab['Paper title'] == title].shape[0] > 1:
                # print(f"tab: {tab} with title: {title} at line {i} is duplicated")
                title_to_duplication[title] = df_tab[df_tab['Paper title'] == title].index.tolist()
        if len(title_to_duplication) > 0:
            summary_dict = {
                "size": len(title_to_duplication),
                "duplicated": title_to_duplication,
            }
            # print(f"tab: {tab} with title to duplication: {title_to_duplication}")
            with open(f"{tab}_duplication.json", "w") as f:
                json.dump(summary_dict, f, indent=1)