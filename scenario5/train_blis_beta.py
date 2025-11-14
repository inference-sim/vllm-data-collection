import argparse
import json


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--results_path",
                            default=".", 
                            help="Location to save alpha model")
    args = parser.parse_args()        




