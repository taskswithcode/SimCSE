import pdb
from scipy.spatial.distance import cosine
import argparse
import json
import os,sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR)

from simcse import SimCSE


def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]



class SimCSEModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug = False
        print("In SimCSE constructor")

    def init_model(self,model_name = None):
        if (model_name == None):
            model_name = "princeton-nlp/sup-simcse-roberta-large"
        self.model = SimCSE(model_name)

    def compute_embeddings(self,input_data,is_file):
        texts = read_text(input_data) if is_file == True else input_data
        embeddings = self.model.encode(texts)
        return texts,embeddings

    def output_results(self,output_file,texts,embeddings):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_dict = {}
        print("Total sentences",len(texts))
        for i in range(len(texts)):
                cosine_dict[texts[i]] = 1 - cosine(embeddings[0], embeddings[i])

        print("Input sentence:",texts[0])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='SGPT model for sentence embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="princeton-nlp/sup-simcse-bert-base-uncased",help="model name")
        results = parser.parse_args()
        obj = SimCSEModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
