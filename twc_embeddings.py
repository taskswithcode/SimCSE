from simcse import SimCSE
import pdb
from scipy.spatial.distance import cosine
import argparse
import json


def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]



class SimCSEModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def init_model(self):
        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

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
        for key in sorted_dict:
            print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict))
        return sorted_dict


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='SGPT model for sentence embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        results = parser.parse_args()
        obj = SimCSEModel()
        obj.init_model()
        texts, embeddings = obj.compute_embeddings(results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
