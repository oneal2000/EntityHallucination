import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import logging 
from data import StrategyQA, WikiMultiHopQA, NaturalQuestion
from generate import RetrieverGenerator

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="llama2")
    parser.add_argument("--method", type=str, required=True, choices=["non-retrieval", "single", "fix-length", "token", "entity"])
    parser.add_argument("--hallucination_threshold", type=float)
    parser.add_argument("--entity_solver", type=str, choices=["avg", "max", "min", "first"], help="when using entity, how to calculate entity score")
    parser.add_argument("--sentence_solver", type=str, choices=["avg", "max", "min"], help="when using entity, how to calculate sentence score")
    parser.add_argument("--dataset", type=str, required=True, choices=["nq", "2wikihop", "strategyqa", "truthfulqa", "hotpotqa"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--generate_max_length", type=int, required=True, help="max length for language model to generate")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--retriever", type=str, choices=["BM25", "sgpt"])
    parser.add_argument("--fewshot", type=int)
    parser.add_argument("--low_prob", type=float, help="just for token-level")
    parser.add_argument("--keep_prob", type=float, help="just for token-level")
    parser.add_argument("--sample", type=int, default=-1, help="if none, use all dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    logger.info(f"{args}")

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.data_path)
    elif args.dataset == '2wikihop':
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "nq":
        data = NaturalQuestion(args.data_path)
    # elif args.dataset == 'truthfulqa':
    #     data = TruthfulQA(args.data_path)
    # elif args.dataset == 'hotpotqa':
    #     data = HotpotQA(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.sample != -1:
        data = data.shuffle()
        data = data.select(range(args.sample))
    
    # output file
    if args.method == "entity": 
        output_file_name = "_".join([
            args.method, 
            args.entity_solver, 
            args.hallucination_threshold,
        ])
    elif args.method == "token": 
        output_file_name = "_".join([
            args.method, 
            args.low_prob, 
            args.keep_prob,
        ])
    else:
        output_file_name = args.method
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    output_file = open(os.path.join(args.output_dir, output_file_name+".txt"), "w")

    model = RetrieverGenerator(args)

    for i in tqdm(range(len(data))):
        batch = data[i]
        question = batch["question"]
        answer = model.inference(question)
        ret = {
            "qid": batch["qid"], 
            "answer": answer,
            # "hallucination_count": hallucination_count,
            # "token_count": token_count - last_token_count, 
            # "sentence_count": sentence_count - last_sentence_count
        }
        output_file.write(json.dumps(ret)+"\n")