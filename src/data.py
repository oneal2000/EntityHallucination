from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
from collections import Counter
from urllib.parse import unquote
from tqdm import tqdm
import numpy as np
from datasets import Dataset
logging.basicConfig(level=logging.INFO)


class BaseDataset:
    # [TODO] evaluate

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))
        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)

            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)



class StrategyQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]

    demo_input_template = test_input_template = lambda self, ques: f'Generate a yes or no answer to the following question.\nQuestion: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the final answer is {ans}.'

    def __init__(self, data_path: str):
        dataset = []
        with open(os.path.join(data_path, 'strategyqa.json'), 'r') as fin:
            for line in fin:
                example = json.loads(line)
                qid = example['_id']
                question = example['text']
                cot = example['metadata']['cot'] if 'cot' in example['metadata'] else ''
                ans = example['metadata']['answer']
                ctxs = example['ctxs']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'cot': cot,
                    'answer': ans,
                    'ctxs': ctxs,
                })
        self.dataset = Dataset.from_list(dataset)


class WikiMultiHopQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            'answer': "19 June 2013",
        },
        {
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
        },
        {
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
        },
        {
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
        },
        {
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
        },
        {
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
        },
        {
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
        },
        {
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
        }
    ]
    demo_input_template = test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'


    def __init__(self, data_path: str): 
        dataset = []
        with open(os.path(data_path, 'queries.jsonl'), 'r') as fin:
            for line in fin:
                example = json.loads(line)
                qid = example['_id']
                question = example['text']
                ans = example['metadata']['answer']
                ans_id = example['metadata']['answer_id']
                ctxs = example['metadata']['ctxs']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                    'ctxs': ctxs,
                })
        self.dataset = Dataset.from_list(dataset)
        self.id_aliases_file = os.path(data_path, 'id_aliases.json')

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if not hasattr(cls, 'id_alias'):
            cls.id_alias: Dict[str, List[str]] = {}
            with open(cls.id_aliases_file, 'r') as fin:
                for l in fin:
                    l = json.loads(l)
                    cls.id_alias[l['Q_id']] = l['aliases']

        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        else:
            return []


class NaturalQuestion(BaseDataset):
    demo_input_template = test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'

    def __init__(self, data_path: str):

        def anno_to_answer(tokens, anno, addition=0):
            ans = []
            for i in range(anno["start_token"], anno["end_token"] + addition):
                if tokens[i]["html_token"] == False:
                    ans.append(tokens[i]["token"])
            return " ".join(ans)

        dataset = []
        with open(os.path.join(data_path, 'v1.0-simplified_nq-dev-all.jsonl'), "r") as fin:
            for line in fin:
                example = json.loads(line)
                qid = example["example_id"]
                question = example["question_text"] + "?"
                
                tokens = example["document_tokens"]
                long_answers = []
                for ans in example["long_answer_candidates"]:
                    long_answers.append(anno_to_answer(tokens, ans, addition=1))
                short_answers = []
                for anno in example["annotations"]:
                    for ans in anno["short_answers"]:
                        short_answers.append(anno_to_answer(tokens, ans, addition=0))
                    if anno["yes_no_answer"] != "NONE":
                        short_answers.append(anno["yes_no_answer"].lower().strip())

                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': long_answers,
                    'short_answer': short_answers,
                })
        self.dataset = Dataset.from_list(dataset)