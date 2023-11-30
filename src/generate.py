import numpy as np
import openai
import spacy
from math import exp
from retriever import BM25
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast

nlp = spacy.load("en_core_web_sm")

class BasicGenerator:
    def __init__(self, model_name_or_path):
        if "gpt" in model_name_or_path:
            self.use_gpt = True
            self.gpt_model_name = model_name_or_path[4:]
        else:
            self.use_gpt = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

    def generate(self, input_text, max_length, return_logprobs=False):
        if self.use_gpt:
            response = openai.Completion.create(
                model = self.gpt_model_name,
                prompt = input_text,
                temperature = 0.0, # 0.0 = deterministic
                max_tokens = max_length,
                logprobs = 1,
            )
            text = response.choices[0].text
            tokens = response["choices"][0]["logprobs"]["tokens"]
            logprobs = response.choices[0]["logprobs"]["token_logprobs"] if return_logprobs else None
            return text, tokens, logprobs

        else:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)

            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            input_length = input_ids.shape[1]

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            if return_logprobs:
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
            else:
                logprobs = None
            return text, tokens, logprobs
        

class RetrieverGenerator:
    def __init__(self, args):
        args = vars(args)
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if self.method != "non-retrieval":
            gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            # TODO: add sgpt
            assert self.retriever == "BM25"
            self.retriever = BM25(
                tokenizer = gpt2_tokenizer, 
                index_name = "drad_wiki", 
                engine = "elasticsearch",
            )
        if self.method in ["token", "entity"]:
            self.modifier = self.entity_modifier if self.method == "entity" else self.token_modifier

    def get_retrieve_passage(self, question, replace_xxx = False, max_query_length = 64):
        return ["1", "2", "3", "4", "5"]
        
        if replace_xxx:
            question = question.replace("[xxx]", "")
        docs_ids, docs = self.retriever.retrieve(
            queries = [question], 
            topk = 5, # TODO
            max_query_length = max_query_length,
        )
        return docs[0]
    
    def get_prompt(self, method, question = None, prev = None, passage = None):
        pass
        # TODO

    def token_modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = sentences[:sid]
                prev = " ".join(prev)
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True

            tid = tr + 1
        
        for sid in range(len(sentences)):
            probs = [1 - exp(v) for v in logprobs[sid]]
            probs = np.array(probs)
            
        # No hallucination
        return text, None, False

    def entity_modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok)
            assert apr != -1
            belonging[pos:pos+apr+len(tok)] = [tid] * (apr + len(tok) - 1)
            pos = apr + len(tok)
        
        entity_intv = []
        pos = 0
        for sid, sent in range(len(sentences)):
            tmp = []
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = sentences[:sid]
                prev = " ".join(prev)
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question):
        if self.method == "non-retrieval":
            text, _, _ = self.generator.generate(question, self.generate_max_length)
        
        elif self.method == "single":
            docs = self.get_retrieve_passage(question)
            prompt = self.get_prompt(
                method = self.method, 
                question = question, 
                passage = docs)
            text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        
        elif self.method == "fix-length":
            prev = ""
            while True:
                docs = self.get_retrieve_passage(question + " " + prev)
                prompt = self.get_prompt(
                    method = self.method, 
                    question = question, 
                    prev = prev, 
                    passage = docs)
                # 目前设计是一个句子检索一次
                text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                sentences = nlp(text).sents
                # 如果没有句子，就结束
                if len(sentences) == 0:
                    break
                # 保留第一个句子
                prev = prev + " " + next(sentences)
                if len(prev) > self.generate_max_length:
                    break
            text = prev
            
        elif self.method == "token" or self.method == "entity":
            prev = ""
            while True:
                text, tokens, logprobs = self.generator.generate(question + " " + prev, self.generate_max_length, return_logprobs=True)
                ptext, curr, hallucination = self.modifier(text, tokens, logprobs)
                if not hallucination:
                    prev = prev + " " + ptext
                else:
                    docs = self.get_retrieve_passage(curr)
                    prompt = self.get_prompt(
                        method = self.method, 
                        question = curr, 
                        prev = prev, 
                        passage = docs)
                    new_text = self.generator.generate(prompt, self.generate_max_length)
                    new_text = new_text.strip()
                    prev = prev + " " + ptext + " " + new_text
                if len(prev) > self.generate_max_length:    
                    break   
            text = prev
        
        else:
            raise NotImplementedError
        
        return text