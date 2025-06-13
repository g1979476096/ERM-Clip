import re
import json
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List, Dict, Tuple, Set
import nltk
from nltk.tokenize import sent_tokenize


class FactVerificationSystem:
    def __init__(self,
                 constituency_parser_model: str = "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
                 bert_model_name: str = 'bert-base-uncased',
                 max_length: int = 512):
        """初始化事实验证系统"""
        # 实体提取模块
        self.predictor = None  # 选区解析器（需要时加载）
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        # 证据句选择模块
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.evidence_model = SentenceSelectionModel(bert_model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evidence_model.to(self.device)
        self.max_length = max_length

    def extract_entity_keywords(self, sentence: str) -> List[str]:
        """使用选区解析器从句子中提取实体关键词"""
        if self.predictor is None:
            from allennlp.predictors.predictor import Predictor
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

        try:
            parse_output = self.predictor.predict(sentence=sentence)
            tree = parse_output['trees']

            np_pattern = re.compile(r'\(NP (.*?)\)')
            np_matches = np_pattern.findall(tree)

            keywords = []
            for np in np_matches:
                np_clean = re.sub(r'\([^)]*\)', '', np).strip()
                if np_clean and len(np_clean.split()) <= 5:
                    words = [w for w in np_clean.split() if w.lower() not in self.stopwords]
                    if words:
                        keywords.append(np_clean)

            return list(set(keywords))
        except Exception as e:
            print(f"关键词提取错误: {e}")
            return []

    def fetch_wikipedia_docs(self, entity: str, limit: int = 5) -> List[Dict]:
        """调用MediaWiki API获取与实体相关的文档"""
        base_url = "https://en.wikipedia.org/w/api.php"
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': entity,
                'srlimit': limit,
                'srprop': 'snippet'
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            results = response.json()

            docs = []
            for item in results['query']['search']:
                docs.append({
                    'title': item['title'],
                    'pageid': item['pageid'],
                    'snippet': item['snippet'],
                    'wordcount': item['wordcount']
                })

            return docs
        except Exception as e:
            print(f"获取Wikipedia文档错误 (实体: {entity}): {e}")
            return []

    def get_page_content(self, pageid: int) -> str:
        """获取Wikipedia页面的完整内容"""
        base_url = "https://en.wikipedia.org/w/api.php"
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'pageids': pageid,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            results = response.json()

            page = results['query']['pages'][str(pageid)]
            return page.get('extract', '')
        except Exception as e:
            print(f"获取页面内容错误 (pageid: {pageid}): {e}")
            return ''

    def build_evidence_set(self, sentence: str, max_docs_per_entity: int = 5) -> List[str]:
        """构建证据句集合"""
        # 提取实体关键词
        entities = self.extract_entity_keywords(sentence)
        if not entities:
            return []

        # 获取每个实体的相关文档
        entity_docs = {}
        for entity in entities:
            docs = self.fetch_wikipedia_docs(entity, limit=max_docs_per_entity)
            entity_docs[entity] = docs

        # 提取相关文档中的句子作为证据
        evidence_sentences = set()
        for entity, docs in entity_docs.items():
            for doc in docs:
                content = self.get_page_content(doc['pageid'])
                if content:
                    sentences = [s for s in sent_tokenize(content) if len(s.split()) > 5]
                    evidence_sentences.update(sentences)

        return list(evidence_sentences)

    def prepare_input(self, claim: str, evidence: str) -> Dict[str, torch.Tensor]:
        """准备证据句选择模型的输入"""
        encoded_input = self.tokenizer.encode_plus(
            claim,
            evidence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoded_input.items()}

    def score_sentences(self, claim: str, evidence_sentences: List[str]) -> List[Tuple[str, float]]:
        """对证据句子进行打分"""
        self.evidence_model.eval()
        scores = []

        with torch.no_grad():
            for evidence in evidence_sentences:
                inputs = self.prepare_input(claim, evidence)
                output = self.evidence_model(**inputs)
                score = torch.sigmoid(output).item()
                scores.append((evidence, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def select_top_evidence(self, claim: str, evidence_sentences: List[str], top_k: int = 5) -> List[str]:
        """选择分数最高的top_k个证据句子"""
        if not evidence_sentences:
            return []

        scored_sentences = self.score_sentences(claim, evidence_sentences)
        return [sent for sent, score in scored_sentences[:top_k]]

    def verify_claim(self, claim: str, max_docs_per_entity: int = 5, top_k: int = 5) -> Dict:
        """完整的验证流程：从声明到证据选择"""
        result = {
            'claim': claim,
            'extracted_entities': [],
            'evidence_sentences': [],
            'top_evidence': []
        }

        # 1. 构建证据句集合
        entities = self.extract_entity_keywords(claim)
        result['extracted_entities'] = entities

        evidence_sentences = self.build_evidence_set(claim, max_docs_per_entity)
        result['evidence_sentences'] = evidence_sentences

        # 2. 选择最相关的证据句
        top_evidence = self.select_top_evidence(claim, evidence_sentences, top_k)
        result['top_evidence'] = top_evidence

        return result


class SentenceSelectionModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """证据句选择模型结构"""
        super(SentenceSelectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids):
        """模型前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.mlp(cls_output)
        score = self.tanh(logits)

        return score


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    system = FactVerificationSystem()

    # 示例新闻句子
    news_sentence = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."

    # 执行完整验证流程
    result = system.verify_claim(news_sentence, max_docs_per_entity=5, top_k=5)

    # 输出结果
    print(f"声明: {result['claim']}")
    print(f"\n提取的实体: {result['extracted_entities']}")
    print(f"\n证据句总数: {len(result['evidence_sentences'])}")

    print("\nTop 5 证据句子:")
    for i, evidence in enumerate(result['top_evidence'], 1):
        print(f"{i}. {evidence[:150]}...")