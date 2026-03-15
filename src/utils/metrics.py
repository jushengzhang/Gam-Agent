import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F
import re

class MetricsCalculator:
    """评估指标计算工具类"""
    
    def __init__(self, use_bleu=True, use_rouge=True, use_bert_score=False):
        """
        初始化指标计算器
        
        Args:
            use_bleu: 是否使用BLEU
            use_rouge: 是否使用ROUGE
            use_bert_score: 是否使用BERTScore
        """
        self.use_bleu = use_bleu
        self.use_rouge = use_rouge
        self.use_bert_score = use_bert_score
        
        # 初始化ROUGE计算器
        if self.use_rouge:
            self.rouge_calculator = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    def extract_choice_number(self, text):
        """
        从文本中提取选择题答案编号
        
        Args:
            text: 文本
            
        Returns:
            str: 提取的选择题答案编号，如果没有找到则返回空字符串
        """
        # 使用多个模式匹配
        patterns = [
            # 直接匹配单个数字
            r'^(\d+)$',
            # 匹配"选择/选项X"格式
            r'选择\s*(\d+)',
            r'选项\s*(\d+)',
            # 匹配"我选择X"格式 
            r'我选择\s*(\d+)',
            r'我的答案是\s*(\d+)',
            # 匹配"Answer is X"格式
            r'[aA]nswer\s*(?:is)?\s*(\d+)',
            # 匹配"The answer is X"格式
            r'[tT]he\s+answer\s+is\s+(\d+)',
            # 匹配"I choose X"格式
            r'[iI]\s+choose\s+(\d+)',
            # 匹配"Option X"格式
            r'[oO]ption\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def calculate_choice_accuracy(self, reference, candidate):
        """
        计算选择题准确率
        
        Args:
            reference: 参考答案（正确的选项编号）
            candidate: 候选答案（模型生成的答案）
            
        Returns:
            float: 1.0表示正确，0.0表示错误
        """
        if not reference:
            return 0.0
            
        # 提取候选答案中的选项编号
        extracted_choice = self.extract_choice_number(candidate)
        
        # 如果参考答案和提取的选项编号匹配，则正确
        if extracted_choice and extracted_choice == reference:
            return 1.0
            
        return 0.0
    
    def calculate_bleu(self, reference, candidate):
        """
        计算BLEU得分
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            float: BLEU得分
        """
        if not self.use_bleu:
            return 0.0
            
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            return sentence_bleu([reference_tokens], candidate_tokens)
        except Exception as e:
            print(f"计算BLEU出错: {e}")
            return 0.0
    
    def calculate_rouge(self, reference, candidate):
        """
        计算ROUGE得分
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            float: ROUGE-1 F1得分
        """
        if not self.use_rouge:
            return 0.0
            
        try:
            rouge_scores = self.rouge_calculator.score(reference, candidate)
            return rouge_scores['rouge1'].fmeasure
        except Exception as e:
            print(f"计算ROUGE出错: {e}")
            return 0.0
    
    def calculate_bert_score(self, reference_features, candidate_features):
        """
        计算BERTScore（基于特征向量的余弦相似度）
        
        Args:
            reference_features: 参考文本特征向量
            candidate_features: 候选文本特征向量
            
        Returns:
            float: BERTScore（余弦相似度）
        """
        if not self.use_bert_score:
            return 0.0
            
        try:
            # 计算余弦相似度
            return F.cosine_similarity(reference_features, candidate_features).item()
        except Exception as e:
            print(f"计算BERTScore出错: {e}")
            return 0.0
    
    def calculate_combined_score(self, reference, candidate, ref_features=None, cand_features=None, is_choice=False):
        """
        计算综合得分
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            ref_features: 参考文本特征向量（可选）
            cand_features: 候选文本特征向量（可选）
            is_choice: 是否为选择题答案
            
        Returns:
            float: 综合得分
        """
        # 如果是选择题，优先使用选择题评分方式
        if is_choice:
            return self.calculate_choice_accuracy(reference, candidate)
            
        scores = []
        
        # 计算BLEU
        if self.use_bleu:
            bleu_score = self.calculate_bleu(reference, candidate)
            scores.append(bleu_score)
        
        # 计算ROUGE
        if self.use_rouge:
            rouge_score = self.calculate_rouge(reference, candidate)
            scores.append(rouge_score)
        
        # 计算BERTScore
        if self.use_bert_score and ref_features is not None and cand_features is not None:
            bert_score = self.calculate_bert_score(ref_features, cand_features)
            scores.append(bert_score)
        
        # 返回平均得分
        if not scores:
            return 0.0
        return np.mean(scores) 