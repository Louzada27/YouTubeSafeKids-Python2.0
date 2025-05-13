"""
Filtro para avaliar a linguagem inapropriada do conteúdo.

Este módulo fornece funcionalidades para detectar linguagem inapropriada em textos
usando uma combinação de regex com palavras do dicionário e o modelo BERTimbau.
"""

import os
import re
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from app.filters.base import BaseFilter



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



class InappropriateFilter(BaseFilter):
    """Filtro para avaliar linguagem inapropriada."""
    
    def __init__(
        self,
        model_path: str = "ModeloInapropriada",
        max_length: int = 128,
        threshold: float = 0.5
    ):
        """
        Inicializa o filtro de linguagem inapropriada.
        
        Args:
            model_path: Caminho para o modelo treinado
            max_length: Tamanho máximo da sequência
            threshold: Limiar para considerar um texto como inapropriado
        """
        super().__init__(
            name="Linguagem Inapropriada",
            description="Avalia o nível de linguagem inapropriada do conteúdo do vídeo",
            default_enabled=True
        )
        self.default_value = 50  # Valor padrão para a barra de inapropriação
        self.max_length = max_length
        self.threshold = threshold
        
        # Carrega o modelo e tokenizer
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            logger.info(f"Modelo carregado com sucesso de {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {str(e)}")
            raise
        
        # Carrega o dicionário de palavras inapropriadas
        try:
            dict_path = os.path.join("app", "nlp", "dicionario.txt")
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.inappropriate_words = [line.strip().lower() for line in f if line.strip()]
            logger.info(f"Dicionário carregado com {len(self.inappropriate_words)} palavras")
        except Exception as e:
            logger.error(f"Erro ao carregar dicionário: {str(e)}")
            self.inappropriate_words = []
    
    def check_dictionary(self, text: str) -> Tuple[bool, List[str]]:
        """
        Verifica se o texto contém palavras do dicionário.
        
        Args:
            text: Texto para verificar
            
        Returns:
            Tuple[bool, List[str]]: (Se encontrou palavras inapropriadas, Lista de palavras encontradas)
        """
        text_lower = text.lower()
        found_words = []
        
        for word in self.inappropriate_words:
            # Cria um padrão regex que corresponde à palavra com ou sem acentos
            pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            if pattern.search(text_lower):
                found_words.append(word)
        
        return len(found_words) > 0, found_words
    
    def predict_text(
        self,
        text: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Faz a predição para um texto.
        
        Args:
            text: Texto para classificar
            
        Returns:
            Tuple[str, Dict[str, float]]: (Classe predita, Probabilidades)
        """
        # Primeiro verifica o dicionário
        found_inappropriate, found_words = self.check_dictionary(text)
        if found_inappropriate:
            logger.info(f"Palavras inapropriadas encontradas: {found_words}")
            return "inapropriado", {"apropriado": 0.0, "inapropriado": 1.0}
        
        # Se não encontrou palavras no dicionário, usa o modelo
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        probs = probabilities[0].numpy()
        probabilities_dict = {
            "apropriado": float(probs[0]),
            "inapropriado": float(probs[1])
        }
        
        pred_label = max(probabilities_dict.items(), key=lambda x: x[1])[0]
        return pred_label, probabilities_dict
    
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro.
        
        Returns:
            Dict com as informações do filtro
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.is_enabled(),
            "type": "inappropriate",
            "default_value": self.default_value,
        }
    
    def get_intensity(self) -> float:
        """
        Retorna o valor atual da barra de intensidade (0-100).
        
        Returns:
            float: Valor da intensidade
        """
        return self.default_value
    
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa os dados do vídeo e retorna o score de linguagem inapropriada.
        
        Args:
            video_data: Dados do vídeo a serem processados
            
        Returns:
            float: Score de linguagem inapropriada (0-1)
        """
        # Verifica se já foi processado antes
        if 'inappropriate_score' in video_data:
            return video_data['inappropriate_score']

        # Se o filtro estiver desativado, retorna 1 (aceita o vídeo)
        if not self.is_enabled():
            video_data['inappropriate_score'] = 1.0
            video_data['inappropriate_details'] = {
                'is_inappropriate': False,
                'inappropriate_level': 'apropriado',
                'probabilities': {'apropriado': 1.0, 'inapropriado': 0.0},
                'filter_enabled': False
            }
            return 1.0

        # Extrai apenas a transcrição do vídeo
        transcript = video_data.get('transcription', '')
        
        if not transcript.strip():
            logger.warning("Transcrição vazia ou não disponível - Rejeitando vídeo")
            video_data['inappropriate_score'] = 0.0
            video_data['inappropriate_details'] = {
                'is_inappropriate': True,
                'inappropriate_level': 'inapropriado',
                'probabilities': {'apropriado': 0.0, 'inapropriado': 1.0},
                'rejection_reason': 'Sem transcrição disponível'
            }
            return 0.0

        try:
            # Faz a predição
            pred_label, probabilities = self.predict_text(transcript)
            
            # Determina se é inapropriado
            is_inappropriate = pred_label == "inapropriado"
            
            # Se encontrou palavras inapropriadas no dicionário, score é 0
            if probabilities['inapropriado'] == 1.0:
                score = 0.0
            else:
                # Caso contrário, usa a probabilidade do modelo
                score = probabilities['apropriado']
            
            # Armazena os resultados
            video_data['inappropriate_score'] = score
            video_data['inappropriate_details'] = {
                'is_inappropriate': is_inappropriate,
                'inappropriate_level': pred_label,
                'probabilities': probabilities,
                'filter_enabled': True
            }
            
            return score

        except Exception as e:
            logger.error(f"Erro ao processar linguagem inapropriada: {e}")
            return 0.0

def get_inappropriate_filter() -> InappropriateFilter:
    """
    Obtém uma instância do filtro de linguagem inapropriada.
    
    Returns:
        Instância do InappropriateFilter
    """
    return InappropriateFilter() 