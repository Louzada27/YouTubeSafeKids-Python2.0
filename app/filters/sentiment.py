"""
Filtro de sentimento para análise de vídeos do YouTube.
Usa o modelo BERTimbau treinado para classificar o sentimento dos títulos e descrições dos vídeos.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base import BaseFilter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from huggingface_hub import login
from dotenv import load_dotenv
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=hf_token) 


class SentimentFilter(BaseFilter):
    def __init__(self, 
                 min_score: float = 0.6,
                 allowed_sentiments: Optional[List[str]] = None):
        """
        Inicializa o filtro de sentimento.
        
        Args:
            model_path: Caminho para o modelo treinado
            min_score: Score mínimo para considerar a classificação
            allowed_sentiments: Lista de sentimentos permitidos
        """
        super().__init__(
            name="Sentimento",
            description="Filtra por sentimento do conteúdo",
            default_enabled=True
        )
        
        
        self.model = AutoModelForSequenceClassification.from_pretrained("GargulaCapixava/ModeloLexiconPT")
        self.tokenizer = AutoTokenizer.from_pretrained("GargulaCapixava/ModeloLexiconPT")
        
        # Mapeamento de índices para sentimentos
        self.sentiment_map = {
            0: "Positivo",
            1: "Negativo",
            2: "Neutro"
        }
        
        # Configurações do filtro
        self.min_score = min_score
        self.allowed_sentiments = allowed_sentiments or ["Negativo", "Neutro"]
        
        logger.info("Modelo de sentimento carregado com sucesso!")

    def analyze_text(self, text: str) -> Dict:
        """
        Analisa o sentimento de um texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dict com a análise de sentimento
        """
        # Tokeniza o texto
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Faz a previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            probabilities = predictions[0].tolist()
        
        # Obtém o score final (probabilidade do sentimento predito)
        final_score = probabilities[predicted_label]
        
        # Formata o resultado
        result = {
            "texto": text,
            "sentimento": self.sentiment_map[predicted_label],
            "score_final": final_score,
            "probabilidades": {
                self.sentiment_map[i]: prob for i, prob in enumerate(probabilities)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result

    def process(self, video_data: Dict[str, Any]) -> float:
        """
   
        
        Args:
            video_data: Dicionário com os dados do vídeo
            
        Returns:
            float: Pontuação entre 0 e 1
        """
 
        description_analysis = self.analyze_text(video_data.get('description', ''))
        
   
        description_prob = description_analysis["probabilidades"]["Negativo"]

      
        description_prob = 1 - description_prob

        
     

        
        # Adiciona as análises aos dados do vídeo
        video_data['sentiment_analysis'] = {
            'title': 0,
            'description': description_analysis,
            'combined_positive_prob':  description_prob ,
            'final_score':  description_prob 
        }
        
        return  description_prob 

    def get_filter_info(self) -> Dict:
        """
        Retorna informações sobre o filtro.
        
        Returns:
            Dict com informações do filtro
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "parameters": {
                "min_score": self.min_score,
                "allowed_sentiments": self.allowed_sentiments
            }
        } 