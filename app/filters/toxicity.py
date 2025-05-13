"""
Filtro de toxicidade usando o modelo HateBR.

Este módulo fornece funcionalidades para detectar conteúdo tóxico em textos
usando o modelo HateBR, que classifica o conteúdo em três níveis:
0: Não tóxico
1: Tóxico
2: Discurso de ódio

O score final é a probabilidade de não ser tóxico (classe 0).
"""

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from app.filters.base import BaseFilter
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

# Mapeamento de rótulos para nomes descritivos
LABEL_MAP = {
    0: "não tóxico",
    1: "tóxico",
    2: "discurso de ódio"
}

class ToxicityFilter(BaseFilter):
    """Filtro de toxicidade usando o modelo HateBR."""
    
    def __init__(
        self,
        threshold: float = 0.5
    ):
        """
        Inicializa o filtro de toxicidade.
        
        Args:
            model_path: Caminho para o modelo treinado
            threshold: Limiar para considerar um texto como tóxico
        """
        super().__init__(
            name="Toxicidade",
            description="Avalia o nível de toxicidade do conteúdo do vídeo",
            default_enabled=True
        )
        self.default_value = 50  # Valor padrão para a barra de toxicidade
        self.threshold = threshold
        
        # Carrega o modelo e tokenizer
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("GargulaCapixava/ModeloHateBR")
            self.tokenizer = AutoTokenizer.from_pretrained("GargulaCapixava/ModeloHateBR")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {str(e)}")
            raise
    
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
            "type": "toxicity",
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
        Processa os dados do vídeo e retorna o score de toxicidade.
        
        Args:
            video_data: Dados do vídeo a serem processados
            
        Returns:
            float: Score de toxicidade (0-1)
        """
        # Verifica se já foi processado antes
        if 'toxicity_score' in video_data:
            return video_data['toxicity_score']

        # Se o filtro estiver desativado, retorna 1 (aceita o vídeo)
        if not self.is_enabled():
            video_data['toxicity_score'] = 1.0
            video_data['toxicity_details'] = {
                'is_toxic': False,
                'toxicity_level': 'não tóxico',
                'probabilities': {'não tóxico': 1.0, 'tóxico': 0.0, 'discurso de ódio': 0.0},
                'filter_enabled': False
            }
            return 1.0

        # Extrai apenas a transcrição do vídeo
        transcript = video_data.get('transcription', '')
        
        if not transcript.strip():
            logger.warning("Transcrição vazia ou não disponível - Rejeitando vídeo")
            video_data['toxicity_score'] = 0.0
            video_data['toxicity_details'] = {
                'is_toxic': True,
                'toxicity_level': 'tóxico',
                'probabilities': {'não tóxico': 0.0, 'tóxico': 1.0, 'discurso de ódio': 0.0},
                'rejection_reason': 'Sem transcrição disponível'
            }
            return 0.0

        try:
            # Faz a predição
            pred_label, probabilities = self.predict_text(transcript)
            
      
            score = probabilities['não tóxico']
            
            # Armazena os resultados
            video_data['toxicity_score'] = score
            video_data['toxicity_details'] = {
                'is_toxic': score,
                'toxicity_level': pred_label,
                'probabilities': probabilities,
                'filter_enabled': True
            }
            
            return score

        except Exception as e:
            logger.error(f"Erro ao processar toxicidade: {e}")
            return 0.0
    
    def predict_text(
        self,
        text: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Faz a predição para um texto.
        
        Args:
            text: Texto para classificar
            
        Returns:
            Tupla contendo a classe predita e as probabilidades
        """
        # Tokeniza o texto
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=False  # Remove a truncagem para processar o texto completo
        )
        
        # Faz a predição
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Converte para dict com as classes
        probs = probabilities[0].numpy()
        probabilities_dict = {LABEL_MAP[i]: float(prob) for i, prob in enumerate(probs)}
        
        # Determina a classe predita
        pred_label = max(probabilities_dict.items(), key=lambda x: x[1])[0]
        
        return pred_label, probabilities_dict

def get_toxicity_filter() -> ToxicityFilter:
    """
    Obtém uma instância do filtro de toxicidade.
    
    Returns:
        Instância do ToxicityFilter
    """
    return ToxicityFilter() 
        
