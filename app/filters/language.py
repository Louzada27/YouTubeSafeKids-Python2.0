from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.filters.base import BaseFilter
from typing import Dict, Any, List, Tuple
import logging
import os
import traceback
import re


logger = logging.getLogger(__name__)

# Classes de toxicidade (mesmas do test_model.py)
LABEL_COLUMNS = [
    'health', 'ideology', 'insult', 'lgbtqphobia', 'other_lifestyle',
    'physical_aspects', 'profanity_obscene', 'racism', 'religious_intolerance', 'sexism'
]

class LanguageFilter(BaseFilter):
    """Filtro para avaliar a adequação da linguagem do conteúdo."""
    
    def __init__(self):
        super().__init__(
            name="Linguagem",
            description="Avalia o nível de adequação da linguagem do conteúdo do vídeo",
            default_enabled=True
        )
        self.default_value = 50  # Valor padrão para a barra de adequação
        
        self.logger = logging.getLogger("filters.language")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained("GargulaCapixava/ModeloOLYD-BR")
        self.tokenizer = AutoTokenizer.from_pretrained("GargulaCapixava/ModeloOLYD-BR")
        self.model.to(self.device)
        self.model.eval()
        
        # Carrega o dicionário de palavras inadequadas
        self.inappropriate_words = self._load_dictionary()
        
        logger.info("Modelo, tokenizer e dicionário carregados com sucesso")
    
    def _load_dictionary(self) -> List[str]:
        """Carrega o dicionário de palavras inadequadas."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            dict_path = os.path.join(project_root, "app", "nlp", "DicicionarioRegex\dicionario.txt")
            
            with open(dict_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
            
            logger.info(f"Dicionário carregado com {len(words)} palavras")
            return words
        except Exception as e:
            logger.error(f"Erro ao carregar dicionário: {e}")
            return []
    
    def _check_inappropriate_words(self, text: str) -> Tuple[bool, List[str]]:
        """
        Verifica se o texto contém palavras inadequadas do dicionário.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Tuple[bool, List[str]]: (True se encontrou palavras inadequadas, lista de palavras encontradas)
        """
        text = text.lower()
        found_words = []
        
        for word in self.inappropriate_words:
            # Cria um padrão regex que corresponde à palavra com ou sem acentos
            pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            if pattern.search(text):
                found_words.append(word)
        
        return len(found_words) > 0, found_words
    
    def _predict(self, text: str, threshold: float = 0.15) -> Tuple[List[str], List[float]]:
        """
        Faz previsão de toxicidade para um texto.
        
        Args:
            text: Texto a ser analisado
            threshold: Limiar para considerar uma classe como tóxica (0.15 por padrão)
            
        Returns:t
            Tuple[List[str], List[float]]: Classes tóxicas e suas probabilidades
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=False,  # Remove a truncagem para processar o texto completo
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
        
        probs = probabilities[0].cpu().numpy()
        
        toxic_classes = []
        toxic_probs = []
        
        logger.debug(f"Probabilidades para o texto:")
        for i, prob in enumerate(probs):
            logger.debug(f"  {LABEL_COLUMNS[i]}: {prob:.4f}")
            # Ignora a categoria de insultos
            if LABEL_COLUMNS[i] == 'insult':
                continue
                
            # Ajusta o threshold para cada categoria
            if LABEL_COLUMNS[i] == 'profanity_obscene':
                categoria_threshold = 0.1  # Mais sensível para palavrões
            else:
                categoria_threshold = threshold
            
            if prob > categoria_threshold:
                toxic_classes.append(LABEL_COLUMNS[i])
                toxic_probs.append(float(prob))
        
        logger.debug(f"Classes tóxicas encontradas: {toxic_classes}")
        logger.debug(f"Probabilidades: {toxic_probs}")
        
        return toxic_classes, toxic_probs
    
    def get_intensity(self) -> float:
        """
        Retorna o valor atual da barra de intensidade (0-100).
        """
        return self.default_value

    def process(self, video_data):
        """
        Processa os dados do vídeo e retorna o score de adequação da linguagem (float).
        Primeiro verifica palavras inadequadas no dicionário, depois passa pelo modelo se necessário.
        
        Args:
            video_data: Dados do vídeo a serem processados
        
        Returns:
            float: Score de adequação entre 0 e 1
        """
        # Verifica se já foi processado antes
        if 'language_score' in video_data:
            return video_data['language_score']

        # Se o filtro estiver desativado, retorna 1 (aceita o vídeo)
        if not self.is_enabled():
            video_data['language_score'] = 1.0
            video_data['language_details'] = {
                'categories': {},
                'filter_enabled': False
            }
            return 1.0

        # Extrai apenas a transcrição do vídeo (tenta ambas as chaves)
        transcript = video_data.get('transcription', video_data.get('transcript', ''))
        
        # Log para debug
        logger.info(f"Título do vídeo: {video_data.get('title', 'Sem título')}")
        logger.info(f"ID do vídeo: {video_data.get('video_id', 'Desconhecido')}")
        logger.info(f"Chaves disponíveis no video_data: {list(video_data.keys())}")
        logger.info(f"Transcrição disponível: {'Sim' if transcript else 'Não'}")
        logger.info(f"Tamanho da transcrição: {len(transcript)} caracteres")

        # Rejeita vídeos sem transcrição
        if not transcript or not transcript.strip():
            logger.warning("Transcrição vazia ou não disponível - Rejeitando vídeo")
            video_data['language_score'] = 0.0  # Rejeita o vídeo
            video_data['language_details'] = {
                'categories': {},
                'rejection_reason': 'Sem transcrição disponível',
                'filter_enabled': True
            }
            return 0.0  # Retorna 0 para rejeitar o vídeo

        try:
            # Primeiro verifica palavras inadequadas no dicionário
            found_inappropriate, inappropriate_words = self._check_inappropriate_words(transcript)
            
            if found_inappropriate:
                logger.warning(f"Palavras inadequadas encontradas: {inappropriate_words}")
                video_data['language_score'] = 0.0  # Rejeita o vídeo
                video_data['language_details'] = {
                    'categories': {},
                    'inappropriate_words': inappropriate_words,
                    'rejection_reason': 'Palavras inadequadas encontradas no dicionário',
                    'filter_enabled': True
                }
                return 0.0  # Retorna 0 para rejeitar o vídeo
            
            # Se não encontrou palavras inadequadas, passa pelo modelo
            toxic_classes, toxic_probs = self._predict(transcript)
            
            # Calcula o score baseado na maior probabilidade tóxica (exceto insultos)
            if toxic_probs:
                max_toxicity = max(toxic_probs) * 100  # Converte para porcentagem
                logger.info(f"Toxicidade máxima encontrada: {max_toxicity:.2f}%")
                
                # Calcula o score do vídeo (inverso da toxicidade)
                video_score = 100.0 - max_toxicity
                logger.info(f"Score do vídeo (100 - toxicidade): {video_score:.2f}%")
                
                # Normaliza para um valor entre 0 e 1
                normalized_score = video_score / 100.0
                logger.info(f"Score normalizado (0-1): {normalized_score:.4f}")

                # Define thresholds para níveis de toxicidade
                toxicity_level = "baixa"
                if max_toxicity >= 80:  # 80% ou mais de toxicidade
                    toxicity_level = "alta"
                elif max_toxicity >= 50:  # 50% ou mais de toxicidade
                    toxicity_level = "média"

                # Obtém o valor atual da barra de intensidade (0-100)
                intensity = self.get_intensity()
                logger.info(f"Intensidade do filtro: {intensity}%")

              
            
                
                # Armazena os resultados para uso futuro
                video_data['language_score'] = normalized_score
                video_data['language_details'] = {
                    'categories': dict(zip(toxic_classes, toxic_probs)),
                    'toxicity_level': toxicity_level,
                    'video_score': video_score,  # Score do vídeo (0-100)
                    'normalized_score': normalized_score,  # Score normalizado (0-1)
                    'filter_intensity': intensity,
                    'filter_enabled': True,
                    'min_threshold': intensity  # Threshold mínimo aceito
                }

                return normalized_score
            else:
                # Se não encontrou toxicidade, retorna score máximo
                video_data['language_score'] = 1.0
                video_data['language_details'] = {
                    'categories': {},
                    'video_score': 100.0,
                    'normalized_score': 1.0,
                    'filter_enabled': True
                }
                return 1.0
                
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {e}")
            logger.error(traceback.format_exc())
            video_data['language_score'] = 0.0
            video_data['language_details'] = {
                'error': str(e),
                'filter_enabled': True
            }
            return 0.0

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.is_enabled(),
            "type": "language",
            "default_value": self.default_value,
        } 