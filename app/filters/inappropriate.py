from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.filters.base import BaseFilter
from typing import Dict, Any
import logging
import os
import traceback
import re
from huggingface_hub import login
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=hf_token)


class inappropriateFilter(BaseFilter):
    """Filtro para avaliar a adequação da linguagem do conteúdo (versão binária)."""

    def __init__(self):
        super().__init__(
            name="Linguagem",
            description="Avalia se a linguagem é apropriada ou não.",
            default_enabled=True
        )
        self.default_value = 50  # Valor padrão da barra de adequação

        self.logger = logging.getLogger("filters.language")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")

        # Carrega o modelo binário
        self.model = AutoModelForSequenceClassification.from_pretrained("GargulaCapixava/LinguagemImpropria")
        self.tokenizer = AutoTokenizer.from_pretrained("GargulaCapixava/LinguagemImpropria")
        self.model.to(self.device)
        self.model.eval()

        # Carrega o dicionário
        self.inappropriate_words = self._load_dictionary()

        logger.info("Modelo, tokenizer e dicionário carregados com sucesso.")

    def _load_dictionary(self):
        """Carrega o dicionário de palavras inadequadas."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            dict_path = os.path.join(project_root, "app", "nlp", "DicicionarioRegex", "dicionario.txt")

            with open(dict_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]

            logger.info(f"Dicionário carregado com {len(words)} palavras.")
            return words
        except Exception as e:
            logger.error(f"Erro ao carregar dicionário: {e}")
            return []

    def _check_inappropriate_words(self, text: str):
        """Verifica se há palavras inadequadas."""
        text = text.lower()
        found_words = []

        for word in self.inappropriate_words:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            if pattern.search(text):
                found_words.append(word)

        return len(found_words) > 0, found_words

    def _predict(self, text: str):
        """Predição binária."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        probs = probabilities[0].cpu().numpy()

        apropriada_prob = float(probs[0])
        inapropriada_prob = float(probs[1])

        return apropriada_prob, inapropriada_prob

    def get_intensity(self) -> float:
        """Retorna intensidade atual."""
        return self.default_value

    def process(self, video_data):
        """Processa o vídeo e calcula o language_score."""
        if 'language_score' in video_data:
            return video_data['language_score']

        if not self.is_enabled():
            video_data['language_score'] = 1.0
            video_data['language_details'] = {
                'filter_enabled': False
            }
            return 1.0

        transcript = video_data.get('transcription', video_data.get('transcript', ''))

        if not transcript or not transcript.strip():
            logger.warning("Transcrição vazia ou não disponível - Rejeitando vídeo")
            video_data['language_score'] = 0.0
            video_data['language_details'] = {
                'rejection_reason': 'Sem transcrição disponível',
                'filter_enabled': True
            }
            return 0.0

        try:
            # Verificação por dicionário
            found_inappropriate, inappropriate_words = self._check_inappropriate_words(transcript)

            if found_inappropriate:
                logger.warning(f"Palavras inadequadas encontradas: {inappropriate_words}")
                video_data['language_score'] = 0.0
                video_data['language_details'] = {
                    'inappropriate_words': inappropriate_words,
                    'rejection_reason': 'Palavras inadequadas encontradas no dicionário',
                    'filter_enabled': True
                }
                return 0.0

            # Análise pelo modelo
            apropriada_prob, inapropriada_prob = self._predict(transcript)

            language_score = apropriada_prob  # Score direto da probabilidade de linguagem apropriada

            # Define nível textual
            adequacao_nivel = "Alta" if language_score >= 0.8 else "Média" if language_score >= 0.5 else "Baixa"

            video_data['language_score'] = language_score
            video_data['language_details'] = {
                'apropriada_prob': apropriada_prob,
                'inapropriada_prob': inapropriada_prob,
                'adequacao_nivel': adequacao_nivel,
                'filter_intensity': self.get_intensity(),
                'filter_enabled': True
            }

            return language_score

        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            logger.error(traceback.format_exc())
            video_data['language_score'] = 0.0
            video_data['language_details'] = {
                'error': str(e),
                'filter_enabled': True
            }
            return 0.0

    def get_filter_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o filtro."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.is_enabled(),
            "intensity": self.get_intensity()
        }
