from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import logging
import os
import re

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extrair_video_id(url: str) -> str:
    """
    Extrai o ID do vídeo de uma URL do YouTube.
    
    Args:
        url: URL do vídeo do YouTube
        
    Returns:
        str: ID do vídeo
    """
    # Padrões de URL do YouTube
    padroes = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',  # URLs normais
        r'youtube\.com\/embed\/([^&\n?]+)',  # URLs de embed
    ]
    
    for padrao in padroes:
        match = re.search(padrao, url)
        if match:
            return match.group(1)
    
    raise ValueError("URL do YouTube inválida")

def obter_transcricao(url: str, salvar_arquivo: bool = True) -> str:
    """
    Obtém a transcrição de um vídeo do YouTube.
    
    Args:
        url: URL do vídeo do YouTube
        salvar_arquivo: Se True, salva a transcrição em um arquivo
        
    Returns:
        str: Transcrição do vídeo
    """
    try:
        # Extrai o ID do vídeo
        video_id = extrair_video_id(url)
        logger.info(f"ID do vídeo extraído: {video_id}")
        
        # Obtém a transcrição
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
        
        # Formata a transcrição
        formatter = TextFormatter()
        texto_formatado = formatter.format_transcript(transcript)
        
        # Se solicitado, salva em arquivo
        if salvar_arquivo:
            # Cria o diretório de transcrições se não existir
            transcript_dir = os.path.join(os.getcwd(), "transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Gera o nome do arquivo
            filename = f"transcript_{video_id}.txt"
            filepath = os.path.join(transcript_dir, filename)
            
            # Salva a transcrição
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(texto_formatado)
            
            logger.info(f"Transcrição salva em: {filepath}")
        
        return texto_formatado
        
    except Exception as e:
        logger.error(f"Erro ao obter transcrição: {str(e)}")
        raise

def main():
    """
    Função principal para testar o código.
    """
    # Exemplo de uso
    url = input("Digite a URL do vídeo do YouTube: ")
    try:
        transcricao = obter_transcricao(url)
        print("\nTranscrição obtida com sucesso!")
        print("\nPrimeiros 500 caracteres da transcrição:")
        print("-" * 50)
        print(transcricao[:500])
        print("-" * 50)
    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main() 