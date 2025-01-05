from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType
from dotenv import load_dotenv
from os import getenv

load_dotenv()

configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key=getenv('AUTH')
   )
)

def recognize(audio):
   model = model_repository.recognition_model()

   model.model = 'general'
   model.language = 'ru-RU'
   model.audio_processing_type = AudioProcessingType.Full

   result = model.transcribe_file(audio)
   return result[0].normalized_text




