import os.path
import whisper

def audio2text(filepath):
    model = whisper.load_model("turbo")
    result = model.transcribe(filepath)
    print(result.keys())
    return result["text"]


if __name__ == '__main__':
    file = os.path.dirname(__file__) + "/assets/recording.wav"
    audio2text(file)
