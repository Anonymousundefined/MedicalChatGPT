import whisper


def transcribe_audio(audio_path, model_name="base.en"):
    """
    Arguments

    :param model_name: Name of the model
    :type audio_path: Path to audio file

    Return:

    string
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]


