import tensorflow as tf
import time
import numpy as np
import gradio as gr
from time_series.resnet import Resnet
from llm.llm_output import LLama
from audio.transcription import transcribe_audio
from time_series.data_preprocessing import load_raw_data


def diagnose_ecg(output_directory: str, x_test: np.array) -> tuple:
    """
    Diagnose ECG data using a trained model.

    Parameters:
        output_directory (str): Directory path containing the model file.
        x_test (np.array): Input ECG data as a numpy array.

    Returns:
        tuple: A tuple containing the predicted results and the test duration.
    """
    start_time = time.time()
    model_path = output_directory + "best_model.hdf5"
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    test_duration = time.time() - start_time
    return y_pred, test_duration


def process_input(
    model: LLama, audio: str, text_input: str, file: list, labels: list
) -> str:
    """
    Process the input data and return some text output based on different input sources.

    Parameters:
        model (LLama): The LLama model for generating responses to queries.
        audio (str): Path to the audio file (optional).
        text_input (str): Text input from the user (optional).
        file (list): List of uploaded file paths (optional).
        labels (list): List of labels for ECG data diagnosis.

    Returns:
        str: The generated text output based on the processed input data.
    """
    # Process the input data and generate text output
    if audio:
        transcription = transcribe_audio(audio)
        if text_input:
            query = transcription + "\n" + text_input
        else:
            query = transcription
        output_text = model.response(query=query)
    elif text_input:
        output_text = model.response(query=text_input)
    elif file:
        ecg_data = load_raw_data(file[0].name, 100, "")
        X = np.expand_dims(ecg_data, axis=0)
        output_dir = "time_series/model"
        resnet_model = Resnet(output_dir, input_shape=[1000, 12], n_classes=52)
        prediction, _ = resnet_model.predict(X)
        prediction = prediction[0]
        masked_arr = (prediction > 0.5).astype(int)
        result_list = [labels[i] for i in range(len(masked_arr)) if masked_arr[i] == 1]
        if text_input:
            query = "I have been diagnosed with {} in my Heart ECG. {}?".format(
                " ".join(result_list), text_input
            )
        else:
            query = "I have been diagnosed with {} in my Heart ECG. What does it mean?".format(
                " ".join(result_list)
            )
        output_text = model.response(query=query)

    return output_text


if __name__ == "__main__":
    # Load labels from JSON file
    with open("time_series/labels.json", "r") as f:
        labels = f.read()

    # Create the LLama model for generating responses. Replace your fine-tuned model repository here
    model = LLama("decapoda-research/llama-7b-hf", "Aditya02/Llama_MultiModal_Text")

    # Create the Gradio interface
    iface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.inputs.Audio(type="filepath", label="Upload Audio (Optional)"),
            gr.inputs.Textbox(label="Enter Text (Optional)"),
            gr.inputs.File(
                type="file", label="Upload File (Optional)", count="multiple"
            ),
        ],
        outputs=gr.outputs.Textbox(),
        title="Multi Modal Medical Chatbot",
        description="""This is a basic demo of a multimodal chatbot. Upload your data in any format - audio, text, 
        or tseries and ask your queries.""",
    )

    # Launch the interface on a public link
    iface.queue().launch(debug=True)
