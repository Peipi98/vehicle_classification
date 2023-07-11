import argparse as ap
import sounddevice as sd
from scipy.io.wavfile import write
from time import time, sleep
import os
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
from datetime import datetime
import uuid
from copy import deepcopy
import zipfile
import paho.mqtt.client as mqtt
import json

'''
    + --------------------------------------------- +
    |                                               |
    |           SETTING GLOBAL VARIABLES            |
    |                                               |
    + --------------------------------------------- +
    
'''



MODEL_NAME = 'model_1600hz2'
MAC_ADDRESS = hex(uuid.getnode())

PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.016,
    'frame_step_in_s': 0.012,
    'num_mel_bins': 40,
    'lower_frequency': 20,
    'upper_frequency': 8000,
    'num_coefficients': 30
}


frame_length = int(PREPROCESSING_ARGS['downsampling_rate'] * PREPROCESSING_ARGS['frame_length_in_s'])
num_spectrogram_bins =  frame_length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    PREPROCESSING_ARGS['num_mel_bins'],
    num_spectrogram_bins,
    PREPROCESSING_ARGS['downsampling_rate'],
    PREPROCESSING_ARGS['lower_frequency'],
    PREPROCESSING_ARGS['upper_frequency']
)

LABELS = [
    "Bus",
    "Minibus",
    "Pickup",
    "Sports Car",
    "Jeep",
    "Truck",
    "Crossover",
    "Car (C Class 4K)"
]

MODEL = dict()

def on_connect(client, userdata, flags, rc):
    '''
    client: mqtt client.
    rc: result code of the connection.
    '''

    print(f'Connected with result code: {str(rc)}')

client = mqtt.Client()
client.on_connect = on_connect
client.connect('mqtt.eclipseprojects.io', 1883)
'''
    + --------------------------------------------- +
    |                                               |
    |           INFERENCE FUNCTIONS                 |
    |                                               |
    + --------------------------------------------- +
    
'''

def allocate_model() -> None:
    '''
    This function extract the model from the zipped version and then it allocate it into a global variable,
    ready to be used into the callback.

    Be careful that if you want to use different models, you have to change the global variable "MODEL_NAME"
    and put the model file ".tflite" or ".zip" into the specified floder in the path.

    Path: './tflite_models/'
    '''

    if not os.path.exists(f"./{MODEL_NAME}.tflite"):
        with zipfile.ZipFile(f"./{MODEL_NAME}.tflite.zip") as zipped_model:
            zipped_model.extract(f"{MODEL_NAME}.tflite")
        zipped_model.close()
    interpreter = tflite.Interpreter(model_path=f"./{MODEL_NAME}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    newModel = {
        'interpreter': interpreter,
        'input_details': input_details,
        'output_details': output_details
    }
    global MODEL
    MODEL = newModel 

def inference(indata):
    '''

    Args:
        • indata: numpy array of the audio.
    Return:
        • predicted_label: Predicted label based on the model output. It can be one from 0 to 7.
    '''

    interpreter = MODEL['interpreter']
    global linear_to_mel_weight_matrix
    downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
    sampling_rate = 44100
    sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
    frame_length = int(downsampling_rate * PREPROCESSING_ARGS['frame_length_in_s'])
    frame_step = int(downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
    num_coefficients = PREPROCESSING_ARGS['num_coefficients']
    
    audio_padded = get_audio_from_numpy(indata)


    if PREPROCESSING_ARGS['downsampling_rate'] != sampling_rate:
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)

    stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.image.resize(mfccs, [32, 32])

    interpreter.set_tensor(MODEL['input_details'][0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(MODEL['output_details'][0]['index'])

    top_index = np.argmax(output[0])
    probability = round(output[0][top_index]*100, 2)
    predicted_label = None

    predicted_label = top_index
    

    return predicted_label, probability

'''
    + --------------------------------------------- +
    |                                               |
    |           SPECTROGRAM FUNCTIONS               |
    |                                               |
    + --------------------------------------------- +
    
'''

def get_audio_from_numpy(indata):
    '''
    Extract the padded audio and map it into a tensor.
    Args:
        • indata: the numpy array of the audio.
    Return:
        • indata: tensor of the audio.
    '''
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * (indata + 32768) / (32767 + 32768) - 1 #2* (..) - 1 done to map from -1 to 1, if you want from 0 to 1 just use only (...)
    indata = tf.squeeze(indata)
    return indata

def get_spectrogram(
    indata, 
    downsampling_rate, 
    frame_length_in_s: int, 
    frame_step_in_s: int):
    '''
    Compute the magnitude of the STFT from numpy array.
    Args :
        • indata: numpy array.
        • downsampling_rate: A Tensor of type int. Sampling rate after downsampling.
        Set equal to the original sampling rate to skip downsampling.
        • frame_length_in_s: int. Frame length (in seconds) of the STFT.
        • frame_step_in_s: int. Frame step (in seconds) of the STFT.
    Returns A tuple of Tensor objects:
        • spectrogram: A Tensor of type float32.
        • sampling_rate: A Tensor of type int32.
    '''
    audio_padded = get_audio_from_numpy(indata)
    sampling_rate = 44100
    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)
        
    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram, downsampling_rate

def is_silence(input, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):
    audio = input
    spectrogram, sampling_rate= get_spectrogram(
        audio,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1

def callback(indata, frames, callback_time, status):
    '''
    InputStream callback, it's called every blocksize.
    '''
    not_silence = is_silence(indata, 44100, 0.016, -140, 0.15)
    
    if not not_silence:
        start = time()
        prediction, probability = inference(indata)
        end = time()
        latency = end - start
        sid = 's301665'
        timestamp_ms = int(time() * 1000)
        label = LABELS[prediction]
        msg = {
        'mac_address': MAC_ADDRESS,
        'timestamp': timestamp_ms,
        'prediction': float(prediction),
        'probability': probability,
        'latency': latency,
        }

        msg = json.dumps(msg) 
        print(f'Sending message to the broker: topic = {sid},  message = {msg}')
        print(f'Label predicted = {label}')
        client.publish(sid, msg)

    else:
        print('No audio detected...')

    
    
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)

    
    args = parser.parse_args()
    allocate_model()


    print('Start Recording...')
    with sd.InputStream( 
        device=args.device, 
        channels= 1, 
        samplerate=44100, 
        dtype='int16', 
        callback=callback, 
        blocksize=(2*44100)
        ):
        while True:
            continue
        