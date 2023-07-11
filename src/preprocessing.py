import tensorflow as tf
import tensorflow_io as tfio
from time import time


LABELS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

def get_audio_and_label(filename: str, seconds):
    '''
    Read audio data and its label from a WAV file.
    Args: Path of an audio file.
    Returns A tuple of three Tensor objects:
        • audio: A Tensor of type float32.
        • sampling_rate: A Tensor of type int32.
        • label: A Tensor of type string.
    ''' 
    audio_binary = tf.io.read_file(filename)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary, desired_channels = 1) 
    
    path_parts = tf.strings.split(filename, '/')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, ' ')
    label = file_parts[0]

    audio = tf.squeeze(audio)
    audio_padded = audio
    if seconds*sampling_rate - tf.shape(audio) > 0:
        zero_padding = tf.zeros(seconds*sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio_padded = tf.concat([audio, zero_padding], axis=0)
    else:
        audio_padded = audio[:int(seconds * sampling_rate)]
    
    return audio_padded, sampling_rate, label

def get_spectrogram(
    filename: str,
    seconds, 
    downsampling_rate, 
    frame_length_in_s: int, 
    frame_step_in_s: int):
    '''
    Compute the magnitude of the STFT of a WAV file.
    Args :
        • filename: str. The path of a WAV file.
        • downsampling_rate: A Tensor of type int. Sampling rate after downsampling.
        Set equal to the original sampling rate to skip downsampling.
        • frame_length_in_s: int. Frame length (in seconds) of the STFT.
        • frame_step_in_s: int. Frame step (in seconds) of the STFT.
    Returns A tuple of three Tensor objects:
        • spectrogram: A Tensor of type float32.
        • sampling_rate: A Tensor of type int32.
        • label: A Tensor of type string
    '''
    audio_padded, sampling_rate, label = get_audio_and_label(filename, seconds)
    
    if downsampling_rate != sampling_rate:
        start = time()
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)
        end = time()
        #print((end - start) * 1000)
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

    return spectrogram, downsampling_rate, label


def get_log_mel_spectrogram(
    filename: str,
    seconds, 
    downsampling_rate, 
    frame_length_in_s, 
    frame_step_in_s, 
    num_mel_bins,
    lower_frequency,
    upper_frequency):
    '''
    Compute log-Mel spectrogram of a WAV file.
    Args:
        • filename: str. The path of the file.
        • downsampling_rate: A Tensor of type int. Sampling rate after downsampling.
        Set equal to the original sampling rate to skip downsampling.
        • frame_length_in_s: int. Frame length (in seconds) of the STFT.
        • frame_step_in_s: int. Frame step (in seconds) of the STFT.
        • num_mel_bins: int. Number of Mel bins of the Mel spectrogram
        • lower_frequency: float. Lower bound on the frequencies to be included in the
        Mel spectrum.
        • upper_frequency: float. Upper bound on the frequencies to be included in the
        Mel spectrum.
    Returns A tuple of two Tensor objects:
        • log_mel_spectrogram: A Tensor of type float32.
        • label: A Tensor of type string
    '''

    spectrogram, sampling_rate, label = get_spectrogram(filename,seconds, downsampling_rate, frame_length_in_s, frame_step_in_s)

    #num_spectrogram_bins = spectrogram.shape[1] below the professor code
    sampling_rate_float32 = tf.cast(sampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sampling_rate,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    return log_mel_spectrogram, label


def get_mfccs(
    filename: str,
    seconds,
    downsampling_rate,
    frame_length_in_s,
    frame_step_in_s,
    num_mel_bins,
    lower_frequency,
    upper_frequency,
    num_coefficients):
    '''
    Compute the MFCCs of a WAV file.
    Args 
        • filename: str. The path of the file.
        • downsampling_rate: A Tensor of type int. Sampling rate after downsampling.
        Set equal to the original sampling rate to skip downsampling.
        • frame_length_in_s: int. Frame length (in seconds) of the STFT.
        • frame_step_in_s: int. Frame step (in seconds) of the STFT.
        • num_mel_bins: int. Number of Mel bins of the Mel spectrogram
        • lower_frequency: float. Lower bound on the frequencies to be included in the
        Mel spectrum.
        • upper_frequency: float. Upper bound on the frequencies to be included in the
        Mel spectrum.
        • num_coefficients: float. Number of MFCCs.
    Returns A tuple of two Tensor objects:
        • mfccs: A Tensor of type float32.
        • label: A Tensor of type string.
    '''
    log_mel_spectrogram, label = get_log_mel_spectrogram(filename,seconds, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)

    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients], label