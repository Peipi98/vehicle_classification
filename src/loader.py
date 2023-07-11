#################################################################
# VISC DATA LOADER
# Data should be loaded from the csv splits
# Data are from 3 to 5 seconds, so we should cut the audios around 3-4 seconds (if less, we add a padding)
# Every time we load one audio, we have to:
# - Decode it from wav
# - Check the lenght and sampling rate
# - Adjust the lenght of the audio (cutting it or padding it)
# - so we have to check sample_rate * seconds!
# - then we take the MFCCS or Spectrogram from the data.
##################################################################
from preprocessing import *
import os
from functools import partial
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

############
class VISCDataLoader:
    def __init__(self, seconds,channels, feature_type, downsampling_rate,frame_length_in_s, frame_step_in_s,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None):
                self.sampling_rate = 44100
                self.SHAPE = None
                self.feature_type = feature_type
                self.seconds = seconds #seconds to crop the audio
                self.channels = channels #mono or stereo
                self.PREPROCESSING_ARGS_MFCCS = {
                    'seconds': seconds,
                    'downsampling_rate': downsampling_rate,
                    'frame_length_in_s': frame_length_in_s,
                    'frame_step_in_s': frame_step_in_s,
                    'num_mel_bins': num_mel_bins,
                    'lower_frequency': lower_frequency,
                    'upper_frequency': upper_frequency,
                    'num_coefficients': num_coefficients
                }
                self.PREPROCESSING_ARGS_MEL = {
                    'seconds': seconds,
                    'downsampling_rate': downsampling_rate,
                    'frame_length_in_s': frame_length_in_s,
                    'frame_step_in_s': frame_step_in_s,
                    'num_mel_bins': num_mel_bins,
                    'lower_frequency': lower_frequency,
                    'upper_frequency': upper_frequency,
                }
                self.get_frozen_mfccs = partial(get_mfccs, **self.PREPROCESSING_ARGS_MFCCS)
                self.get_frozen_mel = partial(get_log_mel_spectrogram, **self.PREPROCESSING_ARGS_MEL)
                
#TODO: Function to model samples
# - Take all the data from a csv file
# -
    #THIS FUNCTION IS HERE JUST FOR EXAMPLE
    def get_feature_type(self):
        return self.feature_type

    def preprocess(self, filename):
        if self.feature_type == 'MEL':
            signal, label = self.get_frozen_mel(filename)
        else:
            signal, label = self.get_frozen_mfccs(filename)
        signal.set_shape(self.SHAPE)
        signal = tf.expand_dims(signal, -1)
        signal = tf.image.resize(signal, [32, 32])

        return signal, int(label) - 1

    def spectogram(self, audio):
        return
    
    def melspectogram(self, audio):
        return
    
    def mfccs(self, audio):
        return

    def get_split(self, split):
        ######################
        #Upload data from csv#
        ######################
        df = pd.read_csv(os.path.join('dataset_splits',f'{split}_split.csv'))
        df = list(df.fname.values)
        num_samples = len(df)
        #df = tf.data.Dataset.list_files(df)
        df = tf.data.Dataset.from_tensor_slices(df)
        if split == 'train':
            df = df.shuffle(num_samples)
        
        ######################
        # Preprocessing Data #
        ######################

        #  Adjust signals  #
        #TODO: upload every signal, cropping it and then padding it if needed
        #check if we have to upload it as mono or stereo
        if self.channels == 2:
            #stereo
            #get audio and label
            df = df.map(self.get_audio_and_label_stereo)
        
        #  Convert the audio taking the type of spectogram #
        if self.feature_type == 'MFCCS':
            if self.channels == 2:
                return print('Not Supported')
            else:
                for spectrogram, label in df.map(self.get_frozen_mfccs).take(1):
                    self.SHAPE = spectrogram.shape
                print(self.SHAPE)
                df = df.map(self.preprocess)
                return df
        elif self.feature_type =='MEL':
            if self.channels == 2:
                return print('Not Supported')
            else:
                for spectrogram, label in df.map(self.get_frozen_mel).take(1):
                    self.SHAPE = spectrogram.shape
                print(self.SHAPE)
                df = df.map(self.preprocess)
                return df
            return
        elif self.feature_type =='SPECTOGRAM':
            return
        else:
            raise NotImplementedError("Feature type not supported.")
            
        return df
        
        ####################
        #  Adjust signals  #
        ####################
        ################################
        # PREPROCESSING FUNCTIONS NEW! #
        ################################

        ################
        # UPLOAD AUDIO #
        ################

        ################ MONO ################
    def get_audio_and_label_mono(self, filename: str):
        audio_binary = tf.io.read_file(filename)
        audio, sampling_rate = tf.audio.decode_wav(audio_binary, desired_channels = 1) 
       
        path_parts = tf.strings.split(filename, '/')
        path_end = path_parts[-1]
        file_parts = tf.strings.split(path_end, ' ')
        label = file_parts[0]

        audio = tf.squeeze(audio)
        audio_padded = audio
        ## PAD OR CROP ##
        if self.seconds*sampling_rate - tf.shape(audio) > 0:
            zero_padding = tf.zeros(self.seconds*sampling_rate - tf.shape(audio), dtype=tf.float32)
            audio_padded = tf.concat([audio, zero_padding], axis=0)
        else:
            audio_padded = audio[:int(self.seconds * sampling_rate)]
        
        return audio_padded, sampling_rate, label

    ################ STEREO ################
    def get_audio_and_label_stereo(self, filename: str):
    
        audio_binary = tf.io.read_file(filename)
        print(tf.shape(audio_binary))
        audio, sampling_rate = tf.audio.decode_wav(audio_binary, 2, self.seconds*44100) 
        print(audio.to_tensor().shape[0], sampling_rate)
        print(audio)
        path_parts = tf.strings.split(filename, '/')
        path_end = path_parts[-1]
        file_parts = tf.strings.split(path_end, ' ')
        label = file_parts[0]
        audio_padded = audio
        if self.seconds*sampling_rate - audio.shape[0] > 0:
            zero_padding = tf.zeros([self.seconds*sampling_rate - audio.shape[0],2], dtype=tf.float32)
            audio_padded = tf.concat([audio, zero_padding], axis=0)
        else:
            audio_padded = audio[:int(self.seconds * sampling_rate),:]

        print(audio_padded.shape)
        return audio_padded, sampling_rate, label

    ###########################
    # FEATURES REPRESENTATION #
    ###########################

    def get_spectrogram(
        filename: str, 
        downsampling_rate: int, 
        frame_length_in_s: int, 
        frame_step_in_s: int):

        audio_padded, sampling_rate, label = get_audio_and_label_mono(filename)
    
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
        return spectrogram, self.PREPROCESSING_ARGS['downsampling_rate'], label


    def get_log_mel_spectrogram(
        filename: str, 
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

        spectrogram, sampling_rate, label = get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s)

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
        log_mel_spectrogram, label = get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)

        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients], label
############