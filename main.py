import librosa
import sys, os, shutil
import numpy as np
from tqdm import tqdm
import subprocess as sp
from scipy import interpolate
from functools import reduce

def get_warping_path(xp, yp):
    interp_func = interpolate.interp1d(xp, yp, kind="linear")
    warping_index = interp_func(np.arange(xp.min(), xp.max())).astype(np.int64)
    # the most left side gives nan, so substitute first index of path
    warping_index[0] = yp.min()

    return warping_index


def stretch(warping_path, num_frames):
    interp_func = interpolate.interp1d(np.arange(0, len(warping_path)), warping_path)
    warping_index = interp_func(np.linspace(0.0, len(warping_path) - 1, num_frames))
    return warping_index


SAMPLE_RATE=16000

def align(ref_filename, warp_filename, warp_frames):
    ref_id = ref_filename.split('.')[0]
    warp_id = warp_filename.split('.')[0]
    ref_audio_filename = "data/" + ref_id + ".wav"
    warp_audio_filename = "data/" + warp_id + ".wav"
    ref_audio, sr = librosa.load(ref_audio_filename, sr=SAMPLE_RATE)
    warp_audio, sr = librosa.load(warp_audio_filename, sr=SAMPLE_RATE)

    n_fft = 4410
    hop_size = 2205

    x_1_chroma = librosa.feature.chroma_stft(y=ref_audio, sr=sr)
    x_2_chroma = librosa.feature.chroma_stft(y=warp_audio, sr=sr)

    D, warping_pairs = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    warping_pairs = np.flip(warping_pairs)
    scaling_factor = warp_frames/warping_pairs[-1:,1]
    scaled_warping_pairs = warping_pairs*scaling_factor
    xp = stretch(scaled_warping_pairs[:,0], warp_frames)
    yp = stretch(scaled_warping_pairs[:,1], warp_frames)
    q2_warping_path = get_warping_path(yp, xp)

    clip_length = 500
    for clip_num in tqdm(range(len(q2_warping_path)//clip_length)):
        selection = q2_warping_path[clip_num*clip_length:(1+clip_num)*clip_length]
        frames_str = reduce((lambda x, y: x + "+" + y), list(map(lambda x: 'eq(n\,' + str(x) + ')', selection)))
        cmd = 'ffmpeg -y -i data/'+warp_filename+' -vf "select='+frames_str+'" -vsync 0 -frame_pts 1 data/warp/f1/%d.bmp'
        sp.check_output([cmd], shell=True)  # , stderr=sp.DEVNULL)
        for i in range(clip_length):
            src = 'data/warp/f1/{}.bmp'.format(selection[i])
            dest = 'data/warp/f2/{}.bmp'.format(i)
            shutil.copy(src, dest)
        filename = 'data/warp/clips/' + str(clip_num) + ".mp4"
        cmd = 'ffmpeg -y -i data/warp/f2/%d.bmp -c:v libx264 -r 25 -pix_fmt yuv420p {}'.format(filename)
        sp.check_output([cmd], shell=True)#, stderr=sp.DEVNULL)
        shutil.rmtree("data/warp/f1/")
        os.mkdir("data/warp/f1/")
        shutil.rmtree("data/warp/f2/")
        os.mkdir("data/warp/f2/")
        with open("clips.txt", "a") as file_object:
            file_object.write("file " + filename + "\n")
    cmd = "ffmpeg -y -f concat -i clips.txt -i {} -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac data/{}.warp.mp4".format(ref_audio_filename, warp_id)
    sp.check_output([cmd], shell=True, stderr=sp.DEVNULL)
    shutil.rmtree("data/warp/clips/")
    os.mkdir("data/warp/clips/")
    os.remove("clips.txt")


if __name__ == '__main__':
    print(sys.argv)
    _, ref_filename, warp_filename, warp_frames = sys.argv
    align(ref_filename, warp_filename, int(warp_frames))

    #warp long to short
    # align('PCicM6i59_I.webm', 'W9u_BYhUxJY.mkv', 4134)

    #warp short to long
    # align('W9u_BYhUxJY.mkv', 'PCicM6i59_I.webm', 7544)
