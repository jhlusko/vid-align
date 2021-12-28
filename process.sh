#!/bin/bash

ref_id=$1
warp_id=$2

cd data

for file in $ref_id $warp_id; do
  #download
  youtube-dl -o '%(id)s.%(ext)s' $file

  #resample video
  ffmpeg -i $file.* -filter:v fps=25 $file.mp4

  #extract audio
  ffmpeg -i $file.mp4 -q:a 0 -map a $file.wav

  #resample audio
  sox $file.wav tmp.wav rate 16000
  mv tmp.wav $file.wav
done

ref_count=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $ref_id.mp4)
cd ..
touch clips.txt
mkdir -p data/warp/clips
mkdir -p data/warp/f1
mkdir -p data/warp/f2
python main.py $ref_id.mp4 $warp_id.mp4 $ref_count