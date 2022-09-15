#!/bin/bash

# API_NAME=$1
API_KEY=$1
text=$2
voice=$3

API_NAME="X-Goog-Api-Key"

res=$(curl -s -H "$API_NAME: $API_KEY" \
  -H "Content-Type: application/json; charset=utf-8" \
  --data "{
    'input':{
      'text':'$text'
    },
    'voice':{
      'languageCode':'fr-FR',
      'name':'$voice',
      'ssmlGender':'FEMALE'
    },
    'audioConfig':{
      'audioEncoding':'mp3'
    }
  }" "https://texttospeech.googleapis.com/v1beta1/text:synthesize" | jq '.audioContent')

echo "$res" | tr -d '"' > tmp.txt

# Converting from text to mp3 using base64 encoding
base64 tmp.txt -d > tmp.mp3

# Converting to wave format
ffmpeg -i tmp.mp3 tmp.wav -hide_banner -loglevel error -y

# Removing temporary files
rm tmp.txt
rm tmp.mp3
