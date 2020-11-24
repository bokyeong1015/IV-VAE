#!/bin/bash

StartTime=$(date +%s)

file_id="1UVf4-weEabptvxHOpuWz7BYf31SaGOLk"
file_name="data.tar.gz"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

tar -zxvf ${file_name}
rm ${file_name}

EndTime=$(date +%s)
echo "** $(($EndTime - $StartTime)) sec elapsed"
