#!/bin/bash

#initialize variables
USER=""
SERVER=""
REMOTE_PATH=""
LOCAL_PATH=""
FILE="list.txt"

# Parse commandline arguments
while getopts u:s:r:l:f: option; do
    case "${option}" in
        u) USER=${OPTARG};;
        s) SERVER=${OPTARG};;
        r) REMOTE_PATH=${OPTARG};;
        l) LOCAL_PATH=${OPTARG};;
        f) FILE=${OPTARG};;
    esac

done

echo "Copying files from xxxxxxx@xxxxxxxxx:${REMOTE_PATH} to ${LOCAL_PATH}"

while read -r DIR; do

    if [ -d "${LOCAL_PATH}/${DIR}" ]; then
        echo "Directory ${LOCAL_PATH}/${DIR} already exists. Skipping."
        continue
    fi

    scp -r "${USER}@${SERVER}:${REMOTE_PATH}/${DIR}" "${LOCAL_PATH}"

done < "${FILE}"
