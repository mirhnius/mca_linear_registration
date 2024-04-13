#!/bin/bash

#initialize variables
USER=""
SERVER=""
REMOTE_PATH=""
LOCAL_PATH=""
FILE="list.txt"

# Parse commandline arguments
while getopts s:r:l:f:m: option; do
    case "${option}" in
        s) SERVER=${OPTARG};;
        r) REMOTE_PATH=${OPTARG};;
        l) LOCAL_PATH=${OPTARG};;
        f) FILE=${OPTARG};;
        m) MODE=${OPTARG};;
    esac

done

echo "Copying files from xxxxxxx@xxxxxxxxx:${REMOTE_PATH} to ${LOCAL_PATH}"

while read -r DIR; do

    if [ "${MODE}" = "dir" ]; then
        if [ -d "${LOCAL_PATH}/${DIR}" ]; then
            echo "Directory ${LOCAL_PATH}/${DIR} already exists. Skipping."
            continue
        fi
        scp -r "${SERVER}:${REMOTE_PATH}/${DIR}" "${LOCAL_PATH}"
    else
        if [ -f "${LOCAL_PATH}/$(basename ${DIR})" ]; then
            echo "File $(basename "${DIR}") already exists. Skipping."
            continue
        fi
        scp "${SERVER}:${DIR}" "${LOCAL_PATH}"
    fi

done < "${FILE}"
