#!/bin/bash
# Download historical data for instruments from the figi.txt file.
# !! ATTENTION !! The figi.txt file must end with an empty line.
# Otherwise, the data for the last instrument will not be downloaded.
#
# Data saved in the format: data/ticker_figi_1m.csv

# To download the data, you need to get a token.
# Details on obtaining the token can be found at https://tinkoff.github.io/investAPI/token/.
# After obtaining the token, it must be saved in the .env file in the root of the project:
# TOKEN=your_token

# This script based on https://github.com/Tinkoff/investAPI/tree/main/src/marketdata

while IFS== read -r key value; do
  printf -v "$key" %s "$value" && export "$key"
done <../.env

figi_list=figi.txt

current_year=$(date +%Y)
url=https://invest-public-api.tinkoff.ru/history-data
function download {
  local ticker=$1
  local figi=$2
  local year=$3

  # If the folder for saving data does not exist, create it.
  mkdir -p $figi
  local file_name=${figi}/${figi}_${year}.zip
  echo "downloading $figi for year $year"
  local response_code=$(curl -s --location "${url}?figi=${figi}&year=${year}" \
    -H "Authorization: Bearer ${TOKEN}" -o "${file_name}" -w '%{http_code}\n')
  if [ "$response_code" = "200" ]; then
    ((year--))
    download "$ticker" "$figi" "$year"
  fi

  # If the limit of requests per minute (30) is exceeded - repeat the request.
  if [ "$response_code" = "429" ]; then
    echo "rate limit exceed. sleep 5"
    sleep 5
    download "$ticker" "$figi" "$year"
    return 0
  fi

  # If the token is invalid - exit.
  if [ "$response_code" = "401" ] || [ "$response_code" = "500" ]; then
    echo 'invalid token'
    exit 1
  fi

  # If data for the instrument for the specified year is not found.
  if [ "$response_code" = "404" ]; then
    echo "data not found for figi=${figi}, year=${year}, skipped"
    rm $file_name
    unzip -uoq "$figi/*.zip" -d "$figi"

    final_file_name="${ticker}_${figi}_1m.csv"

    cat $figi/*.csv >$final_file_name
    sed -i 's/;/,/g' $final_file_name
    sed -i '1iId,Date,Open,Close,High,Low,Vol,' $final_file_name

    rm -rf $figi

  elif
    [ "$response_code" != "200" ]
  then
    # If another error occurs - just write it to the console and exit.
    echo "unspecified error with code: ${response_code}"
    exit 1
  fi
}

while IFS== read -r ticker figi; do
  rm -rf "$figi/"
  download "$ticker" "$figi" "$current_year"
done <${figi_list}
