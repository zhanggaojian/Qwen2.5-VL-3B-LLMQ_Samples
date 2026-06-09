#!/bin/bash

# Used to symlink a local models directory to the location of the test
# models on the mount '/prj/qct/mlg/all/ernst/mha2sha-models'.

current_dir=$(basename "$PWD")

if [ "$current_dir" != "test" ]; then
  echo "Error: This script should be run under the 'test' directory"
  exit 1
fi

test_models_dir="/prj/qct/mlg/all/ernst/mha2sha-test-models"

local_models_dir="./python/"

mkdir -p "$local_models_dir"

ln -s "$test_models_dir" "$local_models_dir"

echo "Symlinked directory '$test_models_dir' -> '$local_models_dir/mha2sha-test-models'"
