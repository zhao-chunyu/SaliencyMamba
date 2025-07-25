#!/bin/bash

# MATLAB Executable file path.（Please replace it !!!!）
MATLAB_BIN="/home/lwb/matlab/bin/matlab"

# MATLAB Script path
SCRIPT_PATH="caculate.m"  # default 

# Parameter file path
PARAM_FILE="params.txt"  # default

# read params.txt ----> parameter array
mapfile -t params < "$PARAM_FILE"

# parameter array ----> MATLAB Array of units : {'param1','param2',...}
param_string=$(printf "'%s'," "${params[@]}")
param_string=${param_string%,}

# echo "Processing parameters: {$param_string}" # print parameters

# run MATLAB.(Record logs for debugging purposes)
$MATLAB_BIN -nodisplay -nosplash -nodesktop -logfile "matlab.log" -r "calculate({$param_string}); exit;"