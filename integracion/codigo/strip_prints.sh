#!/bin/bash

sed '/^\s[^#]*print[(|\s]".*$/d' $1 > $2
