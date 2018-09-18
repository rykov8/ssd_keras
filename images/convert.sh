#!/bin/bash

alias foreach="bash <(echo 'while read LINE; do \"\${@//LINE/\$LINE}\"; done')"

ls -1 *.png | sed 's/.png//g' | foreach convert -quality 90 LINE.png LINE.jpg
