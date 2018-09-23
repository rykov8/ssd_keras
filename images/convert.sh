#!/bin/bash

ls -1 *.png | sed 's/.png//g' | xargs -i convert -quality 90 {}.png {}.jpg
