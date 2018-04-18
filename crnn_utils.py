
import string


alphabet28 = string.ascii_lowercase + ' _' # 26 is space, 27 is CTC blank char
alphabet87 = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'

def decode(chars):
    blank_char = '_'
    new = ''
    last = blank_char
    for c in chars:
        if (last == blank_char or last != c) and c != blank_char:
            new += c
        last = c
    return new
