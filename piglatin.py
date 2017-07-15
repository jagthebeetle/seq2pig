def translate(word):
    vowels = 'aeiou'
    if word[0] in vowels:
        return word + 'way'
    else:
        i = 0
        while i < len(word) and word[i] not in vowels:
            i += 1
        return word[i:] + word[:i] + 'ay'
