import os
from collections import defaultdict
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import math
from nltk.tokenize import word_tokenize
from nltk.translate import AlignedSent, IBMModel1

def language_model():
    folder_path = "C:/Users/weihe/machinetranslation/reviews"
    all_sentences = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), encoding="utf-8") as f:
            text = f.read().strip()

            for line in text.split("\n"):
                if line.strip():
                    all_sentences.append(word_tokenize(line.lower()))

    n = 2
    train_data, vocab = padded_everygram_pipeline(n, all_sentences)

    lm = Laplace(n)
    lm.fit(train_data, vocab)
    return lm

def load_parallel():
    pairs = []
    en = []
    bm = []
    with open("english_lexicon.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                en.append(line)
    with open("malay_lexicon.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bm.append(line)
    for i in range(len(en)):
        pairs.append((bm[i].lower(), en[i].lower()))
    return pairs


def tokenizer(parallel_corpus):
    bitext = []
    for bm, en in parallel_corpus:
        bm_tokens = word_tokenize(bm.lower())
        en_tokens = word_tokenize(en.lower())
        bitext.append((bm_tokens, en_tokens))
    return bitext


def translation_model(bitext):
    aligned_sents = [AlignedSent(en, bm) for bm, en in bitext]
    ibm1 = IBMModel1(aligned_sents, iterations=10)
    translation_table = ibm1.translation_table
    return translation_table


def flip_translation_table(translation_table):
    bm_to_en = defaultdict(dict)
    for en_word, bm_dict in translation_table.items():
        for bm_word, prob in bm_dict.items():
            bm_to_en[bm_word][en_word] = prob
    return bm_to_en


def beam_search_translate(bm_sentence, bm_to_en, lm, beam_width=5, top_k_lm=5):
    bm_tokens = word_tokenize(bm_sentence.lower())
    beams = [([], "<s>", 0.0)]
    lm_weight = 0.05
    tm_weight = 0.95
    unknown_word_prob = 1e-6

    for bm_word in bm_tokens:
        new_beams = []
        # en_options = bm_to_en.get(bm_word, {"<unk>": unknown_word_prob})

        en_options = bm_to_en.get(bm_word)
        if not en_options:
            en_options = {word: 1.0 for word in lm.vocab if word not in ("<s>", "</s>", ",")}

        for seq, prev, score in beams:
            if len(en_options) > top_k_lm and not bm_to_en.get(bm_word):
                limited_options = list(en_options.items())[:top_k_lm]
            else:
                limited_options = en_options.items()
            for en_word, trans_prob in limited_options:
                lm_prob = math.exp(lm.logscore(en_word, [prev]))
                combined_score = score + lm_weight * math.log(lm_prob) + tm_weight * math.log(
                    trans_prob if trans_prob > 0 else unknown_word_prob)
                new_beams.append((seq + [en_word], en_word, combined_score))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

    final = []
    for seq, prev, score in beams:
        lm_prob = math.exp(lm.logscore("</s>", [prev]))
        final_score = score + lm_weight * math.log(lm_prob)
        final.append((" ".join(seq), final_score))

    best_translation, best_score = max(final, key=lambda x: x[1])
    clean_translation = best_translation.replace("</s>", "").strip()

    return clean_translation, best_score

lm = language_model()
parallel_corpus = load_parallel()
bitext = tokenizer(parallel_corpus)
translation_table = translation_model(bitext)
bm_to_en = flip_translation_table(translation_table)

with open("bm_input.txt", encoding="utf-8") as f:
    txt = f.read().split('\n')
    bm_text = []
    for line in txt:
        if line:
            bm_text.append(line)

for bm_input in bm_text:
    print("Malay:", bm_input)
    translation, score = beam_search_translate(bm_input, bm_to_en, lm)
    print("English:", translation)
    print('\n')