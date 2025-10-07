from nltk.tokenize import word_tokenize
from nltk.translate import AlignedSent, IBMModel1
from collections import defaultdict
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import math
import numpy as np
from sklearn.model_selection import KFold

def read_lexicon(filename):
    file = open(filename, encoding="utf-8")
    txt = file.read().split('\n')

    parallel_corpus = []
    for line in txt:
        line = line.split("\t")
        if len(line) == 2:
            parallel_corpus.append((line[0], line[1]))
    return parallel_corpus


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


def language_model(parallel_corpus):
    en_sentences = [word_tokenize(en.lower()) for _, en in parallel_corpus]
    train_data, vocab = padded_everygram_pipeline(2, en_sentences)

    lm = Laplace(2)
    lm.fit(train_data, vocab)
    return lm


def beam_search_translate(bm_sentence, bm_to_en, lm, beam_width=5):
    bm_tokens = word_tokenize(bm_sentence.lower())
    beams = [([], "<s>", 1.0)]  # (seq, prev, prob)

    for bm_word in bm_tokens:
        new_beams = []
        en_options = bm_to_en.get(bm_word)

        if not en_options:
            candidates = [w for w in lm.vocab if w not in {"<s>", "</s>"}]  # all words in LM vocabulary
            lm_scores = [(w, lm.logscore(w, [beams[0][1]])) for w in candidates]
            lm_scores.sort(key=lambda x: x[1], reverse=True)
            en_options = {w: math.exp(s) for w, s in lm_scores[:beam_width]}

        for seq, prev, score in beams:
            for en_word, trans_prob in en_options.items():
                lm_prob = math.exp(lm.logscore(en_word, [prev]))
                combined_score = score + math.log(trans_prob) + math.log(lm_prob)
                new_beams.append((seq + [en_word], en_word, combined_score))
        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

    final = []
    for seq, prev, score in beams:
        final_score = score * math.exp(lm.logscore("</s>", [prev]))
        final.append((" ".join(seq), final_score))

    return max(final, key=lambda x: x[1])

def cross_validate_model(parallel_corpus, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    bitext = tokenizer(parallel_corpus)
    folds = list(kf.split(bitext))

    scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_data = [parallel_corpus[i] for i in train_idx]
        test_data = [parallel_corpus[i] for i in test_idx]

        print(f"\nFold {fold_idx + 1}/{k}")
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}\n")

        bitext_train = tokenizer(train_data)
        translation_table = translation_model(bitext_train)
        bm_to_en = flip_translation_table(translation_table)
        lm = language_model(train_data)

        fold_scores = []
        for bm, en_ref in test_data:
            translation, score = beam_search_translate(bm, bm_to_en, lm)
            fold_scores.append(score)
            print(f"Malay: {bm}")
            print(f"Predicted: {translation}")
            print(f"Reference: {en_ref}")
            print(f"Score: {score:.4f}\n")

        avg_score = np.mean(fold_scores)
        scores.append(avg_score)
        print(f"Average fold score: {avg_score:.4f}")

    print(f"\nOverall cross-validation score: {np.mean(scores):.4f}")
    return np.mean(scores)

def main():
    parallel_corpus = read_lexicon('ms_en.txt')
    avg_cv_score = cross_validate_model(parallel_corpus, k=4)
    print("\nFinal average cross-validation score:", avg_cv_score)

if __name__ == "__main__":
    main()

