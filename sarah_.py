# -*- coding: utf-8 -*-

from sarah import ParallelData, Seq2Seq, L2Regularizer, Adam, Trainer
from sarah import bleu

# The input files.
src_filename = 'en.txt'
trg_filename = 'de.txt'
# Place to save the model.
model_directory = 'sarah_en-de/'

with open(src_filename) as src_fin:
    with open(trg_filename) as trg_fin:
        data = ParallelData(source=src_fin,
                            target=trg_fin,
                            src_vocab_size=100000,
                            trg_vocab_size=100000,
                            dict_saveto=model_directory)
# Define the architecture.
architecture = Seq2Seq(data,
                       encoder_size=512,
                       decoder_size=512,
                       memory='gru',
                       beam_size=3)

# Define the regularizer.
l2_regularizer = L2Regularizer(rate=5e-3)

# Define the optimizer.
optimizer = Adam(learning_rate=1e-4,
                 gradient_clipping=25
                 regularization=l2_regularizer)

# Actual training...
trainer = Trainer(architecture=architecture,
                  optimizer=optimizer,
                  model_saveto=model_directory)

# Training the model.
trainer.train(num_epoch=10)

# Files that needs translating.
test_filename = 'test.txt' # source language, i.e. EN
gold_filename = 'gold.txt' # target language, i.e. DE

# Loads the trained model, choose the 5th epoch.
translator = Seq2Seq.load(model_directory, epoch=5)

# Actual translating...
with open(test_filename) as test_fin:
    for line in test_fin:
        # Fetch the best translation.
        translation = translator.decode(line)
        print(line + '\t' + trasnslation)
        # Retrieve the n-best translation hypotheses.
        for hypothesis in translator.decode(line, beam_size=3):
            print(line + '\t' + hypothesis)

# Computing BLEU.
with open(gold_filename) as gold_fin:
    references = [[line] for line in gold_fin]

with open(test_filename) as test_fin:
    translations = translator.decode_sents(test_fin)

print('BLEU:', bleu(references, translations)) # supports multi-references by default.
