print('-------------------------')
print('Word Embedding:')
print('-------------------------')
import spacy
nlp = spacy.load('en_core_web_lg')
print(nlp(u'lion').vector)
print(len(nlp(u'lion').vector))
print("if we make brave-vector it would seem similar to lion")
print("-------------------------------------------------------------------")
print("""We can also make sentence-vecotr:
it will make the average of the word embedding for each word.""")
print(nlp(u'The quick brown fox jumped over the lazy dogs.').vector)
print("But sentence-vector is often useless, we will proof it with the tool similarity:")
print("-----------------------------------------------------------------------------")
print("""The important use is that we can calculate the similarity between the words,
 so if we choose three words such as cat, lion and dog, then the similarity values between them
  can be calculated by the command similarity""")
tokens = nlp(u'lion cat pet')
for token1 in tokens:
    for token2 in tokens:
        print("Similarity between ", token1.text," and ", token2.text,": ", token1.similarity(token2))
print("-------------------------------------------------------------------")
print("Similarity between Lion and dandelion: ", nlp(u'lion').similarity(nlp(u'dandelion')))
print("Similarity between Lion and dandelion: ", nlp(u'lion').similarity(nlp(u'tiger')))
print("So we can see that the similarity is independent on the similarity in letters.")
print("--------------------------------------------------------------------------------------------")
print("Similarity between lion and man: ", nlp(u'lion').similarity(nlp(u'man')))
print("Similarity between lion and woman: ", nlp(u'lion').similarity(nlp(u'woman')))
print("man is more similar to lion than woman.")
print("--------------------------------------------------------------------------------------------")
print("Similarity between love and hate: ", nlp(u'love').similarity(nlp(u'hate')))
print("Similarity love lion and flower: ", nlp(u'love').similarity(nlp(u'flower')))
print("love is similar to hate that to flower because they have more common features.")
print("--------------------------------------------------------------------------------------------")
print("Why sentence-vector and embedding is useless: ")
print("Similarity: I love school Vs. I hate school:",nlp(u'I love school').similarity(nlp(u'I hate school')))
print("so if we want to compare between to sentence, we have to make SA(sentimental analysis) not word embedding.")
print("--------------------------------------------------------------------------------------------")
print("the number of words in spacy-dictionary: ",len(nlp.vocab.vectors))
print("the number of features for each word: ",nlp.vocab.vectors.shape)
print("to see if a word is included in the dictionary: ")
tokens = nlp(u'dog cat nargle hesham')
for token in tokens:
    print("word:",token.text,"| included?",token.has_vector,"| norm:",token.vector_norm,"| not included?",token.is_oov)
print("--------------------------------------------------------------------------------------------")
print("How to do calculations on words: ")
from scipy import spatial
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector
# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []
words = ['cat','apple','queen','castle','sea','shell','orange','phone' , 'tiffany'
         ,'angry','book','white','land','study','crown','prince','dog',
         'great','princess','elizabeth','wow','eat','dead','horrible']
for word in words:
    similarity = cosine_similarity(new_vector,nlp.vocab[word].vector)
    computed_similarities.append((word, similarity))
#we order words with sorted, and key=lambda item: -item[1] to turn the ordinary.
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
#show the most similar ten words.
print("Which words are similar to the operation: king - man + woman: ")
for a,b in computed_similarities[:10] :
    print(f'Word {a} , has similarity {b}')


