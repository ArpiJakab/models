
import tensorflow as tf
import asciitree
from tensorflow.python.platform import tf_logging
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from external.sentiment.sentiment_parser import SentimentParser
from external.opion import knowledge as know
from external.opion.movie_review import MovieReview
import sys
import logging as log
import os
import json

reload(sys)
sys.setdefaultencoding('utf8')

log_file = ''.join([os.getcwd(),'/sentiment.log'])
print 'log', log_file
if os.path.exists(log_file):
    os.remove(log_file)

log.basicConfig(filename=log_file,level=log.DEBUG,format='# %(message)s')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task_context',
    'syntaxnet/models/parsey_mcparseface/context.pbtxt',
    'Path to a task context with inputs and parameters for '
    'feature extractors.')
flags.DEFINE_string('corpus_name', 'stdin-conll',
    'Path to a task context with inputs and parameters for '
    'feature extractors.')

log.debug('Start logging ')

def main(argv):
    tag_file = 'tags.json'
    for i, arg in enumerate(argv):
        if '=' in arg:
            k,v = arg.split('=')
            if k == '--output_path':
                tag_file = v

    print 'Output file: ' + tag_file

    tf_logging.set_verbosity(tf_logging.INFO)
    with tf.Session() as sess:
        src = gen_parser_ops.document_source(batch_size=32,
            corpus_name=FLAGS.corpus_name,
            task_context=FLAGS.task_context)
        sentence = sentence_pb2.Sentence()
        sentiment_parser = SentimentParser()
        review = know.Miniverse('review')

        if '--test' in argv:
            i = 0
            new_review_dir = 'syntaxnet/testdata/sentiment/movie_reviews/new_review/'
            if not os.path.exists(new_review_dir):
                os.mkdir(new_review_dir)

        while True:
            documents, finished = sess.run(src)
            tf_logging.info('Read %d documents', len(documents))
            for d in documents:
                if 'MOVIE_REVIEW_END' in d:
                    print 'end of review'
                    MovieReview(review).parse_tags(tag_file)
                    review = know.Miniverse('review')
                    continue

                log.debug('Document: ')
                if '--test' in argv:
                    i += 1
                    with open(new_review_dir + str(i) + '.proto', 'w') as f:
                        f.write(d)
                else:
                    with open('doc.proto', 'w') as f:
                        f.write(d)
                sentence.ParseFromString(d)
                tr = asciitree.LeftAligned()
                sentiment_parser.print_tree(sentence)
                try:
                    phrases = sentiment_parser.parse_phrases(sentence)
                    review.add_knowledge(phrases)
                except Exception as e:
                    print 'Sentiment error: ' + e.message

            if finished:
                break

if __name__ == '__main__':
    # Invoked from standard input, script, interactive prompt
    tf.app.run()
