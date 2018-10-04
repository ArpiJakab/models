
import tensorflow as tf
import asciitree
from tensorflow.python.platform import tf_logging
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from external.sentiment.sentiment_parser import SentimentParser
from external.opion import knowledge as know
from external.opion.movie_review import MovieReview
from external.opion.ecommerce_review import ECReview

import sys
import logging as log
import os
import json
import re

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
    product_specs_file = 'product_specs.json'
    tag_file = 'tags.json'
    for i, arg in enumerate(argv):
        if '=' in arg:
            k,v = arg.split('=')
            if k == '--output_path':
                tag_file = v
            elif k == '--product_specs_path':
                product_specs_file = v

    print 'Output file: ' + tag_file

    tf_logging.set_verbosity(tf_logging.INFO)
    with tf.Session() as sess:
        src = gen_parser_ops.document_source(batch_size=32,
            corpus_name=FLAGS.corpus_name,
            task_context=FLAGS.task_context)
        sentence = sentence_pb2.Sentence()
        sentiment_parser = SentimentParser()

        if '--test' in argv:
            i = 0
            new_review_dir = 'syntaxnet/testdata/sentiment/movie_reviews/new_review/'
            if not os.path.exists(new_review_dir):
                os.mkdir(new_review_dir)

        parser = MovieReview()
        def _parse_tags(review):
            parser.parse_tags(review, tag_file)

        offset = 0

        while True:
            documents, finished = sess.run(src)
            # Each document is a single line of input from review_in.txt
            tf_logging.info('Read %d documents', len(documents))
            found_end = False
            review = know.Miniverse('review')
            for d in documents:
                if '__ECOMMERCE_REVIEWS__' in d:
                    log.debug('eCommerce review parse tags')
                    parser = ECReview(product_specs_file)
                    continue
                elif '__OFFSET__' in d:
                    start = d.find('__OFFSET__')
                    offset = int(re.split('([^0-9])', d[start + 10:])[2])
                    print('sentiment offset', offset)
                    continue
                elif '__REVIEW_END__' in d:
                    found_end = True
                    print 'end of review'
                    _parse_tags(review)
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
                ret = sentence.ParseFromString(d)
                tr = asciitree.LeftAligned()
                sentiment_parser.print_tree(sentence)
                try:
                    phrases = sentiment_parser.parse_phrases(sentence, offset)
                    print('phrases', phrases.__str__())
                    review.add_knowledge(phrases)
                except Exception as e:
                    print 'Sentiment error: ' + e.message

            if not found_end:
                _parse_tags(review)

            if finished:
                break


if __name__ == '__main__':
    # Invoked from standard input, script, interactive prompt
    tf.app.run()
