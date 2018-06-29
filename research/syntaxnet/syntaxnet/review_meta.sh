#!/bin/bash

# A script that runs a tokenizer, a part-of-speech tagger and a dependency
# parser on an English text file, with one sentence per line.
#
# parser_eval(tagger) -> parser_eval(parser) -> sentiment
#

PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
MODEL_DIR=syntaxnet/models/parsey_mcparseface
if [ "$1" == "--file" ]; then
    INPUT_FORMAT=file-in
elif [ "$1" == "--conll" ]; then
    INPUT_FORMAT=stdin-conll
elif [ "$1" == "--test" ]; then
    INPUT_FORMAT=test-in
else
    INPUT_FORMAT=stdin
fi

OUTPUT_FORMAT=stdout-conll

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=$OUTPUT_FORMAT \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/tagger-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
   | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/parser-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
  | \
  bazel-bin/syntaxnet/sentiment \
  --task_context=$MODEL_DIR/context.pbtxt \
  --output_path=/Users/arpi/Projects/opion/opion/tags.json \
  $1 \
  --alsologtostderr
