from themes import *
from process_data import *
from answerability import *
from scope import *
from relevance import *



# pre-processing of pdf data
# file_paths=get_files(base_path)
# outFile= process_file(file_paths)
# outText = readTextFile(outFile)
# rm_extra_spaces, sentence_count = format(outText)
# cleaned_data= rm_non_alphabetic(rm_extra_spaces, exceptions)

# information cluster extraction

model_output, topic_metadata= run_model_pipeline(cleaned_data, model_options)
model_output, metadata = post_process_results(model_output, topic_metadata)