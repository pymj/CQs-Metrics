
import logging
from typing import Tuple, Union, List
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd

from process_data import *
TEST = False
logger = logging.getLogger(__name__)

model_name = 'all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_name)
tokenizer = sentence_model.tokenizer 
model_options= {'chunk_size': 'max_length', 'chunk_overlap': '256'}
topic_kwargs={'top_n_words': '10',
 'min_cluster_size': '15',
 'n_neighbors': '15',
 'n_components': '5',
 'nr_topics': '5',
 'seed_multiplier': '2.0',
 'diversity':'0.7'}
def initialise_bert_topic(top_n_words: int, min_cluster_size: int, n_neighbors: int, diversity:float,
                          n_components: int, seed_multiplier: float, nr_topics: int) -> BERTopic:
    """
    
    """
    # Initialize models
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    hdbscan_model = HDBSCAN(min_cluster_size=int(min_cluster_size), metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True)
    umap_model = UMAP(n_neighbors=int(n_neighbors), n_components=int(n_components),
                      metric='cosine', min_dist=0.0)
    vectorizer = CountVectorizer(stop_words='english')
    word_extractor = ClassTfidfTransformer(
        seed_words = None,
        seed_multiplier = seed_multiplier
    )
    representation_model = KeyBERTInspired(
        top_n_words=int(top_n_words)
    )
    # diversity =diversity
    bert_topic_model = BERTopic(
        nr_topics=int(nr_topics),
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=word_extractor,
        representation_model=representation_model
    )
    return bert_topic_model
def model_pipeline(cleaned_data,tokeniser, model_options) -> Tuple[pd.DataFrame]:
    """_summary_

    Args:
        cleaned_data (_type_): _description_
        tokeniser (_type_): _description_
        model_options (_type_): _description_

    Returns:
        Tuple[pd.DataFrame]: _description_
    """
    topic_modeler = TopicModels(topic_kwargs)

    def text_chunks(file,
                       tokeniser,
                       max_length=model_options['chunk_size'],
                       overlap=model_options['chunk_overlap']
        ):
        my_file= file.to_list()
        my_file2 = ' '.join(my_file)
        tokens= tokeniser.encode(my_file2, truncation=True, max_length=512)
        chunks = []
        max_length=512
        for i in range(0, len(tokens),max_length):
            chunk = tokens[i:i + max_length]
            chunks.append(tokeniser.decode(chunk[1:-1]))
            print("chunk added")
            if len(chunk) < max_length:
                break
        return chunks

    # def data_chunks(record, sentence_count):
    #     #tid is number of sentences in chunked data
    #     tid = sentence_count
    #     records= pd.DataFrame(record)
    #     records = records.reset_index()
    #     records = records.rename(columns={"index": "old_index_name", 0: "text_col"})
    #     print("records:",records.columns)
    #     text = records['text_col']
    #     chunks = text_chunks(text, tokeniser)
    #     # chunks = [np.array(ce) for ce in chunks] 
    #     n_chunks = len(chunks)
    #     print("length of chunks:",n_chunks)
    #     recorddf = pd.concat([records.T]*n_chunks)
    #     print("record df", len(recorddf))
    #     recorddf['chunk_id'] = [f"{tid}-{c}" for c in range(n_chunks)]
    #     return recorddf
    # clean_data= pd.DataFrame(cleaned_data)
    import pandas as pd

    def data_chunks(record, sentence_count):
        # tid is the number of sentences in chunked data
        tid = sentence_count
        records = pd.DataFrame(record)
        records = records.reset_index()
        records = records.rename(columns={"index": "old_index_name", 0: "text_col"})
    
        print("records:", records.columns)
        text = records['text_col']
    
        # Assuming text_chunks is defined and working properly
        chunks = text_chunks(text, tokeniser)  # Ensure text_chunks is a valid function
        n_chunks = len(chunks)
    
        print("length of chunks:", n_chunks)
        # print("what does chunks look like:", chunks)
        # Create a new DataFrame to hold the chunks
        # records_expanded = []
        # for chunk in chunks:
        #     for text in chunk:  # Assuming chunk is iterable
        #         records_expanded.append([tid, text, f"{tid}"])

        # Create a DataFrame from the expanded records
        lst_chunked =  chunks[0].split(",")
        recorddf = pd.DataFrame(lst_chunked, columns=['text_col'])
    
        print("record df length:", len(recorddf))
        return recorddf

    # Example usage (make sure cleaned_data and sentence_count are defined)
    # cleaned_data could be a list or a DataFrame (specify it accordingly)
    # print("cleaned data", cleaned_data)
    chunked_data = data_chunks(cleaned_data, sentence_count)
    print("chunked data", len(chunked_data))
    # Concatenation depending on how you want to combine results
    # lst_chunked_data = chunked_data['text_col'].tolist()
    all_chunked_data = pd.concat([chunked_data], axis=0)  # If chunked_data is a DataFrame

    
    # chunked_data = data_chunks(cleaned_data,sentence_count)
    # all_chunked_data = pd.concat(
    #     chunked_data,
    #         axis=1
    #     )
   
    def _topic_model_inference(textFile):
        lst_chunked_data = textFile['text_col'].astype(str).tolist()
        # print("length of chunked data list:", lst_chunked_data)
        themes, probs = topic_modeler.get_themes(lst_chunked_data)

        topic_meta = topic_modeler.get_topic_info()

        return themes, probs, topic_meta
    def _postprocess_themes(topic_list, metadata):
        topic_metadata = {}
        topic_metadata['topic']= topic_list
        topic_metadata['Keywords'] = metadata
        return topic_list, topic_metadata
    topics, probs, topic_meta = _topic_model_inference(
        chunked_data
    )
    print("Types of topics and probabilities:")
    print({i: type(theme) for i, theme in enumerate(topics)})
    print({i: type(prob) for i, prob in enumerate(probs)})
    processed_topics, processed_meta = _postprocess_themes(topics, topic_meta)
    print("number of custers:", len(processed_topics))
    try:
        processed_topics = [str(topic) for topic in processed_topics]  # Convert to strings
        probs = [float(prob) for prob in probs]                       # Ensure probabilities are floats
    except ValueError as e:
       raise ValueError(f"Could not convert topics or probabilities: {e}")
    # Ensure the length of processed_topics matches the DataFrame's length.
    if len(processed_topics) != len(chunked_data):
        raise ValueError("The length of processed_topics does not match the length of lst_chunked_data.")
    # Assuming 'probs' must also match the length of 'lst_chunked_data'.
    if len(probs) != len(chunked_data):
        raise ValueError("The length of probs does not match the length of lst_chunked_data.")
    topic_output_df = pd.DataFrame(chunked_data, columns=['text_col']).assign(
        topic_id=processed_topics,
        topic_probability=probs
    )

    return topic_output_df, processed_meta


class TopicModels:
    """
   
    """
    def __init__(self, topic_kwargs):
        self.bert_model = None
        
        self.bert_model = initialise_bert_topic(**topic_kwargs)

    def get_themes(self, textFile):
        """
        Generates themes and their probabilities using the topic modeler.

        Parameters:
            textFile (pd.Series): textfile to generate themes from.

        Returns:
            themes (numpy.array): An array of the themes generated by the topic modeler.
            probs (numpy.array): An array of the probabilities of each topic
                generated by the topic modeler.
        """
        themes, probs = self.bert_model.fit_transform(textFile)
        return themes, probs

    def get_topic_info(self):
        """
        Returns information about the themes generated by the topic modeler.

        Returns:
            pd.DataFrame: A DataFrame containing details about each topic, such as
            topic ID, frequency, and representative keywords.
        """
        return self.bert_model.get_topic_info()
def run_model_pipeline(
    textFile: pd.DataFrame, model_options: dict
    ) -> Tuple[pd.DataFrame]:
    """

    Args:
        textfile (pd.DataFrame): processed intake data
        text_col (str): Column name containing main body
        model_options (dict): Parameters dictionary containing
            - topic_models (list): Name of topic model(s) to use.
            - sentiment_models (list): Name of sentiment model(s) to use.
            - topic_limit (int): Maximum number of themes to allow.

    Returns:
        Tuple[pd.DataFrame]: Main model output containing themes and sentiment per entry,
            and topic metadata dataframe
    """
    if TEST:
        textFile = textFile.iloc[:1000]

    return model_pipeline(textFile, tokenizer, model_options)



# def _split_keywords(keywords: str) -> List[str]:
#     # keywords = df[col_name]
#     # print("keywords:", keywords)
#     # keywords = [kw.strip() for kw in re.sub(r"[\[\]\']", "", str(keywords)).split(',')]
#     keywords = re.sub(r"[\[\]\']", "", keywords)  # Remove brackets and single quotes
#     return [kw.strip() for kw in keywords.split(',') if kw.strip()]
#     # return keywords

def _split_keywords(keywords: Union[str, None]) -> List[str]:
    if isinstance(keywords, str):  # Check if the input is a string
        # Remove brackets and extra punctuation, then split the keywords
        keywords_cleaned = re.sub(r"[\\[\],']", "", keywords)  # Clean the string
        return [kw.strip() for kw in keywords_cleaned.split(',') if kw.strip()]  # Return a list of cleaned keywords
    return [] 

def _reset_topic_indices(themes: pd.Series, offset: int) -> pd.Series:
    return themes + offset

def post_process_results(model_output: pd.DataFrame, metadata: dict,
    ) -> Tuple[pd.DataFrame]:
    """Post processing model results, formatting themes table and sentiment scores

    Args:
        model_output (pd.DataFrame): Cleaned text, themes and sentiments
        metadata (pd.DataFrame): Topic information
    Returns:
        Tuple[pd.DataFrame]: Processed output dataframe, and processed metadata
    """
    # print("type meta data:", metadata)
    # print(type(metadata))
    
    # metadatadf = pd.DataFrame(metadata)
    # print(metadatadf.shape)
    # topics = []
    # keywords = []
    flat_metadata=[]
    for topic, data in metadata.items():
        if isinstance(data, dict):
            # Handle dictionary
            flat_metadata.append({"topic": topic, "Keywords": data.get("Keywords")})
        elif isinstance(data, list):
            # Handle list directly - you might need a different logic based on your use case
            flattened_keywords = ", ".join(str(item) for item in data)  # Convert list to string or process as needed
            flat_metadata.append({"topic": topic, "Keywords": flattened_keywords})


    # Create a flat dictionary
    # flat_metadata = {
    #   "topic": topics,
    #   "Keywords": keywords
    # }

    # Create DataFrame
    metadatadf = pd.DataFrame(flat_metadata)
    print("metadata columns:", metadatadf.columns)
    print("metadata shape:", metadatadf.shape)
    # metadata.reset_index('index', inplace=True)
    print("metadata before applying _split_keywords:")
    # metadata = metadata.rename(columns={"index": "old_index",  0: "topic", 1: "Keywords"})
    print("metadatadf view:", metadatadf)
   
    col = ['Keywords']
    metadatadf['Keywords'] = metadatadf['Keywords'].apply(_split_keywords)
    # Display the modified DataFrame
    # metadata[col] = metadata.apply(_split_keywords, axis=1, col_name=col)
    print("\nmetadata after applying _split_keywords:", metadatadf)
   
    metadatadf = metadatadf.explode(column=col)
    print("metadata:", metadatadf.head())
    print("grouping the keywords to get the last number of clusters")
    topic_offset = metadatadf['topic'].unique()
    # print("unique topics:", topic_offset)
    metadatadf['Topic'] = _reset_topic_indices(metadatadf['topic'], topic_offset)
    model_output['topic_id'] = _reset_topic_indices(model_output['topic_id'], topic_offset)
    print("model output:", model_output)
    print("model output:", metadatadf)
    model_output.to_csv("CQsMetrics/outputs/model_output.csv", sep="|")
    metadatadf.to_csv("CQsMetrics/outputs/metadatadf.csv", sep="|")
    return model_output, metadatadf
# model_output, metadata = post_process_results(model_output, model_output)
# def save_output(data: pd.DataFrame, metadata:pd.DataFrame, output_file: str, path: str) -> None:
#     """Save a dataframe to s3

#     Args:
#         data (pd.DataFrame): Datafrane t0 save
#         output_file (str): output file name 
#         path (str): path
#     """
#     logger.info('Saving to: %s%s', output_file, path)
#     data.to_csv(data, sep='|', index=False)
#     metadata.to_csv(metadata, sep='|', index=False)
    
# save_output(model_output, metadata)