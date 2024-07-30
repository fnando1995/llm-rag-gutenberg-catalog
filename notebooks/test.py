# # """
# # if trained on 512, chunks of 1000 will be truncated always.

# # Train on langers vectors or vectorbase of chunks more reduced. Overlap will help to gather information around. 

# # Try to visualiza de database.

# # As the proyect if for summarization, when tokenizing the prompt? what will happen?

# # """

# # from transformers import AutoTokenizer

# # # Load a pre-trained tokenizer
# # tokenizer = AutoTokenizer.from_pretrained('t5-small')

# # # Sample text (assuming 8000 words)
# # with open('pgc_dataset/20240728173212_processed/African American Writers/book_99.txt') as f:
# #     text = f.read() # Repeat to simulate 8000 words

# # # Tokenize with a max_length of 512
# # tokens = tokenizer(text, max_length=512, truncation=True)

# # # Print the number of tokens
# # print(len(tokens['input_ids']))  # Should print 512
# # print(tokens['input_ids'])  # Print the tokens

# # # If you want to decode the truncated tokens back to text
# # truncated_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
# # print(truncated_text)


from langchain_community.document_loaders import GutenbergLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class RAGPipeline:
    def __init__(self, model_name, device='cpu'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def summarize(self, text, max_length=200):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        summary_ids = self.model.generate(inputs, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def load_book(url):
    loader = GutenbergLoader(url)
    data = loader.load()
    return data

def chunk_text(text, max_chunk_size=1000):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in text.split('. '):
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def summarize_book(book_text, rag_pipeline, chunk_size=1000):
    chunks = chunk_text(book_text, chunk_size)
    summaries = [rag_pipeline.summarize(chunk) for chunk in chunks]
    return ' '.join(summaries)

# Initialize components
model_name = "fnando1995/t5-small-ft-bookSum"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rag_pipeline = RAGPipeline(model_name, device)

# Load and summarize book
book_url = "https://www.gutenberg.org/cache/epub/69972/pg69972.txt"
book_data = load_book(book_url)
book_text = book_data[0].page_content
final_summary = summarize_book(book_text, rag_pipeline)

print(final_summary)
