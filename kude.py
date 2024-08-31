import PyPDF2
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline
import re

# Ensure that the NLTK data path includes the global location
nltk.data.path.append(r'C:\Users\user\AppData\Roaming\nltk_data')

# Download NLTK data if not already available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ''  # Ensure text is not None
    return text

def preprocess_text(text):
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = text.lower()  # Lowercase
    # Tokenize
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    # Remove stopwords and lemmatize
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed_words = []
    for word_list in words:
        processed_words.append([lemmatizer.lemmatize(word) for word in word_list if word not in stop_words])
    return processed_words

# List of all PDF file paths
pdf_files = [
    r"C:\Users\user\Downloads\5. Adolescent Mental Health Matters Author United Nations International Childrens Emergency Fund.pdf",
    r"C:\Users\user\Downloads\4. Mental Health Problems Among Adolescents Author Marie Dahlen Granrud.pdf",
    r"C:\Users\user\Downloads\3. Look After your Mental Health Using Exercise Author Mental Health Foundation (1).pdf",
    r"C:\Users\user\Downloads\1. Mental Health is for Everyone Author The Durham Community Collective.pdf",
    r"C:\Users\user\Downloads\3. Look After your Mental Health Using Exercise Author Mental Health Foundation.pdf"
]

# To store all extracted and processed text
all_texts = []

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_file)
    processed_text = preprocess_text(text)  # Preprocess if needed
    all_texts.append(processed_text)

# Flatten the processed text
flattened_text = [" ".join([" ".join(sentence) for sentence in text]) for text in all_texts]

# Format the data into a training dataset
def format_for_training(texts):
    # Combine all texts into a single string, with each document separated by a newline
    combined_text = "\n\n".join(texts)
    
    # Write to a file
    with open('training_data.txt', 'w', encoding='utf-8') as f:
        f.write(combined_text)

# Format the data
format_for_training(flattened_text)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the dataset
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset('training_data.txt')

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine-tuned-gpt2')
tokenizer.save_pretrained('fine-tuned-gpt2')

# Load sentiment analysis and intent recognition pipelines
sentiment_analyzer = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
intent_recognizer = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

# Candidate labels for intent recognition
candidate_labels = ["seeking support", "asking for advice", "sharing experiences"]

def generate_gpt2_response(prompt):
    # Tokenize the prompt with padding and truncation
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    # Generate the response with adjusted parameters
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        pad_token_id=tokenizer.pad_token_id,
        max_length=150,  # Increased max_length (optional)
        num_return_sequences=1,
        temperature=0.7,  # Adjust temperature for more varied responses
        top_k=50,  # Limit the sampling pool to the top-k most probable tokens
        top_p=0.9  # Use nucleus sampling with top_p to restrict the pool to a cumulative probability
    )

    try:
        # Attempt decoding
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except (TypeError, ValueError) as e:
        # Handle potential errors during decoding
        print(f"Decoding error: {e}")
        return "An error occurred while generating the response."  # Default message

def generate_combined_response(user_input):
    print(f"User input received in generate_combined_response: {user_input}")  # Debugging: Check user input

    try:
        # Analyze sentiment and recognize intent
        sentiment = sentiment_analyzer(user_input)[0]
        print(f"Sentiment: {sentiment}")  # Debugging: Check sentiment analysis result
        
        intent_result = intent_recognizer(user_input, candidate_labels)
        print(f"Intent result: {intent_result}")  # Debugging: Check intent recognition result

        sentiment_label = sentiment['label']
        intent_label = intent_result['labels'][0]

        # Create a more refined prompt for GPT-2
        prompt = (
            f"The user is feeling {sentiment_label.lower()} and is {intent_label}. "
            f"Given this context, provide a supportive and relevant response to: {user_input}"
        )
        print(f"Generated prompt: {prompt}")  # Debugging: Check the prompt generated for GPT-2

        # Generate a response with GPT-2
        response = generate_gpt2_response(prompt)
        print(f"Generated GPT-2 response: {response}")  # Debugging: Check GPT-2 response
        return response

    except Exception as e:
        print(f"Error in generate_combined_response: {e}")  # Debugging: Print error message
        return "An error occurred while processing your request."
