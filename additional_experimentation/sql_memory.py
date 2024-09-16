import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
import ast
from tqdm import tqdm 
## Initialize Chromadb client to interact with vectordb
client = chromadb.Client()
## Set system prompt to focus on memory
system_prompt=(
    'You are an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any revelant message you have had with the user.'
    'If any embedded previous conversations are attached, use them for context to responding to the user.'
    'If the context is relevant and useful to responding. If the recalled conversations are irrelevant,'
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations.'
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)
convo=[{'role':'system','content':system_prompt}]
convo = []
## credentials to connect to database
DB_PARAMS={
    'dbname':'memory_agent',
    'user':'postgres',
    'password':'aspirine13z',
    'host':'localhost',
    'port':'5432'
}
## Connecting to database
def connect_db():
    conn=psycopg.connect(**DB_PARAMS)
    return conn
## using sql to retireve conversation history for LLM knowledgebase
def fetch_conversations():
    conn=connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations=cursor.fetchall()
    conn.close()
    return conversations
## Insert the conversation into the database
def store_conversation(prompt, response):
    conn=connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
         'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
         (prompt,response)   
        )
        conn.commit()
    conn.close()
## Function for later to remove the row from the database
def remove_last_conversation():
    conn=connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM  conversations)')
        conn.commit()
    conn.close()
## output parsing to give the content only from the response of the LLM
def stream_response(prompt):
    response = ""
    stream = ollama.chat(
        model="llama3", messages=
        convo, stream=True
    )
    print(f"ASSISTANT:")
    
    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)
    print("\n")
    store_conversation(prompt=prompt,response=response)
    convo.append({"role": "assistant", "content": response})
## Store conversations in the vectordb, also connects vectorDB with postgres
def create_vector_db(conversations):
    global vector_db
    vector_db_name = "conversations"
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass
    vector_db = client.create_collection(name=vector_db_name)
    for c in conversations:
        serialized_convo = f"prompt:{c['prompt']} response:{c['response']}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]
        vector_db.add(ids=[str(c["id"])], embeddings=[embedding], documents=[serialized_convo])
## Retrieving the embeddings from the vector database using a list of queries which were synthetically generated
def retrieve_embeddings(queries,results_per_query=2):
    embeddings=set()
    for query in tqdm(queries,desc='Processing queries to vector database'):
        response=ollama.embeddings(model='nomic-embed-text',prompt=query)
        query_embeddings=response['embedding']
        results = vector_db.query(query_embeddings=[query_embeddings], n_results=results_per_query)
        best_embeddings=results['documents'][0]
        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classisfy_embedding(query=query,context=best):
                    embeddings.add(best)
    return embeddings
## Create list of queries based on question asked 
def create_queries(prompt):
    query_msg=(
        'You are a first principle resoning search query AI agent.'
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to it '
        'inorder to correctly respond to the prompt. Your response must be a Python list with no synta errors. '
        'Do not explain anything and do not ever generate anything byt a perfect syntax Python list.')
    query_convo=[
        {'role':'system','content': query_msg},
        {'role':'user','content':'What my name?'},
        {'role':'assistant','content':'["what is my name?","What is the user name?"]'},
        {'role':'user','content':prompt}]
    response=ollama.chat(model='llama3',messages=query_convo)
    print(f'\n Vector database queries: {response["message"]["content"]}\n')
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]
## Classifier to checks whether the retireved message (conversation) is relevant to the question asked or not.
def classisfy_embedding(query,context):
    classify_msg=(
        'You are an embeddings classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no".'
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly '
        'related resond "no". Do not respond "yes" unless the content is highly relevant to the search query. '
    )
    classify_convo=[
        {'role':'system','content':classify_msg},
        {'role':'user','content':f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTENT: You are Skanda Vijaykumar.'},
        {'role':'assistant','content':'yes'},
        {'role':'user','content':f'SEARCH QUERY: Where does user work? \n\nEMBEDDED CONTENT: You work at Nicomatic India Pvt Ltd.'},
        {'role':'assistant','content':'no'},
        {'role':'user','content':f'SEARCH QUERY: {query} \n\nEMBEDDED CONTENT: {context}'}
    ]
    response=ollama.chat(model='llama3',messages=classify_convo)
    return response['message']['content'].strip().lower()
## Function to access the database and fetch memories
def recall(prompt):
    queries=create_queries(prompt=prompt)
    embeddings=retrieve_embeddings(queries=queries)
    convo.append({'role':'user','content':f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message:response embeddings added for context.')
## Call all these functions
conversations=fetch_conversations()
create_vector_db(conversations=conversations)
## Main pipeline with all the controle
while True:
    prompt = input("USER: \n")
    if prompt.lower() == "done":
        break
    else:     
        if prompt[:7].lower()=='/recall':
            prompt=prompt[8:]
            recall(prompt=prompt)
            stream_response(prompt=prompt)
        elif prompt[:7]=='/forget':
            remove_last_conversation()
            convo=convo[:-2]
            print('\n')
            print('Last 2 conversations have been removed.')
        elif prompt[:9].lower()=='/memorize':
            prompt=prompt[10:]
            store_conversation(prompt=prompt,response='Memory stored.')
            print('\n')
        else:
            convo.append({'role':'user','content':prompt})
            stream_response(prompt=prompt)   
