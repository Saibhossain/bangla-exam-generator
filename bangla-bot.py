import json
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from torch.nn.functional import embedding
from transformers.models.auto.image_processing_auto import model_type

with open("/Users/mdsaibhossain/code/python/Bangla Exam Generator/processed_bangla_mcqs.json","r",encoding="utf-8") as f:
    mcq_data = json.load(f)
#print(mcq_data[9])
documents = []

for item in mcq_data:
    content = f"""
        প্রশ্ন: {item['question']}
        অপশন: {', '.join(item['options'])}
        সঠিক উত্তর: {item['answer']}
        ব্যাখ্যা: {item.get('explanation', '')}
        বিষয়: {item['subject']}, শ্রেণি: {item['class']}
        """
    documents.append(Document(page_content=content, metadata=item))

#print(f" loaded {len(documents)} MCQs")

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
vectorstore = FAISS.from_documents(documents, embedding_model)

template = """
তুমি একজন শিক্ষিত শিক্ষক। নিচে একটি প্রশ্ন, অপশন ও উত্তর দেওয়া হয়েছে। অনুরূপ একটি নতুন প্রশ্ন তৈরি করো একই বিষয়ের উপর, তবে বাক্য গঠন ভিন্ন হতে হবে। 

তথ্য:
{context}

✅ নতুন প্রশ্ন তৈরি করো:
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

# ------------------ Step 4: LLM + RetrievalQA ------------------
llm = OllamaLLM(model="gemma3:1b")  # or phi, if you have it downloaded
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ------------------ Step 5: Ask for Similar Question Variant ------------------
while True:
    query = input("\n🔎 প্রশ্ন বা বিষয় লিখুন (exit লিখলে বন্ধ হবে): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print("\n🧠 নতুন প্রশ্ন:\n", result["result"])

    print("\n📄 উৎস প্রশ্ন:\n", result["source_documents"][0].page_content[:300])