# rag_utils.py
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from googletrans import Translator 
from dotenv import load_dotenv




load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)


def extract_video_id(video_input: str) -> str:
    parsed_url = urlparse(video_input)
    if parsed_url.hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        return parse_qs(parsed_url.query).get("v", [video_input])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return video_input


def fetch_transcript(video_id: str) -> str:
    video_id = extract_video_id(video_id)
    api = YouTubeTranscriptApi()

    try:
        try:
            transcript = api.fetch(video_id, languages=("en",))
            transcript_text = " ".join(snippet.text for snippet in transcript)
            return transcript_text
        except Exception:
            pass

        transcript_list = api.list(video_id)
        transcript = next(iter(transcript_list)).fetch()
        transcript_text = " ".join(snippet.text for snippet in transcript)

        translator = Translator()
        detected_lang = translator.detect(transcript_text[:500]).lang
        if detected_lang != "en":
            transcript_text = translator.translate(transcript_text, src=detected_lang, dest="en").text

        return transcript_text

    except TranscriptsDisabled:
        return "Transcript not available (subtitles disabled)."
    except Exception as e:
        return f"Error fetching transcript: {e}"



def build_vector_store(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def run_rag(video_id: str, question: str):
    transcript_text = fetch_transcript(video_id)
    if transcript_text.startswith("Error"):
        return transcript_text

    vector_store = build_vector_store(transcript_text)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)
    return answer.content
