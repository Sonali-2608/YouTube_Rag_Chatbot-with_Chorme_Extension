import asyncio
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
TRANSLATION_CHUNK_SIZE = 3500


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


def chunk_text(text: str, max_chars: int = TRANSLATION_CHUNK_SIZE) -> list[str]:
    chunks = []
    current_words = []
    current_length = 0

    for word in text.split():
        next_length = current_length + len(word) + 1
        if current_words and next_length > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length = next_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


async def translate_text_to_english(text: str, source_language: str) -> str:
    translator = Translator()
    try:
        translated_chunks = []
        for chunk in chunk_text(text):
            translated = await translator.translate(chunk, src=source_language, dest="en")
            translated_chunks.append(translated.text)
        return " ".join(translated_chunks)
    finally:
        await translator.client.aclose()


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

        if transcript.language_code != "en":
            transcript_text = asyncio.run(
                translate_text_to_english(transcript_text, transcript.language_code)
            )

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
