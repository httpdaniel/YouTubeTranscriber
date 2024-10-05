from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from pytube import YouTube
from huggingface_hub import InferenceClient
import gradio as gr
from langchain_community.document_loaders import YoutubeLoader

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
client = InferenceClient(model=model_name)


def langhchain_summary(link):
    loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)

    documents = loader.load()

    transcription = " ".join([doc.page_content for doc in documents])
    return transcription


def transcribe_video(url):
    video_id = parse_youtube_url(url)
    if video_id:
        video_metadata = get_video_metadata(video_id)
        # transcript_content = get_transcript_content(video_id)
        transcript_content = langhchain_summary(url)
        transcript_summary = summarise_transcript(transcript_content)
        return (
            f"Title: {video_metadata['title']}\nAuthor: {video_metadata['author']}",
            transcript_content,
            transcript_summary,
        )
    else:
        return None


def parse_youtube_url(url):
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    return None


def get_video_metadata(video_id):
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    title = yt.title or "Unknown"
    author = yt.author or "Unknown"

    metadata = {"title": title, "author": author}

    return metadata


def get_transcript_content(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_content = parse_transcript(transcript)
        return transcript_content
    except Exception as e:
        raise e


def parse_transcript(transcript):
    content = " ".join(
        map(
            lambda transcript_piece: transcript_piece["text"].strip(" "),
            transcript,
        )
    )
    return content


def summarise_transcript(transcript_content):
    prompt = f"""Provide a summary of the following video transcription in 150-350 words, focusing on the key points and core ideas discussed: {transcript_content}"""

    message = [{"role": "user", "content": prompt}]

    result = client.chat_completion(
        messages=message,
        max_tokens=2048,
        temperature=0.1,
    )

    return result.choices[0].message["content"].strip()


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("<H1>YoutTube Transcriber</H1>")
    gr.Markdown(
        "<H3>Provide a link to a YouTube video and get a transcription and summary</H3>"
    )
    gr.Markdown(
        "<H6>This project uses the youtube_transcript_api to fetch a transcript from a YouTube link, pytube to get video metadata, and Mistral 7B to generate a summary.</H6>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_link = gr.Textbox(
                label="Link to video",
                value="https://www.youtube.com/watch?v=ZIyB9e_7a4c",
            )
            transcribe_btn = gr.Button(
                value="Transcribe & Summarise ⚡️", variant="primary"
            )

        with gr.Column(scale=5):
            video_info = gr.Textbox(label="Video Info")
            transcription = gr.TextArea(
                label="Transcription", scale=1, lines=12, max_lines=12
            )
            transcription_summary = gr.TextArea(
                label="Summary", scale=1, lines=12, max_lines=12
            )

    transcribe_btn.click(
        fn=transcribe_video,
        inputs=video_link,
        outputs=[video_info, transcription, transcription_summary],
    )

demo.launch()
