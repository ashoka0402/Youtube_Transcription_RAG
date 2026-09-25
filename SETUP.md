# YT Lecture RAG — Setup Guide

This guide explains how to install, run, and extend YT Lecture RAG.

The application is built around this pipeline:

~~~text
YouTube playlist
      ↓
download audio
      ↓
faster-whisper transcription
      ↓
cached transcript JSON
      ↓
time-aware chunking
      ↓
local embeddings
      ↓
Qdrant vector index
      ↓
search / grounded answer
~~~

The important design choice is that transcripts and the vector index are separate from the query application. That makes it possible to add new lectures later without throwing away the existing corpus.

---

# 1. Prerequisites

Install or have access to:

- Python 3.10+
- Git
- Internet access
- Enough disk space for model downloads, transcripts, and temporary audio
- A CUDA-capable GPU if you want faster Whisper transcription; CPU fallback is supported

For generated answers, configure either Groq or Google Gemini.

The search-only path does not require an LLM API key.

---

# 2. Clone the repository

~~~bash
git clone https://github.com/ashoka0402/Youtube_Transcription_RAG.git
cd Youtube_Transcription_RAG
~~~

---

# 3. Create a Python environment

## Windows

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

If PowerShell blocks activation:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

## Linux / macOS

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

---

# 4. Install dependencies

The repository contains a uv.lock but currently does not contain a pyproject.toml, so the most predictable setup is to install the runtime packages directly.

~~~bash
pip install fastapi uvicorn typer rich python-dotenv groq google-genai qdrant-client sentence-transformers faster-whisper yt-dlp
~~~

Verify the important imports:

~~~bash
python -c "import fastapi, qdrant_client, sentence_transformers, faster_whisper, yt_dlp; print('dependencies OK')"
~~~

---

# 5. Configure .env

The application reads configuration from the repository-root .env file.

For generated answers, add one provider key.

## Gemini

~~~dotenv
GEMINI_API_KEY=your_key_here
~~~

## Groq

~~~dotenv
GROQ_API_KEY=your_key_here
~~~

You can explicitly select the backend:

~~~dotenv
YTRAG_LLM_BACKEND=gemini
~~~

or:

~~~dotenv
YTRAG_LLM_BACKEND=groq
~~~

or disable LLM generation:

~~~dotenv
YTRAG_LLM_BACKEND=none
~~~

When YTRAG_LLM_BACKEND is omitted, the current code automatically prefers:

~~~text
Gemini key present → Gemini
otherwise Groq key present → Groq
otherwise → none
~~~

For a basic local Qdrant setup, leave these empty:

~~~dotenv
QDRANT_URL=
QDRANT_API_KEY=
~~~

That makes the application use an embedded local Qdrant store under ~/.ytrag/qdrant by default.

### Recommended starting configuration

~~~dotenv
GEMINI_API_KEY=your_key_here
YTRAG_LLM_BACKEND=gemini

YTRAG_WHISPER_LANG=en
YTRAG_EMBED_MODEL=all-MiniLM-L6-v2

QDRANT_URL=
QDRANT_API_KEY=
~~~

Do not commit real API keys or exported browser cookies.

---

# 6. Verify the installation

Run:

~~~bash
python main.py --help
~~~

The CLI currently includes commands such as:

~~~text
ingest
reindex
ask
search
progress
stats
eval
langtest
preflight
export-vectors-cmd
load
export-transcripts
clean-audio
serve
~~~

Then check the current index:

~~~bash
python main.py stats
~~~

---

# 7. First run using the existing repository data

This repository already contains a transcript snapshot and a prebuilt vector export.

You do not need to download and transcribe the original playlist just to run the application.

## Option A — load the prebuilt index

Run:

~~~bash
python main.py load
~~~

Then:

~~~bash
python main.py serve
~~~

Open:

http://127.0.0.1:8000

## Option B — rebuild the index from bundled transcripts

Use this when you want to test different chunking or embedding settings:

~~~bash
python main.py reindex --transcripts transcripts --replace
~~~

Then:

~~~bash
python main.py serve
~~~

The second option performs embedding locally, so it is slower than loading the prebuilt vector export.

---

# 8. Test retrieval before testing the LLM

Always test retrieval first:

~~~bash
python main.py search "binary search"
~~~

This shows the actual ranked chunks and their distances.

Then test the generated answer path:

~~~bash
python main.py ask "binary search ki time complexity kya hoti hai?"
~~~

Recommended debugging order:

~~~text
search
  ↓
check lecture + timestamp
  ↓
ask
  ↓
check grounded answer + citations
  ↓
serve
  ↓
test browser UI
~~~

---

# 9. How to add a NEW YouTube playlist

This is the workflow to use when you want to extend the current corpus.

## Important: do not replace the existing playlist

A video is identified by its YouTube video ID, and each chunk is derived from the video ID plus its starting second.

Therefore:

~~~text
Existing playlist
       +
New playlist
       ↓
same Qdrant collection
       ↓
larger searchable corpus
~~~

You do not need to delete the old transcripts or create a second collection just to add another playlist.

---

## Step 1 — test one new lecture

Before processing a large playlist, test the transcription language on one lecture:

~~~bash
python main.py langtest "NEW_LECTURE_URL"
~~~

The default comparison is English versus Hindi for the first 180 seconds.

For a longer comparison:

~~~bash
python main.py langtest "NEW_LECTURE_URL" --seconds 300 --languages en,hi
~~~

Check technical terms carefully, then set the selected language in .env.

Example:

~~~dotenv
YTRAG_WHISPER_LANG=en
~~~

---

## Step 2 — ingest the new playlist

Run:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

Example:

~~~bash
python main.py ingest --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
~~~

The ingest flow is:

~~~text
new playlist
   ↓
resolve videos
   ↓
download best audio
   ↓
transcribe
   ↓
write transcript cache
   ↓
time-aware chunks
   ↓
local embeddings
   ↓
Qdrant upsert
~~~

This adds the new videos to the existing local index.

---

# 10. If the new playlist is large, test first

Start with a small sample:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL" --limit 3
~~~

Then inspect:

~~~bash
python main.py stats
~~~

Test one topic from the new playlist:

~~~bash
python main.py search "TOPIC FROM NEW PLAYLIST"
~~~

If the retrieved lecture and timestamp look correct, run the full playlist:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

---

# 11. What happens if the new playlist overlaps with existing videos?

The ingest logic uses cached transcripts and already-indexed video IDs.

If a video has already been processed, it can be skipped instead of being downloaded, transcribed, and embedded again.

That makes re-running the same playlist safe.

For example, this is safe after an interruption:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

The command resumes using the work that is already cached.

---

# 12. Monitor a long ingest

In another terminal:

~~~bash
python main.py progress
~~~

Or compare against the playlist total:

~~~bash
python main.py progress --playlist "NEW_PLAYLIST_URL"
~~~

Inspect the vector index:

~~~bash
python main.py stats
~~~

The main numbers to watch are:

~~~text
cached transcripts
indexed chunks
indexed videos
~~~

---

# 13. If the network drops

The ingestion code distinguishes network outages from ordinary video failures.

When connectivity disappears, it can wait for the network and retry the current video.

If the run eventually stops, simply run the same command again:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

Already completed transcript work remains cached, so the run does not need to start from zero.

---

# 14. If YouTube requires cookies

A cookies file can be configured with:

~~~dotenv
YTRAG_COOKIES_FILE=C:\path\to\cookies.txt
~~~

The code also supports browser extraction:

~~~dotenv
YTRAG_COOKIES_FROM_BROWSER=firefox
~~~

A cookies file takes precedence in the current implementation.

Treat cookies as credentials and never commit them.

---

# 15. Verify the new playlist after ingestion

First check the index:

~~~bash
python main.py stats
~~~

Then search for a topic that definitely exists in the new playlist:

~~~bash
python main.py search "TOPIC FROM NEW PLAYLIST"
~~~

Confirm:

- the expected lecture appears
- the timestamp is reasonable
- the transcript preview matches the topic

Then test the grounded answer:

~~~bash
python main.py ask "QUESTION ABOUT NEW PLAYLIST"
~~~

Finally run the web UI:

~~~bash
python main.py serve
~~~

---

# 16. Changing chunking or the embedding model

Do not re-transcribe just because you changed chunking or embeddings.

The transcript cache is the reusable source artifact.

For example, after changing:

~~~dotenv
YTRAG_CHUNK_SECONDS=90
YTRAG_CHUNK_OVERLAP=20
~~~

rebuild the index:

~~~bash
python main.py reindex --replace
~~~

After changing the embedding model:

~~~dotenv
YTRAG_EMBED_MODEL=NEW_MODEL
~~~

rebuild again:

~~~bash
python main.py reindex --replace
~~~

This reuses the transcript files and avoids downloading/transcribing audio again.

---

# 17. Adding a playlist vs reindexing vs retranscribing

These operations have different purposes.

## Add new lecture data

Use:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

This is additive.

## Rebuild embeddings/chunks from cached transcripts

Use:

~~~bash
python main.py reindex --replace
~~~

This is for changing chunking, embedding configuration, or rebuilding an index.

## Force a fresh transcription

Use:

~~~bash
python main.py ingest --playlist "PLAYLIST_URL" --force
~~~

Only use --force when you intentionally want fresh Whisper output.

---

# 18. Update the repository after adding a new playlist

Your local runtime index and the repository snapshot are different things.

The runtime index normally lives under:

~~~text
~/.ytrag/qdrant/
~~~

The shareable repository artifacts are:

~~~text
transcripts/
index/vectors.npz
~~~

If you only ingest locally, your machine will know about the new playlist but another clone of the repository will not.

To update the repository snapshot:

## Export transcripts

~~~bash
python main.py export-transcripts
~~~

## Export vectors

~~~bash
python main.py export-vectors-cmd
~~~

The vector export defaults to:

~~~text
index/vectors.npz
~~~

## Verify

~~~bash
python main.py stats
python main.py search "TOPIC FROM NEW PLAYLIST"
~~~

## Commit

~~~bash
git status
git add transcripts/ index/vectors.npz
git commit -m "data: add new lecture playlist"
git push
~~~

---

# 19. Complete workflow for every new playlist

Use this sequence when extending the project:

~~~bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Test one lecture's transcription language
python main.py langtest "NEW_LECTURE_URL"

# 3. Ingest a small sample
python main.py ingest --playlist "NEW_PLAYLIST_URL" --limit 3

# 4. Inspect the corpus
python main.py stats

# 5. Validate retrieval
python main.py search "TOPIC FROM NEW PLAYLIST"

# 6. Validate grounded generation
python main.py ask "QUESTION ABOUT NEW PLAYLIST"

# 7. Ingest the complete playlist
python main.py ingest --playlist "NEW_PLAYLIST_URL"

# 8. Verify again
python main.py stats
python main.py search "ANOTHER TOPIC FROM NEW PLAYLIST"

# 9. Update repository artifacts
python main.py export-transcripts
python main.py export-vectors-cmd

# 10. Commit the updated corpus
git add transcripts/ index/vectors.npz
git commit -m "data: add new lecture playlist"
git push

# 11. Start the web UI
python main.py serve
~~~

---

# 20. Local runtime directories

By default:

~~~text
~/.ytrag/
├── audio/
│   └── temporary downloaded audio
├── transcripts/
│   └── VIDEO_ID.json
├── qdrant/
│   └── embedded Qdrant database
└── langtest/
    └── language comparison output
~~~

The repository's transcripts/ and index/vectors.npz are shareable snapshots.

---

# 21. Clean downloaded audio

After successful transcription, audio is normally no longer required.

Remove downloaded audio:

~~~bash
python main.py clean-audio
~~~

This does not remove transcripts or the vector index.

---

# 22. Hosted Qdrant

For local development, leave QDRANT_URL empty.

For a hosted Qdrant cluster:

~~~dotenv
QDRANT_URL=https://YOUR-QDRANT-ENDPOINT
QDRANT_API_KEY=YOUR_API_KEY
~~~

Hosted Qdrant is useful when:

- the application has multiple processes
- you need remotely persistent storage
- you want to deploy the application publicly

The current collection name includes the embedding dimension, which prevents accidental mixing of differently sized vector spaces.

---

# 23. Preflight before a serious ingestion

Run:

~~~bash
python main.py preflight
~~~

Optionally check a playlist too:

~~~bash
python main.py preflight --playlist "PLAYLIST_URL"
~~~

This exercises the major dependencies and real code paths before a long run.

The current preflight also checks configured provider services, so an otherwise valid search-only/local setup may still report failures for provider-specific checks that are not required for retrieval.

---

# 24. Evaluation

Run:

~~~bash
python main.py eval
~~~

Verbose output:

~~~bash
python main.py eval --verbose
~~~

Different retrieval depth:

~~~bash
python main.py eval --k 10
~~~

The evaluation cases live in:

~~~text
eval/golden.json
~~~

For a new playlist, add retrieval cases such as:

~~~json
{
  "q": "How does sliding window work?",
  "expect_video_id": "VIDEO_ID",
  "expect_around_sec": 724,
  "tolerance_sec": 120
}
~~~

Then rerun:

~~~bash
python main.py eval --k 5
~~~

Keep refusal cases as well. Adding more lecture data should not accidentally turn unrelated questions into confident answers.

---

# 25. Troubleshooting

## Missing LLM key

For Gemini:

~~~dotenv
GEMINI_API_KEY=...
YTRAG_LLM_BACKEND=gemini
~~~

For Groq:

~~~dotenv
GROQ_API_KEY=...
YTRAG_LLM_BACKEND=groq
~~~

For retrieval-only mode:

~~~dotenv
YTRAG_LLM_BACKEND=none
~~~

The search command still works without an LLM.

## Vector model mismatch

If loading a prebuilt vector export reports that it was built with a different embedding model, make the model match or rebuild:

~~~bash
python main.py reindex --replace
~~~

## Local Qdrant already in use

Stop the other process using the local embedded store, especially another running server.

The usual command to stop is Ctrl+C in the terminal running:

~~~bash
python main.py serve
~~~

For multiple concurrent readers/processes, use hosted Qdrant.

## CUDA problems

The default Whisper device is auto.

The code tries CUDA when available and can fall back to CPU int8.

To force CPU:

~~~dotenv
YTRAG_WHISPER_DEVICE=cpu
YTRAG_WHISPER_COMPUTE=int8
~~~

## YouTube download problems

Check:

1. internet access
2. video visibility/restrictions
3. authentication requirements
4. cookies configuration
5. YouTube rate limiting

For restricted/authenticated content, configure a cookies file.

---

# 26. Safe maintenance rules

**Adding a playlist:** use ingest.

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

**Changing chunking or embeddings:** use reindex.

~~~bash
python main.py reindex --replace
~~~

**Forcing new transcription:** use --force deliberately.

~~~bash
python main.py ingest --playlist "PLAYLIST_URL" --force
~~~

**Updating the shareable repository snapshot:** export transcripts and vectors.

~~~bash
python main.py export-transcripts
python main.py export-vectors-cmd
~~~

**Testing retrieval:** use search before ask.

~~~bash
python main.py search "QUERY"
~~~

---

# 27. The one command to remember

When you have a completely new playlist and want to add it without deleting the existing lecture corpus:

~~~bash
python main.py ingest --playlist "NEW_PLAYLIST_URL"
~~~

Then verify:

~~~bash
python main.py stats
python main.py search "TOPIC FROM NEW PLAYLIST"
~~~

And when you want the new data to ship with the GitHub repository:

~~~bash
python main.py export-transcripts
python main.py export-vectors-cmd
~~~

The mental model is:

~~~text
ingest   = ADD new lecture data
reindex  = REBUILD the index from transcripts
export   = UPDATE the repository's shared snapshot
load     = RESTORE the repository's shared snapshot
~~~
