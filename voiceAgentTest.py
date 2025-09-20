# import os
# import time
# import threading
# import json
# import queue
# import numpy as np
# import sounddevice as sd
# import asyncio

# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import JSONResponse

# from deepgram import (
#     DeepgramClient,
#     DeepgramClientOptions,
#     AgentWebSocketEvents,
#     AgentKeepAlive,
# )
# from deepgram.clients.agent.v1.websocket.options import SettingsOptions
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # or ["*"] to allow all (not recommended for production)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Globals for managing the agent thread and WebSocket clients
# agent_thread = None
# agent_running = False
# audio_q = queue.Queue()
# transcript_queue = asyncio.Queue()
# clients = set()

# # Store the Deepgram connection globally to finish it later
# connection = None

# # Store main asyncio event loop here after app startup
# main_event_loop = None


# @app.on_event("startup")
# async def startup_event():
#     global main_event_loop
#     main_event_loop = asyncio.get_running_loop()
#     print("🌟 FastAPI startup complete, event loop stored.")


# def deepgram_agent():
#     global connection, agent_running

#     try:
#         api_key = os.getenv("DEEPGRAM_API_KEY", "22496a86ae68818740d8dee64bf41b3d6a70a186")
#         if not api_key:
#             raise ValueError("Missing DEEPGRAM_API_KEY")

#         print(f"✅ Using API key: {api_key[:6]}...")

#         config = DeepgramClientOptions(options={"keepalive": "true"})
#         deepgram = DeepgramClient(api_key, config)
#         connection = deepgram.agent.websocket.v("1")

#         options = SettingsOptions()
#         options.audio.input.encoding = "linear16"
#         options.audio.input.sample_rate = 24000
#         options.audio.output.encoding = "linear16"
#         options.audio.output.sample_rate = 24000
#         options.audio.output.container = "wav"
#         options.agent.language = "en"
#         options.agent.listen.provider.type = "deepgram"
#         options.agent.listen.provider.model = "nova-3"
#         options.agent.think.provider.type = "open_ai"
#         options.agent.think.provider.model = "gpt-4o-mini"
#         options.agent.think.prompt = "You are a friendly AI assistant."
#         options.agent.speak.provider.type = "deepgram"
#         options.agent.speak.provider.model = "aura-2-thalia-en"
#         options.agent.greeting = "Hello! How can I help you today?"

#         def send_keep_alive():
#             while agent_running:
#                 time.sleep(5)
#                 connection.send(str(AgentKeepAlive()))

#         threading.Thread(target=send_keep_alive, daemon=True).start()

#         def audio_player():
#             """ Continuously play audio from agent """
#             with sd.OutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
#                 while agent_running:
#                     try:
#                         data = audio_q.get(timeout=1)
#                     except queue.Empty:
#                         continue
#                     if data is None:
#                         break
#                     audio_array = np.frombuffer(data, dtype=np.int16)
#                     stream.write(audio_array)

#         threading.Thread(target=audio_player, daemon=True).start()

#         # Event Handlers

#         def on_audio_data(self, data, **kwargs):
#             audio_q.put(data)

#         def on_conversation_text(self, conversation_text, **kwargs):
#             text = getattr(conversation_text, "text", str(conversation_text))
#             print(f"📝 Agent Text: {text}")

#             # Use the stored main_event_loop for coroutine scheduling
#             if main_event_loop:
#                 asyncio.run_coroutine_threadsafe(transcript_queue.put(text), main_event_loop)
#             else:
#                 print("⚠️ main_event_loop not set, cannot enqueue transcript")

#         def on_agent_started_speaking(self, evt, **kwargs):
#             print("🔊 Agent started speaking")

#         def on_welcome(self, welcome, **kwargs):
#             print("🤝 Connected:", welcome)

#         def on_error(self, error, **kwargs):
#             print("❌ Error:", error)

#         connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
#         connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
#         connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
#         connection.on(AgentWebSocketEvents.Welcome, on_welcome)
#         connection.on(AgentWebSocketEvents.Error, on_error)

#         print("🚀 Starting connection...")
#         if not connection.start(options):
#             print("❌ Failed to connect")
#             agent_running = False
#             return

#         print("✅ Connected to Deepgram Agent")
#         agent_running = True

#         def mic_stream():
#             def callback(indata, frames, time_info, status):
#                 if status:
#                     print("⚠️ Mic status:", status)
#                 if agent_running:
#                     connection.send(indata.tobytes())

#             with sd.InputStream(samplerate=24000, channels=1, dtype="int16", callback=callback):
#                 print("🎤 Speak now! (CTRL+C to exit)")
#                 while agent_running:
#                     time.sleep(0.1)

#         mic_stream()

#     except Exception as e:
#         print(f"❌ Error in agent thread: {e}")

#     finally:
#         agent_running = False
#         audio_q.put(None)
#         if connection:
#             connection.finish()
#         print("👋 Agent stopped.")


# # API endpoints

# @app.post("/start")
# async def start_agent():
#     global agent_thread, agent_running

#     if agent_running:
#         return JSONResponse({"message": "Agent already running"}, status_code=400)

#     agent_running = True
#     agent_thread = threading.Thread(target=deepgram_agent, daemon=True)
#     agent_thread.start()

#     return {"message": "Agent started"}


# @app.post("/stop")
# async def stop_agent():
#     global agent_running, connection

#     if not agent_running:
#         return JSONResponse({"message": "Agent not running"}, status_code=400)

#     agent_running = False

#     # Give some time for the thread to cleanup
#     time.sleep(1)

#     return {"message": "Agent stopped"}


# # WebSocket to send transcript texts

# @app.websocket("/ws/transcripts")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     clients.add(websocket)
#     print("Client connected for transcripts")

#     try:
#         while True:
#             text = await transcript_queue.get()
#             # Broadcast to all connected clients
#             disconnected = []
#             for client in clients:
#                 try:
#                     await client.send_text(text)
#                 except WebSocketDisconnect:
#                     disconnected.append(client)
#             for d in disconnected:
#                 clients.remove(d)

#     except WebSocketDisconnect:
#         print("Client disconnected from transcripts")
#         clients.remove(websocket)
