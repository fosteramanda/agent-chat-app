# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

from typing import AsyncGenerator, Dict, Optional, Tuple
from quart import Blueprint, jsonify, request, Response, render_template, current_app

import asyncio
import json, os

import os
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential

from azure.ai.projects.models import (
    MessageDeltaChunk,
    ThreadMessage,
    FileSearchTool,
    AsyncToolSet,
    FilePurpose,
    ThreadMessage,
    StreamEventData,
    AsyncAgentEventHandler,
    Agent,
    VectorStore
)

class ChatBlueprint(Blueprint):
    ai_client: AIProjectClient
    agent: Agent
    files: Dict[str, str]
    vector_store: VectorStore

bp = ChatBlueprint("chat", __name__, template_folder="templates", static_folder="static")

class MyEventHandler(AsyncAgentEventHandler[str]):

    async def on_message_delta(
        self, delta: "MessageDeltaChunk" 
    ) -> Optional[str]:
        stream_data = json.dumps({'content': delta.text, 'type': "message"})
        return f"data: {stream_data}\n\n"

    async def on_thread_message(
        self, message: "ThreadMessage" 
    ) -> Optional[str]:
        if message.status == "completed":
            annotations = [annotation.as_dict() for annotation in message.file_citation_annotations]
            stream_data = json.dumps({'content': message.text_messages[0].text.value, 'annotations': annotations, 'type': "completed_message"})
            return f"data: {stream_data}\n\n"
        return None

    async def on_error(self, data: str) -> Optional[str]:
        print(f"An error occurred. Data: {data}")
        stream_data = json.dumps({'type': "stream_end"})
        return f"data: {stream_data}\n\n"

    async def on_done(
        self,
    ) -> Optional[str]:
        stream_data = json.dumps({'type': "stream_end"})
        return f"data: {stream_data}\n\n"



@bp.before_app_serving
async def start_server():
    
    ai_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
    )
    
    # TODO: add more files are not supported for citation at the moment
    file_names = ["product_info_1.md", "product_info_2.md"]
    files: Dict[str, str] = {}
    for file_name in file_names:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files', file_name))
        print(f"Uploading file {file_path}")
        file = await ai_client.agents.upload_file_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
        files.update({file.id: file_path})
    
    vector_store = await ai_client.agents.create_vector_store_and_poll(file_ids=list(files.keys()), name="sample_store")

    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    
    tool_set = AsyncToolSet()
    tool_set.add(file_search_tool)
    
    print(f"ToolResource: {tool_set.resources}")
        
    agent = await ai_client.agents.create_agent(
        model="gpt-4o-mini", name="my-assistant", instructions="You are helpful assistant", tools = tool_set.definitions, tool_resources=tool_set.resources
    )
    
    print(f"Created agent, agent ID: {agent.id}")

    bp.ai_client = ai_client
    bp.agent = agent
    bp.vector_store = vector_store
    bp.files = files
        

@bp.after_app_serving
async def stop_server():
    for file_id in bp.files.keys():
        await bp.ai_client.agents.delete_file(file_id)
        print(f"Deleted file {file_id}")
    
    await bp.ai_client.agents.delete_vector_store(bp.vector_store.id)
    print(f"Deleted vector store {bp.vector_store.id}")
    
    await bp.ai_client.agents.delete_agent(bp.agent.id)

    print(f"Deleted agent {bp.agent.id}")        
    
    await bp.ai_client.close()
    print("Closed AIProjectClient")




@bp.get("/")
async def index():
    return await render_template("index.html")

    

async def get_result(thread_id: str, agent_id: str) -> AsyncGenerator[str, None]:
    async with await bp.ai_client.agents.create_stream(
        thread_id=thread_id, assistant_id=agent_id,
        event_handler=MyEventHandler()
    ) as stream:
        # Iterate over the steam to trigger event functions
        async for _, _, event_func_return_val in stream:
            if event_func_return_val:
                yield event_func_return_val
                
@bp.route('/chat', methods=['POST'])
async def chat():
    thread_id = request.cookies.get('thread_id')
    agent_id = request.cookies.get('agent_id')
    thread = None
    
    if thread_id and agent_id == bp.agent.id:
        # Check if the thread is still active
        try:
            thread = await bp.ai_client.agents.get_thread(thread_id)
        except Exception as e:
            current_app.logger.error(f"Failed to retrieve thread with ID {thread_id}: {e}")
    if thread is None:
        thread = await bp.ai_client.agents.create_thread()    
                    
    thread_id = thread.id
    agent_id = bp.agent.id    
    user_message = await request.get_json()

    if not hasattr(bp, 'ai_client'):
        return jsonify({"error": "Agent is not initialized"}), 500

    message = await bp.ai_client.agents.create_message(
        thread_id=thread.id, role="user", content=user_message['message']
    )
    print(f"Created message, message ID {message.id}")


    # Set necessary headers for SSE
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'text/event-stream'
    }

    response = Response(get_result(thread_id, agent_id), headers=headers)
    response.set_cookie('thread_id', thread_id)
    response.set_cookie('agent_id', agent_id)    
    return response

@bp.route('/fetch-document', methods=['GET'])
async def fetch_document():
    file_id = request.args.get('file_id')
    current_app.logger.info(f"Fetching document: {file_id}")
    if not file_id:
        return jsonify({"error": "file_id is required"}), 400

    try:
        # Read the file content asynchronously using asyncio.to_thread
        data = await asyncio.to_thread(read_file, bp.files[file_id])
        return Response(data, content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def read_file(path):
    with open(path, 'r') as file:
        return file.read()
