# LMStudio-Chat
Remote Chat with LM Studio in a local network

# Remote PC - LM Studio Server
cd LM_Studio_LMS_CLI_Server/
source LM_CLI_Server/bin/activate
python3 lmstudio_server.py

Start LM Studio 
Start Server at Developer Tab
Serve on Local Network

# Client PC - Client
python3 lmstudio_chat_v04.py

# Unloading Models
For unloading models remotely there is no REST API natively - lmstudio_server.py does that and it fetches Model Data 
