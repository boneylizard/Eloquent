# test_memory.py
import asyncio
from fastapi.testclient import TestClient
from memory_agent import memory_router
from fastapi import FastAPI

# Create a simple test app
app = FastAPI()
app.include_router(memory_router)

client = TestClient(app)

def test_memory_operations():
    # Test adding a memory
    memory = {
        "content": "User prefers technical explanations with code examples",
        "category": "preferences",
        "importance": 0.8,
        "type": "auto"
    }
    
    response = client.post("/memory/add", json=memory)
    print("Add memory response:", response.json())
    
    # Test listing memories
    response = client.get("/memory/list")
    print("List memories response:", response.json())
    
    # Test enhancing a prompt
    prompt = "Can you explain how to implement a binary search tree?"
    response = client.post("/memory/enhance", json={"prompt": prompt})
    print("Enhanced prompt:", response.json())
    
    # Test observing a conversation
    conversation = {
        "prompt": "I prefer explanations with diagrams whenever possible",
        "response": "I understand. I'll include diagrams in my explanations."
    }
    response = client.post("/memory/observe", json=conversation)
    print("Observation result:", response.json())
    
    # Check if the new memory was added
    response = client.get("/memory/list")
    print("Updated memories:", response.json())

if __name__ == "__main__":
    test_memory_operations()