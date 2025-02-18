from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatHistory
import random

# Example responses
responses = {
    "hello": "Hi! How can I assist you today?",
    "symptoms": "Can you describe your symptoms?",
    "fever": "If you have a fever, make sure to stay hydrated and rest. If severe, consult a doctor.",
    "default": "I'm not sure about that. Please consult a medical professional."
}

def chatbot_response(user_message):
    return responses.get(user_message.lower(), responses["default"])

def chat(request):
    if request.method == "POST":
        user_message = request.POST.get('message', '')
        response = chatbot_response(user_message)

        # Save to database
        chat_entry = ChatHistory.objects.create(user_input=user_message, bot_response=response)

        return JsonResponse({"response": response})

    return render(request, "chatbot/chat.html")
