import speech_recognition as sr
import pyttsx3
from ai_backend import get_ai_response

# Function for voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Alpha is listening...")
        speak_response("I'm listening.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            error_msg = "Sorry, I didn't understand that."
            print("Alpha:", error_msg)
            speak_response(error_msg)
            return None
        except sr.RequestError:
            error_msg = "Sorry, speech service is down."
            print("Alpha:", error_msg)
            speak_response(error_msg)
            return None

# Function for voice output
def speak_response(response):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Set female voice if available (usually index 1)
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    engine.say(response)
    engine.runAndWait()

# Main chat function
def chat_with_alpha():
    print("Alpha AI with Location Search and Voice Commands. Say 'exit' to stop.")
    speak_response("Hello, I am Alpha, your AI assistant with location search capabilities.")

    while True:
        user_input = get_voice_input()
        if user_input is None:
            continue
        if user_input.lower() == 'exit':
            speak_response("Goodbye!")
            break

        response = get_ai_response(user_input)

        print("Alpha:", response)
        speak_response(response)

if __name__ == "__main__":
    chat_with_alpha()