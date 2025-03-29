class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };
    }

    init() {
        this.display();
        this.addWelcomeMessage();
    }

    display() {
        const { sendButton } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton());
        
        const inputField = this.args.chatBox.querySelector('input');
        inputField.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton();
            }
        });
    }

    addWelcomeMessage() {
        this.addMessage("Hello! I'm the NITJ chatbot. How can I help you today?", false);
    }

    async onSendButton() {
        const input = this.args.chatBox.querySelector('input');
        const text = input.value.trim();
        
        if (!text) return;

        this.addMessage(text, true);
        input.value = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) throw new Error(await response.text());
            
            const data = await response.json();
            
            const formattedAnswer = data.answer
                .replace(/\n/g, '<br>')
                .replace(/More info: (https?:\/\/[^\s]+)/g, 
                    '<a href="$1" target="_blank" class="chat-link">More info →</a>');
            
            this.addMessage(formattedAnswer, false);
            
        } catch (error) {
            console.error('Error:', error);
            this.addMessage("Sorry, there was an error. Please try again.", false);
        }
    }

    addMessage(text, isUser) {
        const div = document.createElement('div');
        div.classList.add('messages__item');
        // Reversed these classes:
        div.classList.add(isUser ? 'messages__item--visitor' : 'messages__item--operator');
        div.innerHTML = text;
        
        const messagesContainer = document.querySelector('.chatbox__messages');
        messagesContainer.insertBefore(div, messagesContainer.firstChild);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

const chatbox = new Chatbox();
chatbox.init();