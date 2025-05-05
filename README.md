# NeuralNexus - AI Chatbot

NeuralNexus is an intelligent chatbot powered by TensorFlow and Natural Language Processing (NLP). It uses a neural network to understand and respond to user queries with context awareness and personality.

## Features

- 🤖 Natural Language Understanding
- 🧠 Neural Network-based Response Generation
- 🔄 Context-Aware Conversations
- 💬 Modern Web Interface
- 🎨 Beautiful UI with Animations
- 📱 Responsive Design
- ⚡ Real-time Processing

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tanishpoddar/NeuralNexus.git
cd NeuralNexus
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
NeuralNexus/
├── main-host.py          # Main application file
├── intents.json          # Training data and responses
├── model.weights.h5      # Trained model weights
├── data.pickle           # Processed training data
├── templates/
│   └── index.html        # Web interface
└── requirements.txt      # Python dependencies
```

## Required Packages

- tensorflow==2.15.0
- numpy==1.24.3
- nltk==3.8.1
- flask==3.0.0
- python-dotenv==1.0.0

## Running the Application

1. Ensure you're in the virtual environment
2. Run the Flask application:
```bash
python main-host.py
```
3. Open your browser and navigate to:
```
http://localhost:5000
```

## Training the Model

The model is automatically trained when you run the application. It:
1. Processes the intents from `intents.json`
2. Creates a bag-of-words representation
3. Trains a neural network with:
   - Input layer: Size based on vocabulary
   - Hidden layers: Two layers with 16 neurons each
   - Dropout layers: To prevent overfitting
   - Output layer: Size based on number of intents

## Customization

### Adding New Intents
Edit `intents.json` to add new conversation patterns and responses:
```json
{
    "tag": "new_intent",
    "patterns": ["pattern1", "pattern2"],
    "responses": ["response1", "response2"]
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all packages are installed correctly
   - Check Python version compatibility
   - Verify virtual environment activation

2. **Model Training Issues**
   - Delete `model.weights.h5` and `data.pickle`
   - Restart the application to retrain

3. **Web Interface Issues**
   - Clear browser cache
   - Check console for JavaScript errors
   - Verify Flask server is running

### Error Messages

- "ModuleNotFoundError": Install missing packages
- "TypeError": Check input format in intents.json
- "ValueError": Verify model architecture matches data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ♥ by [Tanish Poddar](https://github.com/tanish-poddar)
