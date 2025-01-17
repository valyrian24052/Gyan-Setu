from flask import Flask, render_template, request, jsonify
from chatbot import TeacherBot, PERSONALITIES, TONES
from database import log_interaction, get_scenarios, add_scenario
import random

app = Flask(__name__)
bot = TeacherBot()

# Add some initial scenarios if none exist
INITIAL_SCENARIOS = [
    {
        "name": "Science Class - Photosynthesis",
        "description": "Student asking about how plants make their food",
        "expected_response": "Plants make their food through photosynthesis, using sunlight, water, and carbon dioxide to produce glucose and oxygen."
    },
    {
        "name": "Math Class - Fractions",
        "description": "Student confused about adding fractions",
        "expected_response": "To add fractions with different denominators, we first need to find a common denominator. Then we convert each fraction to equivalent fractions with this common denominator before adding."
    }
]

@app.route('/')
def home():
    scenarios = get_scenarios()
    if not scenarios:
        for scenario in INITIAL_SCENARIOS:
            add_scenario(**scenario)
        scenarios = get_scenarios()
    
    return render_template('index.html', 
                         scenarios=scenarios,
                         personalities=PERSONALITIES,
                         tones=TONES)

@app.route('/generate_query', methods=['POST'])
def generate_query():
    data = request.json
    scenario = data.get('scenario', '')
    personality = data.get('personality', random.choice(PERSONALITIES))
    tone = data.get('tone', random.choice(TONES))
    
    query = bot.generate_student_query(scenario, personality, tone)
    return jsonify({'query': query})

@app.route('/evaluate_response', methods=['POST'])
def evaluate_response():
    data = request.json
    teacher_response = data.get('teacher_response', '')
    expected_response = data.get('expected_response', '')
    
    similarity_score = bot.evaluate_response(teacher_response, expected_response)
    feedback = bot.get_feedback(similarity_score, teacher_response, expected_response)
    
    # Log the interaction
    log_interaction(
        scenario=data.get('scenario', ''),
        personality=data.get('personality', ''),
        query=data.get('query', ''),
        teacher_response=teacher_response,
        expected_response=expected_response,
        similarity_score=similarity_score
    )
    
    return jsonify({
        'similarity_score': similarity_score,
        'feedback': feedback
    })

if __name__ == '__main__':
    app.run(debug=True) 