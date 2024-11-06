from flask import Flask, render_template, request, jsonify
from api_recommender import APIRecommender

app = Flask(__name__)

# 初始化API推荐器
ZHIPUAI_KEY = "02dc99f44be91fc5e6345ee2e90b6a27.S3AlhwR7tFNpnlcK"
PINECONE_KEY = "pcsk_7KKVnM_TyrTke9bYMbGr3z4PTPXJwmThqQD7SDcUw3VM4HmnFoxyqBfgCfpjweDiFX9yZC"
recommender = APIRecommender(ZHIPUAI_KEY, PINECONE_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({'error': '请输入查询内容'})
    
    # 获取API推荐
    recommendations = recommender.get_api_recommendations(user_query)
    
    # 获取解释性文本
    response = recommender.stream_chat_response(user_query, recommendations)
    explanation = ""
    if response:
        for chunk in response:
            if chunk.choices[0].delta.content:
                explanation += chunk.choices[0].delta.content
    
    return jsonify({
        'recommendations': recommendations,
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=True) 